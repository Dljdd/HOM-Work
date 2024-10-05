import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from diffusers import DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import os

def preprocess_function(examples, tokenizer):
    images = [Image.open(image_file).convert("RGB") for image_file in examples["image"]]
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    examples["pixel_values"] = [transform(image) for image in images]
    examples["input_ids"] = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length).input_ids
    return examples

def run_job(job_config, model, dataset, accelerator):
    # Set up the UNet for LoRA training
    unet = model.unet
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=4,
        )

    unet.set_attn_processor(lora_attn_procs)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=job_config['job']['steps'][0]['params']['train']['lr'])

    # Set up the noise scheduler
    noise_scheduler = DDIMScheduler.from_pretrained(job_config['job']['steps'][0]['params']['model_path'], subfolder="scheduler")

    # Training loop
    num_update_steps_per_epoch = len(dataset)
    num_train_epochs = job_config['job']['steps'][0]['params']['train']['steps'] // num_update_steps_per_epoch

    progress_bar = tqdm(range(num_train_epochs))
    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(dataset):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = model.vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = model.text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % job_config['job']['steps'][0]['params']['sample']['sample_every'] == 0:
                generate_sample_images(unet, model.text_encoder, model.vae, job_config)

        progress_bar.update(1)

    # Save the trained LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_attn_procs(job_config['job']['steps'][0]['params']['name'])

def generate_sample_images(unet, text_encoder, vae, job_config):
    # Set models to evaluation mode
    unet.eval()
    text_encoder.eval()
    vae.eval()

    # Get sampling parameters from config
    sample_config = job_config['job']['steps'][0]['params']['sample']
    prompts = sample_config['prompts']
    height = sample_config['height']
    width = sample_config['width']
    num_inference_steps = sample_config['sample_steps']
    guidance_scale = sample_config['guidance_scale']
    
    # Initialize scheduler
    scheduler = DDIMScheduler.from_pretrained(job_config['job']['steps'][0]['params']['model_path'], subfolder="scheduler")
    
    # Tokenize prompts
    tokenizer = CLIPTokenizer.from_pretrained(job_config['job']['steps'][0]['params']['model_path'], subfolder="tokenizer")
    text_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]
    
    # Prepare uncond embeddings for classifier free guidance
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * len(prompts), padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(unet.device))[0]
    
    # Concatenate the unconditional and text embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Generate random latent noise
    latents = torch.randn((len(prompts), unet.in_channels, height // 8, width // 8))
    latents = latents.to(unet.device)
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps)
    
    # Denoising loop
    for t in scheduler.timesteps:
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    
    # Convert to PIL images
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    
    # Save images
    output_dir = os.path.join(job_config['job']['steps'][0]['params']['training_folder'], 'samples')
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(pil_images):
        image.save(f"{output_dir}/sample_{i}.png")

    print(f"Generated {len(pil_images)} sample images in {output_dir}")