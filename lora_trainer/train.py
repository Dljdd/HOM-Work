import os
import sys
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from utils import preprocess_function, run_job
import yaml
from collections import OrderedDict
from huggingface_hub import hf_hub_download

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # Adjust this number based on your CPU cores

try:
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set up the environment
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Ask for dataset path
    dataset_path = input("Enter the path to your dataset (default: /app/dataset): ") or "/app/dataset"
    print(f"Using dataset path: {dataset_path}")

    # Set up the model components
    print("Loading model components...")
    model_path = config['model']['name_or_path']

    print("Loading VAE...")
    vae_path = hf_hub_download(repo_id=model_path, filename="flux1-dev.safetensors")
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", use_safetensors=True, torch_dtype=torch.float32)
    print("VAE loaded successfully")

    print("Loading UNet...")
    unet_path = hf_hub_download(repo_id=model_path, filename="flux1-dev.safetensors")
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", use_safetensors=True, torch_dtype=torch.float32)
    print("UNet loaded successfully")

    print("Loading text encoder...")
    text_encoder_path = hf_hub_download(repo_id=model_path, filename="flux1-dev.safetensors")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, subfolder="text_encoder", use_safetensors=True, torch_dtype=torch.float32)
    print("Text encoder loaded successfully")

    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    print("Tokenizer loaded successfully")

    print("Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    print("Scheduler loaded successfully")

    model = {
        "vae": vae,
        "unet": unet,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler
    }

    print("All model components loaded successfully")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("imagefolder", data_dir=dataset_path)
    print("Dataset loaded successfully")

    print("Processing dataset...")
    processed_dataset = dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    print("Dataset processed successfully")

    # Set up training
    print("Setting up accelerator...")
    accelerator = Accelerator()
    model, processed_dataset = accelerator.prepare(model, processed_dataset)
    print("Accelerator setup complete")

    # Create job configuration
    job_to_run = OrderedDict([
        ('job', OrderedDict([
            ('name', config['meta']['name']),
            ('steps', [
                OrderedDict([
                    ('type', 'train_lora'),
                    ('params', OrderedDict([
                        ('name', config['meta']['name']),
                        ('model_path', config['model']['name_or_path']),
                        ('dataset', OrderedDict([
                            ('path', dataset_path),
                            ('caption_column', "text"),
                            ('image_column', "image"),
                        ])),
                        ('train', config['train']),
                        ('sample', config['sample']),
                    ]))
                ])
            ])
        ])),
        ('meta', OrderedDict([
            ('name', config['meta']['name']),
            ('version', config['meta']['version'])
        ]))
    ])

    # Run the training job
    print("Starting training job...")
    run_job(job_to_run, model, processed_dataset, accelerator)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()