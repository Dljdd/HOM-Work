import os
import sys
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from utils import preprocess_function, run_job
import yaml
from collections import OrderedDict

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up the environment
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Ask for dataset path
dataset_path = input("Enter the path to your dataset (default: /app/dataset): ") or "/app/dataset"

# Set up the model and dataset
model = DiffusionPipeline.from_pretrained(
    config['model']['name_or_path'],
    use_safetensors=True,
    torch_dtype=torch.float16 if config['train']['dtype'] == 'fp16' else torch.bfloat16
)
tokenizer = CLIPTokenizer.from_pretrained(config['model']['name_or_path'], subfolder="tokenizer")

# Load dataset
dataset = load_dataset("imagefolder", data_dir=dataset_path)
processed_dataset = dataset["train"].map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Set up training
accelerator = Accelerator()
model, processed_dataset = accelerator.prepare(model, processed_dataset)

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
run_job(job_to_run, model, processed_dataset, accelerator)