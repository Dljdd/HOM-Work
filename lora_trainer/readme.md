# LoRA Trainer for FLUX.1

This repository contains a LoRA (Low-Rank Adaptation) trainer for the FLUX.1 model.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your dataset and update the path in `train.py`.

3. Adjust the configuration in `config.yaml` as needed.

## Usage

Run the training script:

## Docker Usage

1. Build the Docker image:
   ```
   docker build -t lora-trainer .
   ```

2. Run the Docker container:
   ```
   docker run -it --gpus all -v /path/to/your/dataset:/app/dataset lora-trainer
   ```

   Replace `/path/to/your/dataset` with the actual path to your dataset on the host machine.

3. When prompted, enter the dataset path as `/app/dataset`.

Note: Make sure you have Docker and NVIDIA Container Toolkit installed for GPU support.
