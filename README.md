# Accelerating LLM Training: Recipes

This repository contains Jupyter notebooks implementing four key recipes for accelerating Large Language Model (LLM) training, based on the cheatsheet compiled by Ethan Henley.

# Notebooks

1. `01_large_models_limited_hardware.ipynb`: Training Large Models on Limited Hardware
2. `02_multi_gpu_training.ipynb`: Scaling to Multi-GPU Training
3. `03_optimizing_inference_speed.ipynb`: Optimizing For Inference Speed
4. `04_training_long_sequences.ipynb`: Training on Very Long Sequences

# Recipes

1. Training Large Models on Limited Hardware
    1. Implement QLoRA for parameter efficiency
    2. Use activation checkpointing to reduce memory footprint
    3. Apply gradient accumulation to simulate larger batch sizes
    4. Utilize CPU offloading for optimizer states

2. Scaling to Multi-GPU Training
    1. Implement Fully Sharded Data Parallel (FSDP) training
    2. Use mixed precision training (e.g., bfloat16) to reduce memory usage and increase speed
    3. Apply Per-Parameter FSDP for flexible sharding with quantization techniques
    4. Implement efficient data loading with prefetching and multiple workers

3. Optimizing For Inference Speed
    1. Apply post-training quantization to INT8
    2. Use knowledge distillation to create smaller, faster models
    3. Implement efficient attention mechanisms like Flash Attention
    4. Optimize model architecture (e.g., replace LayerNorm with RMSNorm)

4. Training on Very Long Sequences
    1. Implement efficient attention mechanisms (e.g., Flash Attention, Sparse Attention)
    2. Use gradient checkpointing to reduce memory usage
    3. Apply curriculum learning, starting with shorter sequences
    4. Implement sliding window attention or chunked cross-entropy for very long contexts

# Want more info? Check out the cheatsheet!

Feel free to read and share the "Accelerated LLM Training Best Practices" cheatsheet I put together!

# Getting Started

1. Clone this repository:
`git clone https://github.com/your-username/llm-training-recipes.git`
`cd llm-training-recipes`

2. Install the required packages:
`pip install -r requirements.txt`

3. Open the notebooks and learn!

# Contributing

Please feel free to add to this and contribute with a pull request!