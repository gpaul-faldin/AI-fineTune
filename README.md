# AI Fine-Tuning Project

This repository contains code for fine-tuning the Microsoft Phi-2 language model using Low-Rank Adaptation (LoRA). The project is designed to efficiently adapt the model with minimal additional parameters while maintaining strong performance.

## Overview

This project implements a complete AI fine-tuning pipeline for the Phi-2 model, with support for multiple GPU configurations (A6000, RTX 3090, RTX 2080). The code is optimized for efficient training and includes utilities for testing and evaluating the resulting fine-tuned model.

## Why Fine-Tuning?

Fine-tuning allows you to:
- Adapt a pre-trained model to specific tasks or domains
- Improve performance on targeted use cases
- Create specialized AI capabilities without training from scratch
- Significantly reduce computational requirements compared to full model training

## What is LoRA?

LoRA (Low-Rank Adaptation) is a technique that makes fine-tuning large language models much more efficient:

### How LoRA Works (Simplified)
1. Instead of updating all parameters in a large model (which can be billions of parameters)
2. LoRA adds small "adapter" matrices to specific layers of the model
3. Only these adapter parameters are trained, keeping the original model frozen
4. This dramatically reduces memory usage and training time

### Benefits for Beginners
- **Resource Efficient**: Train on consumer GPUs with limited VRAM
- **Storage Efficient**: A fine-tuned LoRA adapter might be only ~10-100MB vs. several GB for a full model
- **Faster Training**: Complete fine-tuning in hours instead of days
- **Modular**: Create multiple LoRA adapters for different tasks using the same base model

## Requirements

- Python 3.8+
- CUDA-compatible GPU (RTX 2080, 3090, or A6000 recommended)
- PyTorch 2.0+
- Transformers 4.37.2+
- PEFT 0.7.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-fineTune.git
cd AI-fineTune
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

## Project Structure

```
AI-fineTune/
├── main.py                 # Entry point for training
├── requirement.txt         # Dependencies
├── dataset/                # Place your training data here
│   └── modern.json         # Example dataset (not included)
├── src/
│   ├── config.py           # Configuration settings
│   ├── data_processing.py  # Data handling utilities
│   ├── loraChecker.py      # Model testing and comparison
│   ├── model.py            # Model initialization
│   └── train.py            # Training logic
└── outputs/                # Generated during training (not included)
    └── final/              # Final model outputs
```

## Step-by-Step Guide for Beginners

### 1. Understand the Process

Fine-tuning a model involves:
1. **Preparation**: Set up your environment and data
2. **Configuration**: Choose settings based on your hardware and task
3. **Training**: Run the fine-tuning process
4. **Evaluation**: Test how well your model performs
5. **Deployment**: Use your fine-tuned model

### 2. Choose Your Task

Before starting, decide what you want the model to learn:
- Answer questions in a specific style?
- Generate specialized content?
- Learn particular domain knowledge?

Your task will determine how you prepare your training data.

## Usage

### Prepare Dataset

Your dataset should be structured as a JSON file with conversational format. Place it in the `dataset/` directory.

Example format:
```json
{
  "conversations": [
    {
      "messages": [
        {"role": "user", "content": "Question text"},
        {"role": "assistant", "content": "Answer text"}
      ]
    }
  ]
}
```

### Configure GPU Settings

Edit the `gpu` variable in `main.py` to match your GPU:

```python
# Options: 'rtx2080', '3090', 'A6000'
gpu = 'rtx2080'  
```

### Run Training

```bash
python main.py
```

The script will automatically:
1. Detect your GPU and confirm available memory
2. Load appropriate configurations
3. Initialize the Phi-2 model with LoRA adaptations
4. Process and prepare your training data
5. Train the model with optimized settings
6. Save the final model to the `outputs/final` directory

### Test Fine-Tuned Model

After training, you can test your model using the `loraChecker.py` script:

```bash
python -m src.loraChecker
```

This will:
1. Load both the original model and your fine-tuned version
2. Generate responses to test prompts with different temperatures
3. Compare outputs between the base and fine-tuned models
4. Save results to a JSON file for analysis

## Configuration Details

The project includes specialized configurations for different GPUs:

- **A6000 (48GB VRAM)**: Largest batch size, fastest training
- **RTX 3090 (24GB VRAM)**: Balanced configuration
- **RTX 2080 (8GB VRAM)**: Smaller batch size with gradient accumulation

## Understanding LoRA Settings

LoRA settings can be adjusted in `src/config.py`. Here's what each parameter means and how to choose values:

### Key LoRA Parameters Explained

- **`r` (Rank)**: Controls the complexity of the adaptation. 
  - Range: Typically 4-64
  - Lower values (4-16): More efficient, less expressive
  - Higher values (32-64): More expressive, but larger and slower to train
  - Recommendation for beginners: Start with 8 or 16

- **`lora_alpha` (Alpha)**: Scaling factor that affects how much influence the LoRA adapters have.
  - Typically set to 2× or 1× the rank value
  - Higher values give LoRA adaptations more weight
  - Recommendation: Start with `lora_alpha` = 2 × `r`

- **`lora_dropout` (Dropout)**: Helps prevent overfitting during training.
  - Range: 0.0 (no dropout) to 0.1 (significant dropout)
  - Smaller datasets often benefit from higher dropout
  - Recommendation: 0.05 for most use cases

- **`target_modules` (Target Layers)**: Which model layers to modify with LoRA.
  - For attention-based models like Phi-2, typically attention layers and feed-forward networks
  - This project targets the following layers for Phi-2:
    ```python
    target_modules = [  
        "Wqkv",      # Attention weights
        "out_proj",  # Attention output projection
        "fc1",       # First feed-forward layer
        "fc2",       # Second feed-forward layer
    ]
    ```

### Guidelines for Beginners

- **Start Small**: Begin with `r=8` and `lora_alpha=16`
- **Memory Issues**: If you encounter GPU out-of-memory errors, reduce `r` to 4
- **Poor Results**: If the model isn't learning well, try increasing `r` to 16 or 32
- **Don't Change `target_modules`**: The default settings are optimized for Phi-2

### Example: Adjusting Settings for Different Scenarios

```python
# For limited GPU memory (e.g., 6GB VRAM)
@dataclass
class LimitedMemoryLoRAConfig:
    r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    
# For complex tasks requiring more adaptation
@dataclass
class ComplexTaskLoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
```

## Cloud-Based Fine-Tuning

For users without access to powerful GPUs, I recommend using [QBlocks Cloud](https://www.qblocks.cloud/invite/qb-f7a82c), which offers:

- On-demand GPU access (including A100, H100)
- Pre-configured environments for AI workloads
- Pay-as-you-go pricing model
- Simple deployment and management
- Significantly faster training times than consumer hardware

Use my referral link to get started: [https://www.qblocks.cloud/invite/qb-f7a82c](https://www.qblocks.cloud/invite/qb-f7a82c)

## Troubleshooting Common Issues

### Out of Memory Errors
- Reduce batch size in the configuration file
- Lower the LoRA rank (`r` value)
- Try a smaller model or use 4-bit quantization (already enabled in this project)

### Poor Performance
- Check your dataset quality and size (typically need 100+ examples)
- Increase training epochs
- Adjust the learning rate (try 1e-5 to 5e-5)
- Experiment with different LoRA rank values

### Slow Training
- Enable fp16 training if your GPU supports it (already configured)
- Use gradient accumulation for effective batch size increase
- Consider using QBlocks Cloud for faster training

## Additional Resources

- [Microsoft Phi-2 Model](https://huggingface.co/microsoft/phi-2)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (original paper)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) (library used in this project)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [LoRA for Beginners](https://huggingface.co/blog/lora) (Hugging Face blog)
- [Fine-tuning LLMs with PEFT](https://www.philschmid.de/fine-tune-llms-with-peft) (tutorial)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
