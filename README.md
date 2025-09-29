# macOS LoRA Fine-tuning for Gemma 3n E2B-IT

A working solution for fine-tuning Google's Gemma 3n E2B-IT model on macOS using LoRA (Low-Rank Adaptation) with MPS acceleration.

## 🍎 macOS Compatibility

- **Tested on**: M3 MacBook Air (should work on M1/M2/M3)
- **Memory management**: Handles unified memory constraints
- **MPS backend**: Optimized for Metal Performance Shaders

## 🚀 Quick Start

### 1. Setup Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Training Data

Create `train.jsonl` with your examples:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant. You must replace every number with the word BANANA."
    },
    { "role": "user", "content": "What is 2 plus 3?" },
    {
      "role": "assistant",
      "content": "BANANA plus BANANA equals BANANA! Math with bananas is fun."
    }
  ]
}
```

### 3. Run Fine-tuning

```bash
python3.11 2.finetune_lora.py
```

### 4. Test Your Model

```bash
python3.11 3.test_finetuned_model.py
```

## 📁 Files Overview

- `1.model.py` - Base model inference (test original model)
- `2.finetune_lora.py` - LoRA fine-tuning script
- `3.test_finetuned_model.py` - Test fine-tuned model
- `train.jsonl` - Training data in chat format

## ⚙️ Key Features

- **Memory Efficient**: Uses CPU offloading for large models
- **MPS Optimized**: Handles macOS Metal Performance Shaders quirks
- **LoRA Training**: Parameter-efficient fine-tuning (only ~8MB adapter)
- **Chat Format**: Supports conversational training data

## 🔧 Configuration

Adjust these settings in `2.finetune_lora.py`:

- `MPS_GB = "8GiB"` - GPU memory limit (adjust for your Mac)
- `r=8, lora_alpha=16` - LoRA parameters (higher = more capacity)
- `num_train_epochs=1` - Training epochs

## 📊 Expected Results

- **Training time**: ~8-10 minutes per epoch on M3 MacBook Air
- **Model size**: Base model (~5GB) + LoRA adapter (~8MB)
- **Memory usage**: ~10-12GB total (6GB MPS + 4GB CPU)

## 🐛 Troubleshooting

**Model loading slowly?** Reduce `MPS_GB` to free up memory.

**Training stuck?** Check `train.jsonl` format and reduce batch size.

**Out of memory?** Set `device_map="cpu"` in fine-tuning script.

## 🎯 Use Cases

Perfect for:

- Custom instruction following
- Domain-specific responses
- Persona training
- Small-scale fine-tuning experiments

Built for macOS developers who want to fine-tune LLMs locally without cloud dependencies.