#!/bin/bash

set -e  # Exit on error

# Usage function
usage() {
    echo "Usage: ./convert_to_gguf.sh <model_name> [quantization_type] [test_prompt]"
    echo ""
    echo "Arguments:"
    echo "  model_name         Name for your model (e.g., gemma-3-1b-finetuned)"
    echo "  quantization_type  (Optional) Quantization type (default: Q4_K_M)"
    echo "                     Options: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q8_0"
    echo "  test_prompt        (Optional) Custom test prompt (default: 'What is 2 plus 3?')"
    echo ""
    echo "Examples:"
    echo "  ./convert_to_gguf.sh gemma-3-1b-finetuned"
    echo "  ./convert_to_gguf.sh gemma-3-1b-finetuned Q4_K_M"
    echo "  ./convert_to_gguf.sh gemma-3-1b-finetuned Q4_K_M \"Your custom prompt here\""
    echo ""
    echo "This script will:"
    echo "  1. Convert outputs/merged_model to FP16 GGUF"
    echo "  2. Quantize to specified type"
    echo "  3. Test the quantized model with your prompt"
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

MODEL_NAME="$1"
QUANT_TYPE="${2:-Q4_K_M}"  # Default to Q4_K_M
TEST_PROMPT="${3:-What is 2 plus 3?}"  # Default prompt

# Paths
MERGED_MODEL="outputs/merged_model"
FP16_OUTPUT="outputs/${MODEL_NAME}-fp16.gguf"
QUANT_OUTPUT="outputs/finalized_models/${MODEL_NAME}-${QUANT_TYPE}.gguf"

# Validate merged model exists
if [ ! -d "$MERGED_MODEL" ]; then
    echo "❌ Error: Merged model not found at '$MERGED_MODEL'"
    echo "   Run your fine-tuning notebook first to create the merged model"
    exit 1
fi

# Validate llama.cpp is set up
if [ ! -f "llama.cpp/convert_hf_to_gguf.py" ]; then
    echo "❌ Error: llama.cpp/convert_hf_to_gguf.py not found"
    echo "   Make sure llama.cpp is cloned and set up"
    exit 1
fi

if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "❌ Error: llama-quantize not found"
    echo "   Build llama.cpp first:"
    echo "   cd llama.cpp && cmake -B build && cmake --build build --config Release"
    exit 1
fi

# Create output directories
mkdir -p outputs/finalized_models

echo "=========================================="
echo "🚀 Starting GGUF Conversion Pipeline"
echo "=========================================="
echo "Model name: $MODEL_NAME"
echo "Quantization: $QUANT_TYPE"
echo "Test prompt: $TEST_PROMPT"
echo ""

# Step 1: Convert to FP16 GGUF
echo "🔄 [1/3] Converting to FP16 GGUF..."
python llama.cpp/convert_hf_to_gguf.py "$MERGED_MODEL" \
    --outfile "$FP16_OUTPUT" \
    --outtype f16

if [ ! -f "$FP16_OUTPUT" ]; then
    echo "❌ Error: FP16 conversion failed"
    exit 1
fi

echo "✅ FP16 GGUF created: $FP16_OUTPUT"
echo ""

# Step 2: Quantize
echo "🔄 [2/3] Quantizing to $QUANT_TYPE..."
./llama.cpp/build/bin/llama-quantize \
    "$FP16_OUTPUT" \
    "$QUANT_OUTPUT" \
    "$QUANT_TYPE"

if [ ! -f "$QUANT_OUTPUT" ]; then
    echo "❌ Error: Quantization failed"
    exit 1
fi

echo "✅ Quantized model created: $QUANT_OUTPUT"
echo ""

# Step 3: Test the model
echo "🔄 [3/3] Testing quantized model..."
echo "Running test prompt..."
echo ""

./llama.cpp/build/bin/llama-cli \
    -m "$QUANT_OUTPUT" \
    -p "$TEST_PROMPT" \
    -n 256 \
    --temp 0.7 \
    --repeat-penalty 1.1

echo ""
echo "=========================================="
echo "✅ Conversion Complete!"
echo "=========================================="
echo "Your model is ready at: $QUANT_OUTPUT"
echo ""
echo "File sizes:"
ls -lh "$FP16_OUTPUT" "$QUANT_OUTPUT"