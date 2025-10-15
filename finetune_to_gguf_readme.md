# Convert the merged model to GGUF

python llama.cpp/convert_hf_to_gguf.py outputs/merged_model \
 --outfile outputs/gemma-3n-finetuned-fp16.gguf \
 --outtype f16

# Quantize to Q4_K_M (recommended - good balance)

./llama.cpp/build/bin/llama-quantize \
 outputs/gemma-3n-finetuned-fp16.gguf \
 outputs/gemma-3n-finetuned-Q4_K_M.gguf \
 Q4_K_M

# Then test it

./llama.cpp/build/bin/llama-cli \
 -m outputs/gemma-3n-finetuned-Q4_K_M.gguf \
 -p "What is 2 plus 3?" \
 -n 50
