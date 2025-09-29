import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("🚀 Simple CPU-only test...")

# Load everything on CPU for simplicity
print("[1] Loading base model (CPU only)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

print("[2] Loading LoRA...")
model = PeftModel.from_pretrained(base_model, "outputs/lora")
model = model.merge_and_unload()

print("[3] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("outputs/lora")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("[4] Testing...")
# Use the SAME format as your training data
messages = [
    {"role": "system", "content": "You are a helpful assistant. You must replace every number with the word BANANA."},
    {"role": "user", "content": "What is 2 plus 3?"}
]

inputs = tok.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    return_tensors="pt"
)

print("🎯 Generating (this should be faster on CPU)...")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=20,        # Very short for quick test
        do_sample=False,          # Greedy = faster
        use_cache=True,
        pad_token_id=tok.pad_token_id,
    )

response = tok.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*50)
print("📝 FULL RESPONSE:")
print(response)
print("="*50)

# Extract just the generated part
input_text = tok.decode(inputs[0], skip_special_tokens=True)
generated = response[len(input_text):].strip()
print(f"🤖 GENERATED ONLY: '{generated}'")

# Check for BANANA
if "BANANA" in generated.upper():
    print("✅ SUCCESS: Found BANANA!")
else:
    print("❌ No BANANA found")