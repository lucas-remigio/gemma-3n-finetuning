from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading your finetuned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, "outputs/lora")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("outputs/merged_model")
tokenizer = AutoTokenizer.from_pretrained("outputs/lora")
tokenizer.save_pretrained("outputs/merged_model")

print("Done! Now convert to GGUF using llama.cpp")