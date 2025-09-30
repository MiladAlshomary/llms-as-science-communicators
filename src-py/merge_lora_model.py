from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model  = 'meta-llama/Meta-Llama-3-8B-Instruct'
lora_path   = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-llama3-trained-journalist-on-deepseek-3epochs-final/'
output_path = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-llama3-trained-journalist-on-deepseek-3epochs-final-full-model/'

# base_model  = 'Qwen/Qwen2.5-7B-Instruct'
# lora_path   = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-qwen-trained-journalist-on-deepseek-3epochs-high-capacity/'
# output_path = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-qwen-trained-journalist-on-deepseek-3epochs-full-model/'


# Load base + LoRA
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
model = PeftModel.from_pretrained(model, lora_path)

# Merge LoRA into base weights
model = model.merge_and_unload()

# Save merged model to disk
model.save_pretrained(output_path)

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_path)
