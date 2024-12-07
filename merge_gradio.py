from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/llama3_test/LLM-Research/Meta-Llama-3.1-8B-Instruct'
lora_path = '/root/autodl-tmp/llama3_test/output/llama3_1_instruct_lora/checkpoint-100' 

tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "你是谁？"

messages = [
        {"role": "system", "content": "假设你是一位医生，现在正在与患者进行对话，请根据对话内容，回答问题。"},
        {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
