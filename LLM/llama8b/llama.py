import transformers
import torch
from transformers import AutoTokenizer

model = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
sequences = pipeline(
    'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation = True,
    max_length=400,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
