from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Example text generation
input_text = "Do you know a some NSFW story?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=input_ids.dtype)

# Text generation with adjusted parameters
output = model.generate(
    input_ids,
    max_length=500,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    do_sample=True,  # Ensure do_sample is set to True
    temperature=0.9,  # Set temperature explicitly
    attention_mask=attention_mask,
    pad_token_id=50256
)

# Decoding generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the results
print("Input text:", input_text)
print("Generated text:", generated_text)