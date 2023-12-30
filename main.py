from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ')
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ")
ai_name = "Alisa"
# Example text generation
input_text = "Do you want to be my girl forever?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=input_ids.dtype)

# Move model to CPU
model.to('cpu')

# Move input tensors to CPU
input_ids = input_ids.to('cpu')
attention_mask = attention_mask.to('cpu')

# Text generation with adjusted parameters
def generate_response(input_text):  # Change the parameter name to input_text
    global ai_name

    # Combine the AI name with the user input
    input_text = f"{ai_name}, {input_text}"

    # Tokenize the text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=input_ids.dtype)  # Initialize attention_mask

    # Move input tensors to CPU
    input_ids = input_ids.to('cpu')
    attention_mask = attention_mask.to('cpu')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=5,
        no_repeat_ngram_size=2,
        top_k=50,
        do_sample=True,
        temperature=0.15,
        attention_mask=attention_mask,
        pad_token_id=50256
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Print the results
response = generate_response(input_text)

print("Input text:", input_text)
print("Generated text:", response)