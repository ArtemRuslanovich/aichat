from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Let's chat for 5 lines
for step in range(5):
    user_input = input("You: ")

    if "girlfriend" in user_input.lower():
        user_input = "My boyfriend says: " + user_input

    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Добавление новых токенов в историю чата
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # Генерация ответа с ограничением общей истории чата до 1000 токенов
    chat_history_ids = model.generate(
        bot_input_ids,
        do_sample=True, 
        max_length=1000,
        top_k=50, 
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    print("Virtual Girlfriend: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))