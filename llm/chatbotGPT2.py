import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------
# 1️⃣ Load Pretrained Model & Tokenizer
# ---------------------------
MODEL_NAME = "distilgpt2"  # Lightweight GPT-2 model
device = torch.device("cpu")  # Force CPU instead of CUDA

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ---------------------------
# 2️⃣ Generate Chatbot Response
# ---------------------------
def generate_response(prompt, max_length=50, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Create an attention mask: 1 for real tokens, 0 for padding
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,  # Set EOS as PAD token
            attention_mask=attention_mask,  # FIXED: Now properly defined!
            use_cache=False  # Reduce memory usage
        )

    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()


# ---------------------------
# 3️⃣ Chat Loop
# ---------------------------
def chat():
    print("🤖 Transformer Chatbot (type 'exit' to quit)")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! 👋")
            break

        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# ---------------------------
# 4️⃣ Start Chatbot
# ---------------------------
if __name__ == "__main__":
    chat()
