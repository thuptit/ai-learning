import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # ✅ Enable 4-bit mode
    bnb_4bit_compute_dtype=torch.float16,  # ✅ Faster computation
    bnb_4bit_use_double_quant=True,  # ✅ Further reduces memory
)
# ---------------------------
# 1️⃣ Load DeepSeek Model & Tokenizer
# ---------------------------
MODEL_PATH = "deepseek-llm-7b-chat"  # Change if using a local model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS for macOS

print(f"✅ Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model with proper offloading (if needed)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
).to(device)

# ---------------------------
# 2️⃣ Chat Function (Generates AI Responses)
# ---------------------------
def chat(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            attention_mask=attention_mask,  # ✅ Fixes attention mask issues
            pad_token_id=tokenizer.eos_token_id,  # ✅ Prevents padding warnings
            use_cache=True  # ✅ Improves performance
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------------------------
# 3️⃣ Run Chatbot
# ---------------------------
if __name__ == "__main__":
    print("\n🤖 DeepSeek Chatbot (type 'exit' to quit)")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! 👋")
            break

        response = chat(user_input)
        print(f"Chatbot: {response}")
