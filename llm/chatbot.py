import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F

# -----------------------------------
# 1️⃣ Tokenizer and Vocabulary
# -----------------------------------
import json
import os

# -------------------------------
# 🔹 Auto-Generate Vocabulary 🔹
# -------------------------------
def build_vocab(filename):
    """Read train_data.jsonl and build a vocabulary dynamically."""
    vocab = {"<PAD>": 0, "<UNK>": 1}  # Special tokens
    word_index = 2  # Start indexing from 2 (0 and 1 are for PAD & UNK)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Error: '{filename}' not found!")

    with open(filename, "r") as file:
        for line in file:
            entry = json.loads(line)
            words = entry["input"].lower().split() + entry["output"].lower().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = word_index
                    word_index += 1  # Assign the next available index

    return vocab

# -------------------------------
# 🔹 Build and Use Vocabulary 🔹
# -------------------------------
VOCAB_FILE = "train_data.jsonl"  # Update with correct filename
word_to_index = build_vocab(VOCAB_FILE)
index_to_word = {v: k for k, v in word_to_index.items()}  # Reverse mapping


def tokenize(sentence):
    """Convert words to token IDs."""
    return [word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence.lower().split()]

def detokenize(token_ids):
    words = []
    for i in token_ids:
        word = index_to_word.get(i, "<UNK>")
        if word != "<UNK>":  # Ignore unknown words if possible
            words.append(word)
    return " ".join(words) if words else "I don't understand."

class ChatDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        if not os.path.exists(filename):
            raise FileNotFoundError(f"❌ Error: '{filename}' not found!")

        with open(filename, "r") as file:
            for line in file:
                entry = json.loads(line)
                input_ids = tokenize(entry["input"])
                output_ids = tokenize(entry["output"])
                self.data.append((input_ids, output_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# -----------------------------------
# 3️⃣ Transformer Model Components
# -----------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = self.softmax(scores)
        return torch.matmul(attn_weights, V), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = SelfAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        Q, K, V = [w(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for w, x in zip([self.W_q, self.W_k, self.W_v], [Q, K, V])]
        attn_output, _ = self.attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        return self.W_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dim_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_ff):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.output_layer(x)

# -----------------------------------
# 4️⃣ Train & Save Model
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniTransformer(len(word_to_index), 128, 8, 2, 512).to(device)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(max(len(seq) for seq in inputs), max(len(seq) for seq in targets))
    padded_inputs = [seq.tolist() + [0] * (max_len - len(seq)) for seq in inputs]
    padded_targets = [seq.tolist() + [0] * (max_len - len(seq)) for seq in targets]

    return torch.tensor(padded_inputs), torch.tensor(padded_targets)

train_dataset = ChatDataset("train_data.jsonl")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        for inputs, targets in train_loader:
            inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in inputs], batch_first=True).to(device)
            targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in targets], batch_first=True).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[:, :targets.shape[1], :].contiguous().view(-1, outputs.shape[-1]), targets.contiguous().view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

print("🚀 Training Mini Transformer...")
train_model()
torch.save(model.state_dict(), "mini_transformer.pth")
print("✅ Model saved as 'mini_transformer.pth'!")

model.load_state_dict(torch.load("mini_transformer.pth", map_location=device))
model.eval()
def generate_response(input_text, max_length=10, temperature=1.0):
    """Generates a chatbot response using the trained Transformer model."""
    input_tokens = tokenize(input_text) 
    input_tensor = torch.tensor([input_tokens]).to(device)

    with torch.no_grad():
        output_tokens = []
        for _ in range(max_length):
            output = model(input_tensor)
            probabilities = F.softmax(output[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()
            if next_token == word_to_index["<PAD>"] or next_token == word_to_index["<UNK>"]:
                break 
            output_tokens.append(next_token)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

    return detokenize(output_tokens)

# Chat function remains the same
def chat():
    print("🤖 Mini Transformer Chatbot (type 'exit' to quit)")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! 👋")
            break

        input_tokens = tokenize(user_input)
        input_tensor = torch.tensor([input_tokens]).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Ensure output_tokens is always a list
        output_tokens = output_tensor.argmax(dim=-1).squeeze()
        if output_tokens.dim() == 0:  # Convert single value to list
            output_tokens = [output_tokens.item()]
        else:
            output_tokens = output_tokens.tolist()
        response = detokenize(output_tokens)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
