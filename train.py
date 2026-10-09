import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# ১. ডেটা লোড ও টোকেনাইজার (Character-level Tokenizer)
# =====================================================================
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ইউনিক ক্যারেক্টার লিস্ট এবং ভোকাবুলারি সাইজ
chars = sorted(list(set(text)))
vocab_size = len(chars)

# ক্যারেক্টার থেকে ইন্টিজার (String to Integer - stoi) এবং উল্টো ম্যাপিং
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# পুরো টেক্সটকে সংখ্যায় রূপান্তর করে PyTorch টেনসরে নেওয়া
data = torch.tensor(encode(text), dtype=torch.long)

# =====================================================================
# ২. মডেল আর্কিটেকচার (Small Bigram Language Model)
# =====================================================================
class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # ক্যারেক্টার এমবেডিং টেবিল
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # পরবর্তী ক্যারেক্টারগুলো প্রেডিক্ট করার জন্য লুপ
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # শেষ ক্যারেক্টারের লজিট নেওয়া
            probs = F.softmax(logits, dim=-1) # প্রোবাবিলিটিতে রূপান্তর
            idx_next = torch.multinomial(probs, num_samples=1) # স্যাম্পল সিলেক্ট করা
            idx = torch.cat((idx, idx_next), dim=1) # কনটেক্সটের সাথে অ্যাড করা
        return idx

# =====================================================================
# ৩. মডেল ইনিশিয়ালাইজেশন ও ট্রেনিং
# =====================================================================
# যেহেতু ThinkPad X390-এ CPU ব্যবহার করছি, ডিভাইস সেট করে নেওয়া ভালো প্র্যাকটিস
device = 'cpu'
model = SmallLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# সিম্পল ব্যাচ ডেটা তৈরি (Batch size = 1)
xb = data[:-1].unsqueeze(0).to(device)
yb = data[1:].unsqueeze(0).to(device)

print("Training started on ThinkPad X390 (CPU)...")
for steps in range(1200):
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 300 == 0:
        print(f"Step {steps:4d} | Loss: {loss.item():.4f}")

# ৪. মডেলের ওয়েটস (Weights) সেভ করা
torch.save(model.state_dict(), 'small_model.pt')
print("\n[SUCCESS] Model trained and saved as 'small_model.pt'!")

# =====================================================================
# ৫. মডেল জেনারেশন টেস্ট
# =====================================================================
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nGenerated Output from AI Model:")
print("-" * 40)
print(decode(model.generate(context, max_new_tokens=40).tolist()[0]))
print("-" * 40)
