import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


df = pd.read_csv("/content/nlp-getting-started/train.csv", encoding='utf-8')
test_df = pd.read_csv("/content/nlp-getting-started/test.csv", encoding='utf-8')

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
model.eval()

def preprocess_data(df):
    df = df[['text', 'target']]
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['target'].tolist(), test_size=0.2, random_state=42)
    return train_texts, val_texts, train_labels, val_labels

train_texts, val_texts, train_labels, val_labels = preprocess_data(df)

def get_embeddings(texts, tokenizer, model, batch_size=8, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            masked = last_hidden * attention_mask.unsqueeze(-1)  # apply attention
            avg_emb = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            embeddings.append(avg_emb.cpu())
    return torch.cat(embeddings, dim=0)

train_embeddings = get_embeddings(train_texts, tokenizer, model)
val_embeddings = get_embeddings(val_texts, tokenizer, model)

test_texts = test_df['text'].tolist()
test_embeddings = get_embeddings(test_texts, tokenizer, model)

device = "cuda" if torch.cuda.is_available() else "cpu"

class DeepFC(nn.Module):
    def __init__(self, input_dim, output_dim=2, dropout=0.5):
        super(DeepFC, self).__init__()
        
        self.fc11 = nn.Linear(input_dim, input_dim//2)
        self.bs11 = nn.BatchNorm1d(input_dim//2)  # BatchNorm for first layer
        self.fc12 = nn.Linear(input_dim//2, input_dim//2)
        self.bs12 = nn.BatchNorm1d(input_dim//2)
        self.dropout1 = nn.Dropout(dropout)   # Dropout for first layer

        self.fc21 = nn.Linear(input_dim//2, input_dim//4)
        self.bs21 = nn.BatchNorm1d(input_dim//4)  # BatchNorm for second layer
        self.fc22 = nn.Linear(input_dim//4, input_dim//4)
        self.bs22 = nn.BatchNorm1d(input_dim//4)
        self.dropout2 = nn.Dropout(dropout)   # Dropout for second layer

        self.fc31 = nn.Linear(input_dim//4, input_dim//8)
        self.bs31 = nn.BatchNorm1d(input_dim//8)  # BatchNorm for third layer
        self.fc32 = nn.Linear(input_dim//8, input_dim//8)
        self.bs32 = nn.BatchNorm1d(input_dim//8)  # BatchNorm for third layer
        self.dropout3 = nn.Dropout(dropout)   # Dropout for third layer

        self.fc4 = nn.Linear(input_dim//8, output_dim)  # 最後一層輸出
        
        self.relu = nn.ReLU()  # 激活函數
        self.softmax = nn.Softmax(dim=1)  # Softmax 激活函數進行分類

    def forward(self, x):
        x = self.fc11(x)
        x = self.bs11(x)  # BatchNorm
        x = self.relu(x)  # ReLU 激活函數
        x = self.fc12(x)
        x = self.bs12(x)  # BatchNorm
        x = self.relu(x) 
        x = self.dropout1(x)  # Dropout

        x = self.fc21(x)
        x = self.bs21(x)  # BatchNorm
        x = self.relu(x)  # ReLU 激活函數
        x = self.fc22(x)
        x = self.bs22(x)  # BatchNorm
        x = self.relu(x)
        x = self.dropout2(x)  # Dropout

        x = self.fc31(x)
        x = self.bs31(x)  # BatchNorm
        x = self.relu(x)  # ReLU 激活函數
        x = self.fc32(x)
        x = self.bs32(x)  # BatchNorm
        x = self.relu(x)
        x = self.dropout3(x)  # Dropout

        x = self.fc4(x)  # 最後的全連接層
        return self.softmax(x)
    
input_dim = train_embeddings.shape[1]  # 計算特徵維度
model_fc = DeepFC(input_dim, dropout=0.6).to(device)

criterion = nn.CrossEntropyLoss()  # 交叉熵損失
optimizer = optim.Adam(model_fc.parameters(), lr=1e-4)

epochs = 1000
for epoch in range(epochs):
    model_fc.train()
    optimizer.zero_grad()
    
    outputs = model_fc(train_embeddings.to(device))
    loss = criterion(outputs, torch.tensor(train_labels).to(device))
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

model_fc.eval()
with torch.no_grad():
    val_outputs = model_fc(val_embeddings.to(device))
    val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
print(classification_report(val_labels, val_preds))

with torch.no_grad():
    test_outputs = model_fc(test_embeddings.to(device))
    test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()

kaggle_llm_FC = pd.DataFrame({
    'id': test_df['id'],
    'target': test_pred,
})
kaggle_llm_FC.to_csv("kaggle_llm_FC.csv", index=False)