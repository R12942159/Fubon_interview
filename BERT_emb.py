import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


df = pd.read_csv("./data/train.csv", encoding='utf-8')
test_df = pd.read_csv("./data/test.csv", encoding='utf-8')

cache_dir = os.path.expanduser("~/data_18TB/")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, cache_dir=cache_dir)

class TweetDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['target'].tolist(), test_size=0.15, random_state=42)

train_dataset = TweetDataset(train_texts, train_labels)
val_dataset = TweetDataset(val_texts, val_labels)
test_dataset = TweetDataset(test_df['text'].tolist())

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=["none"],
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

preds = trainer.predict(test_dataset).predictions.argmax(axis=1)
submission = pd.DataFrame({
    "id": test_df["id"],
    "target": preds
})
submission.to_csv("bert_acc.csv", index=False)