import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import spacy
from collections import Counter
import math

nlp = spacy.load("ru_core_news_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        stats = self._get_statistical_features(text)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stats_features': torch.FloatTensor(stats),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _get_statistical_features(self, text):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        sentences = [sent.text for sent in doc.sents]
        
        word_counts = Counter(tokens)
        probs = [count/len(tokens) for count in word_counts.values()]
        perplexity = math.exp(-sum(p * math.log(p) for p in probs)/len(probs)) if probs else 0
        
        avg_sent_len = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        unique_ratio = len(word_counts)/len(tokens) if tokens else 0
        
        pos_tags = [token.pos_ for token in doc]
        noun_ratio = pos_tags.count('NOUN')/len(pos_tags) if pos_tags else 0
        verb_ratio = pos_tags.count('VERB')/len(pos_tags) if pos_tags else 0
        
        case_counts = Counter([token.is_upper for token in doc])
        case_probs = [count/len(doc) for count in case_counts.values()]
        case_entropy = -sum(p * math.log(p+1e-10) for p in case_probs)
        
        punct_ratio = sum(1 for token in doc if token.is_punct)/len(doc) if doc else 0
        digit_ratio = sum(1 for token in doc if token.is_digit)/len(doc) if doc else 0
        stopword_ratio = sum(1 for token in doc if token.is_stop)/len(doc) if doc else 0
        
        return [
            perplexity, avg_sent_len, unique_ratio, 
            noun_ratio, verb_ratio, case_entropy,
            punct_ratio, digit_ratio, stopword_ratio,
            len(text.split())
        ]

class LLMDetector(nn.Module):
    def __init__(self, n_stats_features=10):
        super().__init__()
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        self.emb_dim = self.bert.config.hidden_size
        
        self.temporal_cnn = nn.Conv1d(self.emb_dim, 64, kernel_size=3, padding=1).to(device)
        self.temporal_lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True).to(device)
        
        self.style_attention = nn.MultiheadAttention(self.emb_dim, num_heads=4).to(device)
        self.style_norm = nn.LayerNorm(self.emb_dim).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim + 256 + n_stats_features, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, input_ids, attention_mask, stats_features):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        stats_features = stats_features.to(device)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        
        temporal_features = hidden_state.permute(0, 2, 1)
        temporal_features = self.temporal_cnn(temporal_features)
        temporal_features = temporal_features.permute(0, 2, 1)
        temporal_features, _ = self.temporal_lstm(temporal_features)
        temporal_features = temporal_features.mean(dim=1)
        
        style_features, _ = self.style_attention(hidden_state, hidden_state, hidden_state)
        style_features = self.style_norm(hidden_state + style_features)
        style_features = style_features.mean(dim=1)
        
        combined = torch.cat([style_features, temporal_features, stats_features], dim=1)
        return self.classifier(combined)

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], 
                       batch['attention_mask'], 
                       batch['stats_features'])
        
        loss = criterion(outputs.flatten(), batch['label'].float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).long()
        correct += (predicted.flatten() == batch['label']).sum().item()
    
    return total_loss/len(dataloader), correct/len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch['input_ids'], 
                          batch['attention_mask'], 
                          batch['stats_features'])
            
            loss = criterion(outputs.flatten(), batch['label'].float())
            total_loss += loss.item()
            correct += ((outputs > 0.5).long().flatten() == batch['label']).sum().item()
    
    return total_loss/len(dataloader), correct/len(dataloader.dataset)

def main():
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Ошибка: файл data.csv не найден!")
        print("Создайте CSV файл с колонками 'text' и 'label' (0 для человека, 1 для ИИ)")
        return
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = LLMDetector().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()
    
    train_loader = DataLoader(
        TextDataset(train_df['text'].values, train_df['label'].values, tokenizer),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(val_df['text'].values, val_df['label'].values, tokenizer),
        batch_size=16
    )
    
    best_acc = 0
    for epoch in range(5):
        print(f"\nЭпоха {epoch+1}/5")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'ru-AI-detect-03.pt')
            print("Модель сохранена!")
    
    print(f"\nОбучение завершено. Лучшая точность: {best_acc:.4f}")

if __name__ == "__main__":
    main()
