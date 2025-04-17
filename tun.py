import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import spacy
from collections import Counter
import math
import gradio as gr
import numpy as np

nlp = spacy.load("ru_core_news_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

class LLMDetector(nn.Module):
    def __init__(self, n_stats_features=10):
        super().__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
        self.emb_dim = self.bert.config.hidden_size
        
        self.temporal_cnn = nn.Conv1d(self.emb_dim, 64, kernel_size=3, padding=1)
        self.temporal_lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        
        self.style_attention = nn.MultiheadAttention(self.emb_dim, num_heads=4)
        self.style_norm = nn.LayerNorm(self.emb_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim + 256 + n_stats_features, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask, stats_features):
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

def get_statistical_features(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    sentences = [sent.text for sent in doc.sents]
    
    word_counts = Counter(tokens)
    probs = [count/len(tokens) for count in word_counts.values()] if tokens else []
    perplexity = math.exp(-sum(p * math.log(p) for p in probs)/len(probs)) if probs else 0
    
    avg_sent_len = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
    unique_ratio = len(word_counts)/len(tokens) if tokens else 0
    
    pos_tags = [token.pos_ for token in doc]
    noun_ratio = pos_tags.count('NOUN')/len(pos_tags) if pos_tags else 0
    verb_ratio = pos_tags.count('VERB')/len(pos_tags) if pos_tags else 0
    
    case_counts = Counter([token.is_upper for token in doc])
    case_probs = [count/len(doc) for count in case_counts.values()] if doc else []
    case_entropy = -sum(p * math.log(p+1e-10) for p in case_probs) if case_probs else 0
    
    punct_ratio = sum(1 for token in doc if token.is_punct)/len(doc) if doc else 0
    digit_ratio = sum(1 for token in doc if token.is_digit)/len(doc) if doc else 0
    stopword_ratio = sum(1 for token in doc if token.is_stop)/len(doc) if doc else 0
    
    return [
        perplexity, avg_sent_len, unique_ratio, 
        noun_ratio, verb_ratio, case_entropy,
        punct_ratio, digit_ratio, stopword_ratio,
        len(text.split())
    ]

def load_model(model_path):
    model = LLMDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model("ru-AI-detect-03.pt")

def predict(text):
    encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    stats = get_statistical_features(text)
    stats_tensor = torch.FloatTensor(stats).unsqueeze(0)
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    stats_tensor = stats_tensor.to(device)
    
    with torch.no_grad():
        prob = model(input_ids, attention_mask, stats_tensor).item()
    
    return float(prob), float(1 - prob), "AI-generated" if prob > 0.5 else "Human-written"

def create_interface():
    with gr.Blocks(title="AI Text Detector") as app:
        gr.Markdown("""
        # ru-AI-detect-03
        Определяет, был ли текст написан человеком или нейросетью (GPT и аналоги)
        """)
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Введите текст для анализа",
                    placeholder="Напишите или вставьте текст здесь...",
                    lines=7
                )
                analyze_btn = gr.Button("Анализировать", variant="primary")
            
            with gr.Column():
                ai_prob = gr.Number(label="Вероятность ИИ", precision=4)
                human_prob = gr.Number(label="Вероятность человека", precision=4)
                verdict = gr.Textbox(label="Результат")
                gr.Examples(
                    examples=[
                        ["Привет! Как твои дела?", "Пример человеческого текста"],
                        ["Здравствуйте. Моя работа заключается в анализе и обработке данных.", "Пример AI-текста"]
                    ],
                    inputs=input_text
                )
        
        analyze_btn.click(
            fn=predict,
            inputs=input_text,
            outputs=[ai_prob, human_prob, verdict]
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="127.0.0.1", server_port=7860)