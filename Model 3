#Model 3
!pip install transformers torch pandas numpy scikit-learn

import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/fake_news_models'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"✓ Models will be saved to: {SAVE_DIR}")

# ============================================================================#
# CHECKPOINT MANAGEMENT
# ============================================================================#

def save_checkpoint(model, optimizer, scheduler, epoch, train_acc, val_acc, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    filepath = os.path.join(SAVE_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"  💾 Checkpoint saved: {filename} (Train: {train_acc*100:.2f}%, Val: {val_acc*100:.2f}%)")

def verify_checkpoint(filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        print(f"\n📋 Found existing checkpoint: {os.path.basename(filepath)}")
        print(f"   Epoch: {checkpoint['epoch'] + 1}")
        print(f"   Train Acc: {checkpoint.get('train_acc', 0)*100:.2f}%")
        print(f"   Val Acc: {checkpoint.get('val_acc', 0)*100:.2f}%")
        return True
    return False

print("="*60)
print("Model 3: PURE BERT + INTERPRETABLE REASONING (FIXED)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================#
# Load Data
# ============================================================================#
print("\n[1/7] Loading data...")

with open('train.pkl', 'rb') as f:
    train_df = pickle.load(f)
with open('val.pkl', 'rb') as f:
    val_df = pickle.load(f)
with open('test.pkl', 'rb') as f:
    test_df = pickle.load(f)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================================#
# Content Features for Reasoning
# ============================================================================#

def extract_content_features(text):
    features = {}
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)

    claim_words = ['allegedly', 'reportedly', 'supposedly', 'apparently',
                   'rumor', 'rumour', 'unconfirmed', 'speculation']
    features['unverified_claim_count'] = sum(1 for word in claim_words if word in text_lower)

    verified_words = ['confirmed', 'verified', 'proven', 'evidence', 'study',
                      'research', 'according to', 'official', 'statement']
    features['verification_count'] = sum(1 for word in verified_words if word in text_lower)

    absolute_words = ['always', 'never', 'everyone', 'nobody', 'all', 'none',
                      'every', 'impossible', 'definitely', 'certainly']
    features['absolute_language_count'] = sum(1 for word in absolute_words if word in text_lower)

    hedge_words = ['may', 'might', 'could', 'possibly', 'perhaps']
    features['hedge_word_count'] = sum(1 for word in hedge_words if word in text_lower)
    features['hedge_ratio'] = features['hedge_word_count'] / max(word_count, 1) * 100

    conspiracy_words = ['conspiracy', 'cover-up', 'coverup', 'hidden truth',
                        'they dont want you to know', 'wake up', 'sheeple']
    features['conspiracy_indicator_count'] = sum(1 for phrase in conspiracy_words if phrase in text_lower)

    fear_words = ['danger', 'threat', 'warning', 'alert', 'crisis', 'disaster']
    features['fear_word_count'] = sum(1 for word in fear_words if word in text_lower)

    positive_words = ['success', 'achievement', 'progress', 'improvement', 'solution']
    features['positive_word_count'] = sum(1 for word in positive_words if word in text_lower)
    features['fear_to_positive_ratio'] = features['fear_word_count'] / max(features['positive_word_count'], 1)

    features['quote_count'] = text.count('"') // 2
    features['source_attribution_count'] = sum(1 for pattern in ['according to', 'said', 'stated'] if pattern in text_lower)
    features['anonymous_source_count'] = sum(1 for indicator in ['anonymous source', 'unnamed source'] if indicator in text_lower)

    clickbait_phrases = ['you wont believe', "you won't believe", 'this is why', 'what happened next',
                         'shocking', 'amazing', 'incredible']
    features['clickbait_pattern_count'] = sum(1 for phrase in clickbait_phrases if phrase in text_lower)

    features['multiple_exclamation'] = text.count('!!') + text.count('!!!')
    features['caps_word_ratio'] = len([w for w in words if w.isupper() and len(w) > 1]) / max(word_count, 1)

    inflammatory_words = ['outrageous', 'shocking', 'disgusting', 'scandal', 'corrupt',
                          'insane', 'ridiculous', 'absurd']
    features['inflammatory_count'] = sum(1 for word in inflammatory_words if word in text_lower)

    academic_words = ['study', 'research', 'analysis', 'data', 'findings',
                      'professor', 'university', 'journal']
    features['academic_term_count'] = sum(1 for word in academic_words if word in text_lower)

    features['number_count'] = len(re.findall(r'\d+', text))
    features['percentage_count'] = len(re.findall(r'\d+%', text))

    features['has_date'] = 1 if re.search(r'\d{4}|\d{1,2}/\d{1,2}', text) else 0

    emotional_words = ['love', 'hate', 'angry', 'fear', 'terrible', 'wonderful']
    features['emotional_word_count'] = sum(1 for word in emotional_words if word in text_lower)

    features['question_count'] = text.count('?')

    features['avg_word_length'] = sum(len(w) for w in words) / max(word_count, 1)
    features['text_length'] = len(text)
    features['word_count'] = word_count

    first_person = ['i ', 'me ', 'my ', 'mine ', 'we ', 'our ', 'us ']
    features['first_person_count'] = sum(1 for fp in first_person if fp in text_lower)

    certainty_words = ['certain', 'sure', 'absolutely', 'undoubtedly', 'clearly']
    features['certainty_count'] = sum(1 for word in certainty_words if word in text_lower)

    urgency_words = ['now', 'immediately', 'urgent', 'breaking', 'just in']
    features['urgency_count'] = sum(1 for word in urgency_words if word in text_lower)

    return features

print("\n[2/7] Extracting content features for reasoning...")
train_content_features = pd.DataFrame([extract_content_features(text) for text in tqdm(train_df['cleaned_text'], desc="Train")])
val_content_features = pd.DataFrame([extract_content_features(text) for text in tqdm(val_df['cleaned_text'], desc="Val")])
test_content_features = pd.DataFrame([extract_content_features(text) for text in tqdm(test_df['cleaned_text'], desc="Test")])
print(f"✓ Content features extracted: {train_content_features.shape[1]} features")

# ============================================================================#
# Pure BERT Model
# ============================================================================#

class BERTFakeNewsClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        proba = self.sigmoid(logits)
        return proba

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BERTFakeNewsClassifier(model_name=model_name).to(device)

print(f"✓ BERT model loaded: {model_name}")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================#
# Dataset & DataLoader
# ============================================================================#

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

train_dataset = FakeNewsDataset(train_df['cleaned_text'].values, train_df['label'].values, tokenizer)
val_dataset = FakeNewsDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer)
test_dataset = FakeNewsDataset(test_df['cleaned_text'].values, test_df['label'].values, tokenizer)

# TRAIN loader (shuffled) + EVAL loader (non-shuffled)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("✓ Datasets and dataloaders created")

# ============================================================================#
# Evaluation helper
# ============================================================================#

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask).squeeze()
            preds = (outputs > 0.5).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, np.array(all_preds), np.array(all_labels)

# ============================================================================#
# Training Loop
# ============================================================================#

print("\n[5/7] Training Pure BERT...")

checkpoint_path = os.path.join(SAVE_DIR, 'model3_best.pth')
start_epoch = 0
best_val_acc = 0

if verify_checkpoint(checkpoint_path):
    user_input = input("\n⚠️  Use existing checkpoint? (yes/no): ").strip().lower()
    if user_input == 'yes':
        print("📂 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0)
        print(f"✅ Resuming from epoch {start_epoch + 1}")
    else:
        print("🆕 Starting fresh training (deleting old checkpoint)...")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        model = BERTFakeNewsClassifier(model_name=model_name).to(device)
else:
    print("🆕 No existing checkpoint found. Starting fresh training...")

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

if start_epoch > 0:
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

num_epochs = 4
patience = 3
patience_counter = 0

print(f"\nTraining for {num_epochs} epochs with patience={patience}")
print("="*70)

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = train_loss / len(train_loader)

    print(f"  Evaluating epoch {epoch+1}...")

    # IMPORTANT: use non-shuffled eval loader for train accuracy
    train_acc, _, _ = evaluate_model(model, train_eval_loader, device)
    val_acc, _, _ = evaluate_model(model, val_loader, device)

    scheduler.step(val_acc)

    print(f"\n{'='*70}")
    print(f"📊 Epoch {epoch+1}/{num_epochs} Results:")
    print(f"{'='*70}")
    print(f"  Average Loss:    {avg_train_loss:.4f}")
    print(f"  Train Accuracy:  {train_acc*100:.2f}%")
    print(f"  Val Accuracy:    {val_acc*100:.2f}%")
    print(f"  Gap (Train-Val): {abs(train_acc - val_acc)*100:.2f}%")

    save_checkpoint(model, optimizer, scheduler, epoch, train_acc, val_acc,
                    f'model3_epoch_{epoch+1}.pth')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'agent3_pure_bert_best.pth'))
        save_checkpoint(model, optimizer, scheduler, epoch, train_acc, val_acc,
                        'model3_best.pth')
        print(f"  ✅ NEW BEST MODEL! Val Acc: {val_acc*100:.2f}%")
    else:
        patience_counter += 1
        print(f"  ⏳ No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break

    print(f"{'='*70}\n")

# ============================================================================#
# Load best model and FINAL evaluation
# ============================================================================#

print("\n[6/7] Loading best model for final evaluation...")
best_model_path = os.path.join(SAVE_DIR, 'model3_pure_bert_best.pth')
model.load_state_dict(torch.load(best_model_path))
print(f"✓ Best model loaded (Val Acc: {best_val_acc*100:.2f}%)")

print("\n[7/7] Final evaluation on all datasets...")

def get_predictions_with_probas(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probas = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask).squeeze()
            probas = outputs.cpu().numpy()
            preds = (probas > 0.5).astype(int)

            all_probas.extend(probas)
            all_preds.extend(preds)

    return np.array(all_probas), np.array(all_preds)

# IMPORTANT: use train_eval_loader (non-shuffled) here
train_probas, train_preds = get_predictions_with_probas(model, train_eval_loader, device)
val_probas, val_preds = get_predictions_with_probas(model, val_loader, device)
test_probas, test_preds = get_predictions_with_probas(model, test_loader, device)

train_acc_final = accuracy_score(train_df['label'].values, train_preds)
val_acc_final = accuracy_score(val_df['label'].values, val_preds)
test_acc_final = accuracy_score(test_df['label'].values, test_preds)




print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Train Accuracy: {train_acc_final*100:.2f}%")
print(f"Val Accuracy:   {val_acc_final*100:.2f}%")
print(f"Test Accuracy:  {test_acc_final*100:.2f}%")
print("="*70)
