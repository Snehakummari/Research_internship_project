"""
Model 2: Advanced Linguistic Analysis Agent (True Agentic Architecture)
Provides reasoning and confidence, NOT final predictions
Outputs structured decisions for Coordinator-led debate
"""

# Install required packages
!pip install spacy textstat
!python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import pickle
import os
import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import textstat
import re

print("=" * 60)
print("MODEL 2: ADVANCED LINGUISTIC ANALYSIS (AGENTIC)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Load spaCy
# ============================================================================
print("\n[1/7] Loading spaCy model...")
try:
    nlp = spacy.load('en_core_web_sm')
    print("✓ spaCy loaded successfully")
except:
    print("❌ Error: spaCy model not found!")
    exit()

os.makedirs('/content/models', exist_ok=True)

# ============================================================================
# COMPREHENSIVE Linguistic Feature Extraction
# ============================================================================
print("\n[2/7] Defining comprehensive linguistic feature extraction...")

def extract_comprehensive_linguistic_features(text, max_chars=5000):
    """Extract ~75 comprehensive linguistic features"""
    features = {}
    text_truncated = text[:max_chars]
    doc = nlp(text_truncated)

    total_tokens = len(doc)
    if total_tokens == 0:
        return {f'feature_{i}': 0 for i in range(75)}

    # === 1. PART-OF-SPEECH DISTRIBUTION ===
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tokens
    features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tokens
    features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tokens
    features['adv_ratio'] = pos_counts.get('ADV', 0) / total_tokens
    features['pron_ratio'] = pos_counts.get('PRON', 0) / total_tokens
    features['det_ratio'] = pos_counts.get('DET', 0) / total_tokens
    features['aux_ratio'] = pos_counts.get('AUX', 0) / total_tokens

    # === 2. DEPENDENCY PARSING - SYNTACTIC COMPLEXITY ===
    depths = []
    for token in doc:
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
            if depth > 20:
                break
        depths.append(depth)

    features['avg_dep_depth'] = np.mean(depths) if depths else 0
    features['max_dep_depth'] = np.max(depths) if depths else 0
    features['dep_depth_std'] = np.std(depths) if depths else 0

    dep_types = set([token.dep_ for token in doc])
    features['dep_type_diversity'] = len(dep_types)
    features['dep_type_ratio'] = len(dep_types) / total_tokens

    # === 3. NAMED ENTITY ANALYSIS ===
    entities = [ent for ent in doc.ents]
    features['entity_count'] = len(entities)
    features['entity_density'] = len(entities) / total_tokens * 100

    entity_types = {}
    for ent in entities:
        entity_types[ent.label_] = entity_types.get(ent.label_, 0) + 1

    features['entity_type_diversity'] = len(entity_types)
    features['person_entity_ratio'] = entity_types.get('PERSON', 0) / max(len(entities), 1)
    features['org_entity_ratio'] = entity_types.get('ORG', 0) / max(len(entities), 1)
    features['gpe_entity_ratio'] = entity_types.get('GPE', 0) / max(len(entities), 1)
    features['date_entity_ratio'] = entity_types.get('DATE', 0) / max(len(entities), 1)
    features['event_entity_ratio'] = entity_types.get('EVENT', 0) / max(len(entities), 1)

    # === 4. SENTENCE STRUCTURE ===
    sentences = list(doc.sents)
    sent_lengths = [len(sent) for sent in sentences]

    features['num_sentences'] = len(sentences)
    features['avg_sent_length'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sent_length_std'] = np.std(sent_lengths) if sent_lengths else 0
    features['max_sent_length'] = max(sent_lengths) if sent_lengths else 0
    features['min_sent_length'] = min(sent_lengths) if sent_lengths else 0
    features['sent_length_cv'] = (features['sent_length_std'] / max(features['avg_sent_length'], 1))
    features['sent_length_range'] = features['max_sent_length'] - features['min_sent_length']

    # === 5. READABILITY METRICS ===
    try:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text_truncated)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text_truncated)
        features['gunning_fog'] = textstat.gunning_fog(text_truncated)
        features['smog_index'] = textstat.smog_index(text_truncated)
        features['automated_readability'] = textstat.automated_readability_index(text_truncated)
        features['coleman_liau'] = textstat.coleman_liau_index(text_truncated)
    except:
        features['flesch_reading_ease'] = 0
        features['flesch_kincaid_grade'] = 0
        features['gunning_fog'] = 0
        features['smog_index'] = 0
        features['automated_readability'] = 0
        features['coleman_liau'] = 0

    # === 6. DISCOURSE MARKERS - COHERENCE ===
    discourse_markers = {
        'contrast': ['however', 'but', 'yet', 'although', 'nevertheless', 'nonetheless'],
        'addition': ['moreover', 'furthermore', 'additionally', 'also', 'besides'],
        'causation': ['therefore', 'thus', 'hence', 'consequently', 'because', 'since'],
        'temporal': ['then', 'meanwhile', 'subsequently', 'previously', 'finally']
    }

    text_lower = text_truncated.lower()
    for category, markers in discourse_markers.items():
        count = sum(1 for marker in markers if f' {marker} ' in f' {text_lower} ')
        features[f'{category}_marker_count'] = count

    features['total_discourse_markers'] = sum(features[f'{cat}_marker_count'] for cat in discourse_markers.keys())
    features['discourse_marker_density'] = features['total_discourse_markers'] / max(len(sentences), 1)

    # === 7. VOCABULARY QUALITY ===
    word_tokens = [token for token in doc if token.is_alpha]
    unique_words = set([token.text.lower() for token in word_tokens])
    features['lexical_diversity'] = len(unique_words) / max(len(word_tokens), 1)

    oov_words = [token for token in word_tokens if token.is_oov]
    features['oov_ratio'] = len(oov_words) / max(len(word_tokens), 1)

    stopwords = [token for token in doc if token.is_stop]
    features['stopword_ratio'] = len(stopwords) / total_tokens

    content_words = [token for token in doc if not token.is_stop and token.is_alpha]
    features['content_word_ratio'] = len(content_words) / total_tokens

    features['vocab_richness'] = len(unique_words) / max(len(sentences), 1)

    word_freq = {}
    for token in word_tokens:
        word_freq[token.text.lower()] = word_freq.get(token.text.lower(), 0) + 1
    hapax_legomena = [w for w, count in word_freq.items() if count == 1]
    features['hapax_ratio'] = len(hapax_legomena) / max(len(word_tokens), 1)

    # === 8. WORD LENGTH & COMPLEXITY ===
    word_lengths = [len(token.text) for token in word_tokens]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
    features['word_length_std'] = np.std(word_lengths) if word_lengths else 0
    features['max_word_length'] = max(word_lengths) if word_lengths else 0

    long_words = [w for w in word_lengths if w > 6]
    features['long_word_ratio'] = len(long_words) / max(len(word_lengths), 1)

    very_long_words = [w for w in word_lengths if w > 10]
    features['very_long_word_ratio'] = len(very_long_words) / max(len(word_lengths), 1)

    # === 9. SYNTACTIC COMPLEXITY ===
    passive_count = 0
    for i, token in enumerate(doc[:-1]):
        if token.lemma_ == 'be' and doc[i+1].tag_ in ['VBN', 'VBD']:
            passive_count += 1
    features['passive_voice_ratio'] = passive_count / max(len(sentences), 1)

    subordinate_conj = [token for token in doc if token.pos_ == 'SCONJ']
    features['subordinate_clause_ratio'] = len(subordinate_conj) / max(len(sentences), 1)

    prep_phrases = [token for token in doc if token.pos_ == 'ADP']
    features['prep_phrase_density'] = len(prep_phrases) / total_tokens

    relative_pronouns = [t for t in doc if t.text.lower() in ['who', 'which', 'that'] and t.dep_ in ['nsubj', 'dobj']]
    features['relative_clause_density'] = len(relative_pronouns) / max(len(sentences), 1)

    coord_conj = [token for token in doc if token.pos_ == 'CCONJ']
    features['clause_density'] = (len(coord_conj) + len(subordinate_conj)) / max(len(sentences), 1)

    # === 10. PUNCTUATION PATTERNS ===
    features['punct_ratio'] = len([t for t in doc if t.is_punct]) / total_tokens
    features['comma_ratio'] = text_truncated.count(',') / total_tokens
    features['semicolon_ratio'] = text_truncated.count(';') / total_tokens
    features['colon_ratio'] = text_truncated.count(':') / total_tokens
    features['dash_ratio'] = (text_truncated.count('—') + text_truncated.count('--')) / total_tokens
    features['ellipsis_count'] = text_truncated.count('...') + text_truncated.count('…')
    features['parenthesis_ratio'] = (text_truncated.count('(') + text_truncated.count(')')) / total_tokens

    # === 11. QUOTATION & ATTRIBUTION ===
    features['quote_count'] = (text_truncated.count('"') + text_truncated.count("'")) / 2
    features['quote_density'] = features['quote_count'] / max(len(sentences), 1)

    attribution_verbs = ['said', 'says', 'stated', 'states', 'claims', 'claimed',
                        'reported', 'reports', 'announced', 'declares', 'declared',
                        'according', 'noted', 'explained', 'confirmed']
    features['attribution_count'] = sum(1 for verb in attribution_verbs if verb in text_lower)
    features['attribution_density'] = features['attribution_count'] / max(len(sentences), 1)

    # === 12. NUMBERS & STATISTICS ===
    features['number_count'] = len([token for token in doc if token.like_num])
    features['number_density'] = features['number_count'] / total_tokens
    features['number_per_sentence'] = features['number_count'] / max(len(sentences), 1)

    # === 13. QUESTION & EXCLAMATION ===
    features['question_ratio'] = len([s for s in sentences if '?' in s.text]) / max(len(sentences), 1)
    features['exclamation_ratio'] = len([s for s in sentences if '!' in s.text]) / max(len(sentences), 1)

    # === 14. PRONOUN PERSPECTIVE ANALYSIS ===
    first_person = len([t for t in doc if t.text.lower() in ['i', 'we', 'me', 'us', 'my', 'our', 'mine', 'ours']])
    second_person = len([t for t in doc if t.text.lower() in ['you', 'your', 'yours']])
    third_person = len([t for t in doc if t.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'their']])

    features['first_person_ratio'] = first_person / total_tokens
    features['second_person_ratio'] = second_person / total_tokens
    features['third_person_ratio'] = third_person / total_tokens
    features['pronoun_perspective_diversity'] = len([x for x in [first_person, second_person, third_person] if x > 0])

    # === 15. CONJUNCTION USAGE ===
    features['coord_conjunction_ratio'] = len(coord_conj) / total_tokens
    features['subord_conjunction_ratio'] = len(subordinate_conj) / total_tokens

    return features

print("✓ Comprehensive linguistic feature extraction defined")

# ============================================================================
# Load Data
# ============================================================================
print("\n[3/7] Loading cleaned data splits...")

try:
    with open('train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    with open('val.pkl', 'rb') as f:
        val_df = pickle.load(f)
    with open('test.pkl', 'rb') as f:
        test_df = pickle.load(f)

    print("✅ Data loaded successfully")
    print(f"\n📊 Data Split Sizes:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")

except FileNotFoundError:
    print("❌ Data files not found!")
    raise

# Test extraction
sample_text = train_df['cleaned_text'].iloc[0][:5000]
sample_features = extract_comprehensive_linguistic_features(sample_text)
print(f"✓ Extracted {len(sample_features)} comprehensive linguistic features")

# ============================================================================
# Extract Features
# ============================================================================
print("\n[4/7] Extracting features from all articles...")

def extract_features_from_df(df, desc="Processing"):
    features_list = []
    for text in tqdm(df['cleaned_text'], desc=desc):
        try:
            features = extract_comprehensive_linguistic_features(text)
            features_list.append(features)
        except Exception as e:
            features = {k: 0 for k in sample_features.keys()}
            features_list.append(features)
    return pd.DataFrame(features_list)

train_features = extract_features_from_df(train_df, "Train set")
val_features = extract_features_from_df(val_df, "Val set")
test_features = extract_features_from_df(test_df, "Test set")

print(f"✓ Train features: {train_features.shape}")
print(f"✓ Val features: {val_features.shape}")
print(f"✓ Test features: {test_features.shape}")

# ============================================================================
# Feature Normalization
# ============================================================================
print("\n[5/7] Normalizing features...")

train_features = train_features.replace([np.inf, -np.inf], np.nan).fillna(0)
val_features = val_features.replace([np.inf, -np.inf], np.nan).fillna(0)
test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)

with open('/content/models/linguistic_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✓ Features normalized")

train_labels = train_df['label'].values
val_labels = val_df['label'].values
test_labels = test_df['label'].values

X_train = torch.FloatTensor(train_features_scaled)
y_train = torch.FloatTensor(train_labels)
X_val = torch.FloatTensor(val_features_scaled)
y_val = torch.FloatTensor(val_labels)
X_test = torch.FloatTensor(test_features_scaled)
y_test = torch.FloatTensor(test_labels)

# ============================================================================
# Enhanced Neural Network
# ============================================================================
print("\n[6/7] Building enhanced neural network...")

class EnhancedLinguisticClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = EnhancedLinguisticClassifier(input_dim).to(device)

print(f"✓ Model: {input_dim} → 256 → 128 → 64 → 1")

# ============================================================================
# Training
# ============================================================================
print("\n[7/7] Training model...")

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

num_epochs = 50
batch_size = 64
best_val_acc = 0
patience = 8
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    indices = torch.randperm(len(X_train))

    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_X = X_train[batch_indices].to(device)
        batch_y = y_train[batch_indices].to(device)

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device)).squeeze()
        val_pred = (val_outputs > 0.5).float().cpu().numpy()
        val_acc = accuracy_score(y_val.numpy(), val_pred)
        val_auc = roc_auc_score(y_val.numpy(), val_outputs.cpu().numpy())

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | AUC: {val_auc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), '/content/models/agent2_model.pth')
        print(f"  ✓ Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load('/content/models/agent2_model.pth', weights_only=True))

# ============================================================================
# Generate Reasoning-Based Outputs 
# ============================================================================
print("\n[8/8] Generating reasoning-based decisions...")

def generate_model_reasoning(features_df, probabilities, predictions, feature_names):
    """Generate structured reasoning for each decision"""
    decisions = []

    for idx in range(len(predictions)):
        proba = probabilities[idx]
        pred = predictions[idx]
        features = features_df.iloc[idx]

        # Build linguistic reasoning
        linguistic_signals = {}

        # Readability concerns
        if features['flesch_reading_ease'] < 30:
            linguistic_signals['difficult_readability'] = f"Very difficult reading level ({features['flesch_reading_ease']:.1f})"
        elif features['flesch_reading_ease'] > 80:
            linguistic_signals['overly_simple'] = f"Unusually simple language ({features['flesch_reading_ease']:.1f})"

        # Syntactic complexity
        if features['avg_dep_depth'] < 2.0:
            linguistic_signals['shallow_syntax'] = f"Shallow syntactic structure ({features['avg_dep_depth']:.2f})"
        elif features['avg_dep_depth'] > 5.0:
            linguistic_signals['complex_syntax'] = f"Unusually complex syntax ({features['avg_dep_depth']:.2f})"

        # Entity analysis
        if features['entity_density'] < 1.0:
            linguistic_signals['low_entity_density'] = f"Few named entities ({features['entity_density']:.2f}%)"

        # Attribution analysis
        if features['attribution_density'] < 0.5:
            linguistic_signals['low_attribution'] = f"Minimal source attribution ({features['attribution_density']:.2f})"

        # Discourse coherence
        if features['discourse_marker_density'] < 0.5:
            linguistic_signals['poor_coherence'] = f"Weak discourse markers ({features['discourse_marker_density']:.2f})"

        # Vocabulary quality
        if features['lexical_diversity'] < 0.3:
            linguistic_signals['low_lexical_diversity'] = f"Limited vocabulary ({features['lexical_diversity']:.2%})"

        decision = {
            'decision': 'REAL' if pred == 1 else 'FAKE',
            'confidence': float(proba if pred == 1 else 1 - proba),
            'reasoning': {
                'model_name': 'Linguistic Analysis Model',
                'primary_factors': linguistic_signals,
                'feature_summary': {
                    'flesch_reading_ease': float(features['flesch_reading_ease']),
                    'avg_dep_depth': float(features['avg_dep_depth']),
                    'entity_density': float(features['entity_density']),
                    'attribution_density': float(features['attribution_density']),
                    'discourse_marker_density': float(features['discourse_marker_density']),
                    'lexical_diversity': float(features['lexical_diversity'])
                },
                'reasoning_text': f"Based on linguistic analysis: {len(linguistic_signals)} structural concerns detected" if pred == 0
                                 else f"Linguistic structure appears professional with {len(linguistic_signals)} minor issues"
            }
        }

        decisions.append(decision)

    return decisions

# Generate predictions
model.eval()
with torch.no_grad():
    train_proba = model(X_train.to(device)).squeeze().cpu().numpy()
    val_proba = model(X_val.to(device)).squeeze().cpu().numpy()
    test_proba = model(X_test.to(device)).squeeze().cpu().numpy()

train_pred = (train_proba > 0.5).astype(int)
val_pred = (val_proba > 0.5).astype(int)
test_pred = (test_proba > 0.5).astype(int)

# Generate reasoning
train_decisions = generate_agent_reasoning(train_features, train_proba, train_pred, train_features.columns)
val_decisions = generate_agent_reasoning(val_features, val_proba, val_pred, val_features.columns)
test_decisions = generate_agent_reasoning(test_features, test_proba, test_pred, test_features.columns)

# ============================================================================
# Evaluate and Save
# ============================================================================
train_acc = accuracy_score(train_labels, train_pred)
val_acc = accuracy_score(val_labels, val_pred)
test_acc = accuracy_score(test_labels, test_pred)

print(f"\n📊 Model 2 Performance (for reference):")
print(f"  Train: {train_acc*100:.2f}%")
print(f"  Val:   {val_acc*100:.2f}%")
print(f"  Test:  {test_acc*100:.2f}%")

print(f"\n📋 Classification Report:")
print(classification_report(test_labels, test_pred, target_names=['Fake', 'Real']))

# Save agentic outputs
model2_outputs = {
    'train': {
        'decisions': train_decisions,
        'features': train_features_scaled,
        'feature_names': list(train_features.columns),
        'raw_proba': train_proba,
        'raw_pred': train_pred
    },
    'val': {
        'decisions': val_decisions,
        'features': val_features_scaled,
        'feature_names': list(val_features.columns),
        'raw_proba': val_proba,
        'raw_pred': val_pred
    },
    'test': {
        'decisions': test_decisions,
        'features': test_features_scaled,
        'feature_names': list(test_features.columns),
        'raw_proba': test_proba,
        'raw_pred': test_pred
    }
}

with open('/content/models/model2_outputs.pkl', 'wb') as f:
    pickle.dump(agent2_outputs, f)

print("\n✓ Model saved to '/content/models/model2_model.pth'")
print("✓ Model outputs saved to '/content/models/model2_outputs.pkl'")

# Sample reasoning
print("\n" + "="*70)
print("📝 SAMPLE Model 2 REASONING (First 3 test samples):")
print("="*70)
for i in range(min(3, len(test_decisions))):
    dec = test_decisions[i]
    actual = "REAL" if test_labels[i] == 1 else "FAKE"
    print(f"\nSample {i+1}:")
    print(f"  Model Decision: {dec['decision']} (Confidence: {dec['confidence']:.2%})")
    print(f"  Actual Label: {actual}")
    print(f"  Reasoning: {dec['reasoning']['reasoning_text']}")
    print(f"  Key Factors: {list(dec['reasoning']['primary_factors'].keys())}")

print("\n" + "=" * 60)
print("✅ Model 2  TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📈 Summary:")
print(f"  • Model Type: Linguistic Analysis")
print(f"  • Output Format: Structured decisions with reasoning")
print(f"  • Total Features: ~75 linguistic features")
print(f"  • Architecture: 256→128→64→1")
print(f"  • Ready for Coordinator integration")
