
"""
Model 1: Stylistic Analysis module
Provides reasoning and confidence, NOT final predictions
Outputs structured decisions for Coordinator-led debate
"""

# Install required packages
!pip install textstat

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import re
import textstat
from tqdm import tqdm

print("=" * 60)
print("Model 1: STYLISTIC ANALYSIS Module")
print("=" * 60)

# ============================================================================
# Load Cleaned data splits
# ============================================================================
print("\n📥 LOADING CLEANED DATA SPLITS...")
print("="*70)

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
    print("❌ train.pkl, val.pkl, or test.pkl not found!")
    raise

print(f"\nLabel distribution (Train):")
print(train_df['label'].value_counts())

os.makedirs('/content/models', exist_ok=True)

# ============================================================================
# Enhanced Feature Extraction Functions
# ============================================================================
print("\n[2/6] Defining enhanced feature extraction functions...")

def extract_enhanced_stylistic_features(text):
    """
    Model 1: PURE STYLISTIC FEATURES ONLY
    Extracts stylistic features for fake news detection
    """
    features = {}
    words = text.split()

    # === 1. SENSATIONALISM FEATURES ===
    features['exclamation_count'] = text.count('!')
    features['exclamation_density'] = text.count('!') / max(len(text), 1) * 1000
    features['question_count'] = text.count('?')

    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    features['caps_ratio'] = len(caps_words) / max(len(words), 1)

    clickbait_phrases = [
        'you won\'t believe', 'shocking', 'amazing', 'incredible',
        'must see', 'this is why', 'what happened next', 'gone wrong',
        'mind blown', 'can\'t believe', 'unbelievable'
    ]
    features['clickbait_count'] = sum(1 for phrase in clickbait_phrases if phrase in text.lower())

    emotional_words = [
        'hate', 'love', 'fear', 'angry', 'terrible', 'awesome',
        'horrible', 'wonderful', 'disgusting', 'outrage', 'furious',
        'devastated', 'thrilled', 'ecstatic'
    ]
    features['emotional_word_count'] = sum(1 for word in emotional_words if word in text.lower())

    superlatives = ['best', 'worst', 'most', 'least', 'greatest', 'biggest', 'smallest']
    features['superlative_count'] = sum(text.lower().count(word) for word in superlatives)

    intensifiers = ['very', 'extremely', 'absolutely', 'completely', 'totally',
                   'utterly', 'incredibly', 'remarkably']
    features['intensifier_count'] = sum(text.lower().count(word) for word in intensifiers)

    urgency = ['breaking', 'urgent', 'just in', 'now', 'alert', 'immediate',
               'developing', 'watch']
    features['urgency_count'] = sum(text.lower().count(word) for word in urgency)

    negative_words = ['hate', 'fear', 'angry', 'terrible', 'horrible', 'disgusting',
                     'awful', 'bad', 'worst', 'evil']
    features['negative_emotion_count'] = sum(text.lower().count(word) for word in negative_words)

    positive_words = ['love', 'great', 'awesome', 'wonderful', 'amazing', 'excellent',
                     'good', 'best', 'perfect']
    features['positive_emotion_count'] = sum(text.lower().count(word) for word in positive_words)

    # === 2. READABILITY FEATURES ===
    try:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    except:
        features['flesch_reading_ease'] = 50.0

    try:
        features['gunning_fog'] = textstat.gunning_fog(text)
    except:
        features['gunning_fog'] = 12.0

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    features['avg_sentence_length'] = avg_sent_len

    unique_words = len(set(words))
    features['lexical_diversity'] = unique_words / max(len(words), 1)

    sent_lengths = [len(s.split()) for s in sentences]
    features['sentence_length_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # === 3. PUNCTUATION PATTERNS ===
    features['multiple_punct'] = len(re.findall(r'[!?]{2,}', text))
    features['quote_count'] = text.count('"') + text.count("'")
    features['ellipsis_count'] = text.count('...')
    features['comma_density'] = text.count(',') / max(len(text), 1) * 1000
    features['dash_count'] = text.count('—') + text.count('--') + text.count(' - ')
    features['punct_variety'] = len(set(re.findall(r'[^\w\s]', text)))

    # === 4. WRITING STYLE FEATURES ===
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['number_density'] = len(re.findall(r'\d+', text)) / max(len(words), 1)
    features['long_word_ratio'] = len([w for w in words if len(w) > 10]) / max(len(words), 1)

    first_person = ['i ', 'we ', 'my ', 'our ', 'me ', 'us ']
    features['first_person_count'] = sum(1 for pronoun in first_person if pronoun in ' ' + text.lower() + ' ')

    features['title_case_ratio'] = len([w for w in words if w.istitle()]) / max(len(words), 1)
    features['text_length'] = len(text)

    features['paragraph_count'] = max(text.count('\n\n') + 1, 1)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    features['avg_paragraph_length'] = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0

    return features

print("✓ Feature extraction functions defined")
sample_text = train_df['cleaned_text'].iloc[0]
sample_features = extract_enhanced_stylistic_features(sample_text)
print(f"✓ Extracted {len(sample_features)} features")

# ============================================================================
# Extract Features for All Data
# ============================================================================
print("\n[3/6] Extracting features from all articles...")

def extract_features_from_df(df):
    features_list = []
    for text in tqdm(df['cleaned_text'], desc="Extracting"):
        features = extract_enhanced_stylistic_features(text)
        features_list.append(features)
    return pd.DataFrame(features_list)

train_features = extract_features_from_df(train_df)
val_features = extract_features_from_df(val_df)
test_features = extract_features_from_df(test_df)

print(f"✓ Train features shape: {train_features.shape}")
print(f"✓ Val features shape: {val_features.shape}")
print(f"✓ Test features shape: {test_features.shape}")

train_labels = train_df['label'].values
val_labels = val_df['label'].values
test_labels = test_df['label'].values

# ============================================================================
# Train XGBoost Model (for confidence scoring only)
# ============================================================================
print("\n[4/6] Training XGBoost for confidence scoring...")

model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    gamma=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss'
)

eval_set = [(train_features, train_labels), (val_features, val_labels)]
model.fit(train_features, train_labels, eval_set=eval_set, verbose=20)

print("✓ Training complete!")

# ============================================================================
# Generate Reasoning-Based Outputs 
# ============================================================================
print("\n[5/6] Generating reasoning-based decisions for Coordinator...")

def generate_model_reasoning(features_df, probabilities, predictions, feature_names):
    """
    Generate structured reasoning for each decision
    Format: {"decision": "FAKE/REAL", "confidence": 0.XX, "reasoning": {...}}
    """
    decisions = []

    for idx in range(len(predictions)):
        proba = probabilities[idx]
        pred = predictions[idx]
        features = features_df.iloc[idx]

        # Build reasoning from top contributing features
        feature_contributions = {}

        # Key stylistic indicators
        if features['exclamation_density'] > 2.0:
            feature_contributions['high_exclamation'] = f"High exclamation density ({features['exclamation_density']:.2f})"

        if features['caps_ratio'] > 0.05:
            feature_contributions['excessive_caps'] = f"Excessive caps usage ({features['caps_ratio']:.2%})"

        if features['clickbait_count'] > 0:
            feature_contributions['clickbait_detected'] = f"Clickbait phrases found ({features['clickbait_count']})"

        if features['emotional_word_count'] > 5:
            feature_contributions['emotional_language'] = f"High emotional word count ({features['emotional_word_count']})"

        if features['flesch_reading_ease'] > 70:
            feature_contributions['easy_readability'] = f"Very easy reading level ({features['flesch_reading_ease']:.1f})"

        if features['multiple_punct'] > 0:
            feature_contributions['multiple_punctuation'] = f"Multiple punctuation marks ({features['multiple_punct']})"

        # Decision structure
        decision = {
            'decision': 'REAL' if pred == 1 else 'FAKE',
            'confidence': float(proba if pred == 1 else 1 - proba),
            'reasoning': {
                'agent_name': 'Stylistic Analysis Agent',
                'primary_factors': feature_contributions,
                'feature_summary': {
                    'exclamation_density': float(features['exclamation_density']),
                    'caps_ratio': float(features['caps_ratio']),
                    'clickbait_count': int(features['clickbait_count']),
                    'emotional_word_count': int(features['emotional_word_count']),
                    'flesch_reading_ease': float(features['flesch_reading_ease'])
                },
                'reasoning_text': f"Based on stylistic analysis: {len(feature_contributions)} red flags detected" if pred == 0
                                 else f"Stylistic patterns appear normal with {len(feature_contributions)} minor concerns"
            }
        }

        decisions.append(decision)

    return decisions

# Generate decisions for all splits
train_proba = model.predict_proba(train_features)[:, 1]
val_proba = model.predict_proba(val_features)[:, 1]
test_proba = model.predict_proba(test_features)[:, 1]

train_pred = model.predict(train_features)
val_pred = model.predict(val_features)
test_pred = model.predict(test_features)

train_decisions = generate_model_reasoning(train_features, train_proba, train_pred, train_features.columns)
val_decisions = generate_model_reasoning(val_features, val_proba, val_pred, val_features.columns)
test_decisions = generate_model_reasoning(test_features, test_proba, test_pred, test_features.columns)

# ============================================================================
# Evaluate and Save
# ============================================================================
print("\n[6/6] Evaluating and saving model outputs...")

train_acc = accuracy_score(train_labels, train_pred)
val_acc = accuracy_score(val_labels, val_pred)
test_acc = accuracy_score(test_labels, test_pred)

print(f"\n📊modle 1 Performance (for reference only):")
print(f"  Train Accuracy: {train_acc*100:.2f}%")
print(f"  Val Accuracy:   {val_acc*100:.2f}%")
print(f"  Test Accuracy:  {test_acc*100:.2f}%")

print(f"\n📋 Detailed Test Results:")
print(classification_report(test_labels, test_pred, target_names=['Fake', 'Real']))

# Feature importance
print(f"\n🔍 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': train_features.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# Save model
with open('/content/models/model1_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save Model outputs for Coordinator
agent1_outputs = {
    'train': {
        'decisions': train_decisions,
        'features': train_features.values,
        'feature_names': list(train_features.columns),
        'raw_proba': train_proba,
        'raw_pred': train_pred
    },
    'val': {
        'decisions': val_decisions,
        'features': val_features.values,
        'feature_names': list(val_features.columns),
        'raw_proba': val_proba,
        'raw_pred': val_pred
    },
    'test': {
        'decisions': test_decisions,
        'features': test_features.values,
        'feature_names': list(test_features.columns),
        'raw_proba': test_proba,
        'raw_pred': test_pred
    }
}

with open('/content/models/model1_outputs.pkl', 'wb') as f:
    pickle.dump(model1_outputs, f)

print("\n✓ Model saved to '/content/models/model1_model.pkl'")
print("✓ model outputs saved to '/content/models/model1_outputs.pkl'")

# Display sample reasoning
print("\n" + "="*70)
print("📝 SAMPLE model 1 REASONING (First 3 test samples):")
print("="*70)
for i in range(min(3, len(test_decisions))):
    dec = test_decisions[i]
    actual = "REAL" if test_labels[i] == 1 else "FAKE"
    print(f"\nSample {i+1}:")
    print(f"  model Decision: {dec['decision']} (Confidence: {dec['confidence']:.2%})")
    print(f"  Actual Label: {actual}")
    print(f"  Reasoning: {dec['reasoning']['reasoning_text']}")
    print(f"  Key Factors: {list(dec['reasoning']['primary_factors'].keys())}")
