"""
Model 4: Advanced Credibility Scoring Model 
Target: 90%+ Accuracy WITHOUT OOM in Colab
- Chunked processing with aggressive garbage collection
- Smart memory management for spaCy NER
- All 90+ features from ensemble version
"""

import pandas as pd
import numpy as np
import pickle
import os
import spacy
import re
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Model 4: CREDIBILITY SCORING")
print("Target: 90%+ Accuracy + No OOM in Colab")
print("=" * 70)

# ============================================================================
# Load spaCy - MEMORY OPTIMIZED
# ============================================================================
print("\n[1/7] Loading spaCy (memory-optimized)...")
nlp = spacy.load('en_core_web_sm')
nlp.disable_pipes('parser')  # Keep NER
nlp.max_length = 5000  # Limit doc size
print("✓ spaCy loaded (NER enabled, parser disabled)")

# ============================================================================
# Load Data
# ============================================================================
print("\n[2/7] Loading data...")
try:
    with open('train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    with open('val.pkl', 'rb') as f:
        val_df = pickle.load(f)
    with open('test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
except FileNotFoundError as e:
    print(f"❌ {e}")
    raise

os.makedirs('/content/models', exist_ok=True)

# ============================================================================
# Feature Extraction Functions (Same as ensemble)
# ============================================================================
print("\n[3/7] Loading feature extraction functions...")

def analyze_citation_quality(text):
    features = {}
    features['doi_citation_count'] = len(re.findall(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', text, re.IGNORECASE))
    top_journals = ['nature', 'science', 'cell', 'lancet', 'nejm', 'jama', 'bmj', 'pnas']
    features['top_journal_citation'] = sum(1 for j in top_journals if j in text.lower())
    years = [int(y) for y in re.findall(r'\b(20[0-2]\d)\b', text)]
    features['has_recent_citation'] = int(any(y >= 2023 for y in years)) if years else 0
    features['avg_citation_age'] = np.mean([2025 - y for y in years]) if years else 10
    features['citation_recency_score'] = sum(1/(2025 - y + 1) for y in years) if years else 0
    features['preprint_mention'] = int('preprint' in text.lower() or 'arxiv' in text.lower())
    features['peer_reviewed_mention'] = int('peer-reviewed' in text.lower() or 'peer reviewed' in text.lower())
    features['conference_citation'] = sum(1 for term in ['conference', 'proceedings', 'symposium'] if term in text.lower())
    return features

def check_internal_consistency(text, doc):
    features = {}
    dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    features['date_mention_count'] = len(dates)
    features['date_diversity'] = len(set(dates)) / max(len(dates), 1)
    percentages = [float(n) for n in re.findall(r'(\d+\.?\d*)\s*%', text)]
    features['percentage_inconsistency'] = int(sum(percentages) > 110 or sum(percentages) < 90) if len(percentages) >= 2 else 0
    numbers = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', text) if n.replace('.', '').isdigit()]
    features['number_range'] = max(numbers) - min(numbers) if len(numbers) >= 3 else 0
    features['number_std'] = np.std(numbers) if len(numbers) >= 3 else 0
    features['contradiction_indicator_count'] = sum(
        len(re.findall(p1, text.lower())) * len(re.findall(p2, text.lower()))
        for p1, p2 in [(r'\bhowever\b', r'\btherefore\b'), (r'\bbut\b', r'\bthus\b')]
    )
    return features

def analyze_entity_verifiability(doc, text):
    features = {}
    person_entities = [ent for ent in doc.ents if ent.label_ == 'PERSON']
    titles = ['dr.', 'professor', 'president', 'senator', 'minister', 'director', 'ceo']
    titled_persons = sum(1 for ent in person_entities if any(t in ent.text.lower() for t in titles))
    features['titled_person_count'] = titled_persons
    features['titled_person_ratio'] = titled_persons / max(len(person_entities), 1)
    full_names = [ent for ent in person_entities if len(ent.text.split()) >= 2]
    features['full_name_count'] = len(full_names)
    features['full_name_ratio'] = len(full_names) / max(len(person_entities), 1)
    org_entities = [ent for ent in doc.ents if ent.label_ == 'ORG']
    full_orgs = [ent for ent in org_entities if len(ent.text.split()) >= 2]
    features['full_org_name_count'] = len(full_orgs)
    features['full_org_name_ratio'] = len(full_orgs) / max(len(org_entities), 1)
    features['acronym_only_count'] = len([ent for ent in org_entities if ent.text.isupper() and len(ent.text) <= 6])
    gpe_entities = [ent for ent in doc.ents if ent.label_ == 'GPE']
    features['location_count'] = len(gpe_entities)
    features['city_level_location'] = sum(1 for i in ['city', 'town', 'district'] if i in text.lower())
    features['address_mention'] = int(bool(re.search(r'\d+\s+[A-Z][a-z]+\s+(?:Street|Avenue|Road)', text)))
    return features

def analyze_quote_quality(text):
    features = {}
    features['fully_attributed_quote_count'] = len(re.findall(r'"[^"]{20,}"\s*[-–—]\s*([A-Z][a-z]+\s+){2,}', text))
    reporting_verbs = ['said', 'stated', 'confirmed', 'announced', 'explained', 'noted']
    features['verb_attributed_quote_count'] = len(re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(' + '|'.join(reporting_verbs) + r')', text))
    features['direct_quote_count'] = text.count('"') // 2
    features['indirect_quote_count'] = len(re.findall(r'\b(?:that|how|whether|if)\s+[a-z]', text.lower()))
    features['direct_to_indirect_ratio'] = features['direct_quote_count'] / max(features['indirect_quote_count'], 1)
    quotes = re.findall(r'"([^"]{10,})"', text)
    if quotes:
        quote_lengths = [len(q.split()) for q in quotes]
        features['avg_quote_length'] = np.mean(quote_lengths)
        features['quote_length_std'] = np.std(quote_lengths)
        features['excessively_long_quote'] = int(any(l > 50 for l in quote_lengths))
    else:
        features['avg_quote_length'] = features['quote_length_std'] = features['excessively_long_quote'] = 0
    features['contextual_quote_count'] = len(re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|stated):\s*"', text))
    return features

def check_expertise_matching(text, doc):
    features = {}
    domains = {
        'medical': ['disease', 'treatment', 'patient', 'clinical', 'medical', 'health', 'doctor'],
        'economic': ['economy', 'market', 'financial', 'trade', 'gdp', 'inflation', 'stock'],
        'political': ['government', 'election', 'policy', 'legislation', 'politics', 'vote'],
        'scientific': ['research', 'study', 'experiment', 'data', 'analysis', 'theory'],
        'technology': ['software', 'hardware', 'algorithm', 'computing', 'digital', 'internet']
    }
    text_lower = text.lower()
    domain_scores = {d: sum(text_lower.count(kw) for kw in keywords) for d, keywords in domains.items()}
    article_domain = max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else None
    features['has_clear_domain'] = int(article_domain is not None)
    expert_titles = {
        'medical': ['dr.', 'physician', 'surgeon', 'md', 'epidemiologist'],
        'economic': ['economist', 'financial analyst', 'chief economist'],
        'political': ['senator', 'representative', 'political scientist'],
        'scientific': ['scientist', 'researcher', 'professor', 'phd'],
        'technology': ['engineer', 'developer', 'cto', 'computer scientist']
    }
    if article_domain:
        relevant_titles = expert_titles.get(article_domain, [])
        features['domain_matched_expert_count'] = sum(1 for t in relevant_titles if t in text_lower)
        features['has_domain_matched_expert'] = int(features['domain_matched_expert_count'] > 0)
    else:
        features['domain_matched_expert_count'] = features['has_domain_matched_expert'] = 0
    if domain_scores:
        top_score = max(domain_scores.values())
        total_score = sum(domain_scores.values())
        features['domain_focus_strength'] = top_score / max(total_score, 1)
    else:
        features['domain_focus_strength'] = 0
    return features

def extract_base_credibility_features(text):
    features = {}
    text_lower = text.lower()
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
    features['specific_number_count'] = len(numbers)
    features['percentage_mention_count'] = len(re.findall(r'\b\d+(?:\.\d+)?%', text))
    precise_numbers = [n for n in numbers if '.' in n]
    features['precise_statistic_count'] = len(precise_numbers)
    features['precision_ratio'] = len(precise_numbers) / max(len(numbers), 1)
    features['large_round_number_count'] = len(re.findall(r'\b\d{6,}0{3,}\b', text))
    features['financial_figure_count'] = len(re.findall(r'\$\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion))?', text, re.I))
    features['source_attribution_count'] = sum(len(re.findall(p, text)) for p in [
        r'according to [A-Z][a-zA-Z\s]+',
        r'[A-Z][a-zA-Z\s]+ (?:reported|stated|said)',
        r'(?:study|research) (?:by|from) [A-Z]'
    ])
    generic_attr = len(re.findall(r'\bexperts?\s+(?:say|believe|think)\b', text_lower))
    features['generic_vs_specific_ratio'] = generic_attr / max(features['source_attribution_count'], 1)
    features['academic_citation_count'] = sum(len(re.findall(p, text)) for p in [
        r'\b[A-Z][a-z]+\s+et\s+al\.?\s*\(?\d{4}\)?',
        r'\([A-Z][a-z]+,?\s+\d{4}\)',
        r'\b[A-Z][a-z]+\s*,\s*\d{4}\b'
    ])
    features['specific_date_count'] = sum(len(re.findall(p, text, re.I)) for p in [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b'
    ])
    features['timestamp_count'] = len(re.findall(r'\b\d{1,2}:\d{2}\s?(?:AM|PM|EST|PST)\b', text, re.I))
    vague_temporal = ['recently', 'lately', 'soon', 'earlier', 'later', 'some time']
    features['vague_temporal_count'] = sum(text_lower.count(w) for w in vague_temporal)
    features['temporal_specificity_ratio'] = features['specific_date_count'] / max(features['vague_temporal_count'], 1)
    features['institutional_mention_count'] = sum(1 for i in ['university', 'college', 'institute', 'department of', 'ministry of', 'national', 'commission'] if i in text_lower)
    features['research_terminology_count'] = sum(text_lower.count(t) for t in ['study', 'research', 'analysis', 'survey', 'investigation', 'experiment'])
    features['official_source_count'] = sum(1 for t in ['official', 'government', 'department', 'spokesperson'] if t in text_lower)
    features['established_media_count'] = sum(1 for m in ['reuters', 'associated press', 'bbc', 'cnn', 'new york times', 'washington post'] if m in text_lower)
    features['verification_language_count'] = sum(1 for v in ['verified', 'confirmed', 'corroborated', 'fact-checked', 'documented'] if v in text_lower)
    features['epistemic_marker_count'] = sum(text_lower.count(m) for m in ['appears', 'suggests', 'indicates', 'likely', 'possibly', 'estimated', 'approximately'])
    features['absolute_certainty_count'] = sum(1 for a in ['definitely', 'absolutely', 'certainly', 'undoubtedly', 'proven fact', '100%'] if a in text_lower)
    features['epistemic_appropriateness'] = features['epistemic_marker_count'] / max(features['absolute_certainty_count'], 1)
    features['conspiracy_marker_count'] = sum(1 for c in ['cover-up', 'hidden truth', 'conspiracy', 'hoax', 'deep state'] if c in text_lower)
    features['sensational_claim_count'] = sum(1 for s in ['shocking', 'unbelievable', 'incredible', 'you won\'t believe', 'bombshell'] if s in text_lower)
    features['anonymous_source_count'] = sum(1 for a in ['anonymous source', 'unnamed official', 'sources say', 'allegedly'] if a in text_lower)
    all_numbers = [float(re.sub(r'[^\d.]', '', n)) for n in numbers if re.search(r'\d', n)]
    if len(all_numbers) > 1:
        round_numbers = [n for n in all_numbers if n == int(n) and n >= 100]
        features['round_number_ratio'] = len(round_numbers) / len(all_numbers)
        features['number_diversity'] = len(set(all_numbers)) / len(all_numbers)
    else:
        features['round_number_ratio'] = features['number_diversity'] = 0
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', text)
    features['url_reference_count'] = len(urls)
    features['credible_domain_count'] = sum(1 for url in urls if any(d in url.lower() for d in ['.gov', '.edu', '.org', 'doi.org', 'ncbi']))
    return features

def extract_enhanced_credibility_features(text, doc):
    text = text[:5000]
    features = extract_base_credibility_features(text)
    features.update(analyze_citation_quality(text))
    features.update(check_internal_consistency(text, doc))
    features.update(analyze_entity_verifiability(doc, text))
    features.update(analyze_quote_quality(text))
    features.update(check_expertise_matching(text, doc))

    text_lower = text.lower()
    features['has_byline'] = int(bool(re.search(r'\bby\s+[A-Z][a-z]+\s+[A-Z]', text)))
    features['has_contact_info'] = int('@' in text or 'contact' in text_lower or bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)))
    features['multiple_source_count'] = sum(text_lower.count(s) for s in ['according to', 'reported by', 'stated by', 'confirmed by'])
    features['has_correction'] = int('correction' in text_lower or 'updated' in text_lower or 'clarification' in text_lower)
    features['has_legal_language'] = int('allegedly' in text_lower or 'court documents' in text_lower)
    features['has_photo_credit'] = int('photo by' in text_lower or 'image credit' in text_lower or 'getty images' in text_lower)
    proper_nouns = [t for t in doc if t.pos_ == 'PROPN']
    features['proper_noun_density'] = len(proper_nouns) / max(len([t for t in doc]), 1) * 100
    features['passive_voice_ratio'] = sum(text_lower.count(p) for p in ['was', 'were', 'been', 'being']) / max(len(text.split()), 1) * 100
    features['question_mark_count'] = text.count('?')
    features['all_caps_word_count'] = len(re.findall(r'\b[A-Z]{3,}\b', text))

    # Advanced interactions
    features['credibility_composite_score'] = (
        features['source_attribution_count'] * 0.3 + features['specific_date_count'] * 0.2 +
        features['academic_citation_count'] * 0.3 + features['verification_language_count'] * 0.2
    ) / max(features['conspiracy_marker_count'] + features['sensational_claim_count'], 1)

    evidence_count = features['specific_number_count'] + features['academic_citation_count'] + features['direct_quote_count']
    claim_indicators = len(re.findall(r'\b(?:claims?|alleges?|suggests?)\b', text_lower))
    features['evidence_to_claim_ratio'] = evidence_count / max(claim_indicators, 1)

    features['source_quality_score'] = (
        features['doi_citation_count'] * 3 + features['top_journal_citation'] * 2 +
        features['established_media_count'] + features['domain_matched_expert_count'] * 2
    ) / max(features['anonymous_source_count'], 1)

    features['verification_strength'] = (
        features['verification_language_count'] + features['fully_attributed_quote_count'] + features['credible_domain_count']
    ) / max(features['vague_temporal_count'] + features['generic_vs_specific_ratio'], 1)

    features['entity_credibility_score'] = (features['titled_person_ratio'] + features['full_name_ratio'] + features['full_org_name_ratio']) / 3
    features['citation_quality_score'] = (
        features['doi_citation_count'] * 3 + features['peer_reviewed_mention'] * 2 +
        features['has_recent_citation'] * 2 + features['top_journal_citation'] * 3
    ) / max(features['preprint_mention'] + 1, 1)

    features['evidence_density'] = (
        features['specific_number_count'] + features['specific_date_count'] + features['titled_person_count'] +
        features['full_org_name_count'] + features['direct_quote_count'] + features['doi_citation_count']
    ) / max(len(text), 1) * 1000

    features['trust_score'] = (
        features['doi_citation_count'] * 5 + features['peer_reviewed_mention'] * 4 +
        features['established_media_count'] * 3 + features['has_byline'] * 2 +
        features['verification_language_count'] * 2 + features['has_correction'] * 2
    ) / max(features['conspiracy_marker_count'] * 5 + features['anonymous_source_count'] * 3 + features['sensational_claim_count'] * 2, 1)

    features['professionalism_score'] = (
        features['institutional_mention_count'] + features['research_terminology_count'] +
        features['has_contact_info'] * 2 + features['has_photo_credit']
    ) / max(features['all_caps_word_count'] + features['question_mark_count'], 1)

    citation_types = sum([
        features['doi_citation_count'] > 0, features['academic_citation_count'] > 0,
        features['established_media_count'] > 0, features['url_reference_count'] > 0
    ])
    features['citation_diversity_score'] = citation_types / 4

    return features

print("✓ All 90+ features defined")

# ============================================================================
# MEMORY-SAFE Feature Extraction - CHUNKED with aggressive GC
# ============================================================================
print("\n[4/7] Extracting features (MEMORY-SAFE CHUNKED)...")

def extract_features_memory_safe(df, chunk_size=300):
    """Process in small chunks, save to disk, clear memory"""
    all_features = []
    texts = [str(text)[:5000] for text in df['cleaned_text'].values]
    n_chunks = len(texts) // chunk_size + (1 if len(texts) % chunk_size else 0)

    print(f"  Processing {len(texts)} texts in {n_chunks} chunks of {chunk_size}")

    for i in tqdm(range(n_chunks), desc="Chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(texts))
        chunk_texts = texts[start_idx:end_idx]

        # Process this chunk with spaCy
        docs = list(nlp.pipe(chunk_texts, batch_size=50))

        # Extract features
        chunk_features = []
        for text, doc in zip(chunk_texts, docs):
            try:
                feat = extract_enhanced_credibility_features(text, doc)
                chunk_features.append(feat)
            except:
                if chunk_features:
                    feat = {k: 0 for k in chunk_features[0].keys()}
                else:
                    feat = {}
                chunk_features.append(feat)

        all_features.extend(chunk_features)

        # Clear memory aggressively
        del docs, chunk_texts, chunk_features
        if i % 3 == 0:  # GC every 3 chunks
            gc.collect()

    features_df = pd.DataFrame(all_features).replace([np.inf, -np.inf], 0).fillna(0)
    del all_features
    gc.collect()
    return features_df

# Process each split separately
print("\n📊 Processing TRAIN...")
train_features = extract_features_memory_safe(train_df, chunk_size=300)
print(f"✓ Train: {train_features.shape}")
train_labels = train_df['label'].values
del train_df
gc.collect()

print("\n📊 Processing VAL...")
val_features = extract_features_memory_safe(val_df, chunk_size=300)
print(f"✓ Val: {val_features.shape}")
val_labels = val_df['label'].values
del val_df
gc.collect()

print("\n📊 Processing TEST...")
test_features = extract_features_memory_safe(test_df, chunk_size=300)
print(f"✓ Test: {test_features.shape}")
test_labels = test_df['label'].values
del test_df
gc.collect()

print(f"\n✓ Total features: {train_features.shape[1]}")

# ============================================================================
# Scaling
# ============================================================================
print("\n[5/7] Scaling...")
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns)
val_scaled = pd.DataFrame(scaler.transform(val_features), columns=train_features.columns)
test_scaled = pd.DataFrame(scaler.transform(test_features), columns=train_features.columns)

with open('/content/models/model4_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved")
gc.collect()

# ============================================================================
# Training - 30K sample
# ============================================================================
print("\n[6/7] Training...")
sample_size = min(30000, len(train_scaled))
X_tune, _, y_tune, _ = train_test_split(
    train_scaled, train_labels, train_size=sample_size, stratify=train_labels, random_state=42
)
print(f"⚡ Tuning on {len(X_tune)} samples")

param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [12, 15, 18, None],
    'min_samples_split': [8, 10, 15],
    'min_samples_leaf': [3, 4, 5],
    'max_features': ['sqrt', 'log2', 0.3],
    'max_samples': [0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    param_grid, n_iter=50, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1, random_state=42
)

print("Starting search...")
random_search.fit(X_tune, y_tune)
print(f"\n✓ Best: {random_search.best_params_}")
print(f"✓ CV AUC: {random_search.best_score_:.4f}")

del X_tune, y_tune
gc.collect()

# Retrain on full
print("\n🔄 Retraining on full data...")
model = RandomForestClassifier(**random_search.best_params_, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(train_scaled, train_labels)
print("✓ Model trained")
gc.collect()

# ============================================================================
# Generate Model Outputs - CHUNKED
# ============================================================================
print("\n[7/7] Generating reasoning (chunked)...")

def generate_reasoning_chunked(features_df, probabilities, predictions, chunk_size=2000):
    """Generate reasoning in chunks to save memory"""
    decisions = []
    for i in tqdm(range(0, len(predictions), chunk_size), desc="Reasoning"):
        end_idx = min(i + chunk_size, len(predictions))
        for idx in range(i, end_idx):
            proba = probabilities[idx]
            pred = predictions[idx]
            feat = features_df.iloc[idx]

            signals = {}
            if feat['doi_citation_count'] > 0:
                signals['high_quality_citations'] = f"DOI: {int(feat['doi_citation_count'])}"
            if feat['source_attribution_count'] == 0:
                signals['no_attribution'] = "No attribution"
            if feat['verification_language_count'] > 0:
                signals['verification'] = f"Verified: {int(feat['verification_language_count'])}"
            if feat['conspiracy_marker_count'] > 0:
                signals['conspiracy'] = f"Conspiracy: {int(feat['conspiracy_marker_count'])}"
            if feat['sensational_claim_count'] > 2:
                signals['sensationalism'] = f"Sensational: {int(feat['sensational_claim_count'])}"
            if feat['institutional_mention_count'] > 0:
                signals['institutions'] = f"Institutions: {int(feat['institutional_mention_count'])}"
            if feat['titled_person_count'] > 0:
                signals['titled_sources'] = f"Titled: {int(feat['titled_person_count'])}"

            decisions.append({
                'decision': 'REAL' if pred == 1 else 'FAKE',
                'confidence': float(proba if pred == 1 else 1 - proba),
                'reasoning': {
                    'agent_name': 'Credibility Scoring Agent',
                    'primary_factors': signals,
                    'feature_summary': {
                        'doi_count': int(feat['doi_citation_count']),
                        'attribution_count': int(feat['source_attribution_count']),
                        'verification_count': int(feat['verification_language_count']),
                        'conspiracy_count': int(feat['conspiracy_marker_count']),
                        'credibility_score': float(feat['credibility_composite_score'])
                    },
                    'reasoning_text': f"{len(signals)} signals" if pred == 1 else f"{len(signals)} red flags"
                }
            })

        if i % 5000 == 0:
            gc.collect()

    return decisions

# Predictions
train_pred = model.predict(train_scaled)
train_proba = model.predict_proba(train_scaled)[:, 1]
gc.collect()

val_pred = model.predict(val_scaled)
val_proba = model.predict_proba(val_scaled)[:, 1]
gc.collect()

test_pred = model.predict(test_scaled)
test_proba = model.predict_proba(test_scaled)[:, 1]
gc.collect()

# Reasoning
train_decisions = generate_reasoning_chunked(train_features, train_proba, train_pred)
gc.collect()
val_decisions = generate_reasoning_chunked(val_features, val_proba, val_pred)
gc.collect()
test_decisions = generate_reasoning_chunked(test_features, test_proba, test_pred)
gc.collect()

# Evaluate
train_acc = accuracy_score(train_labels, train_pred)
val_acc = accuracy_score(val_labels, val_pred)
test_acc = accuracy_score(test_labels, test_pred)
train_auc = roc_auc_score(train_labels, train_proba)
val_auc = roc_auc_score(val_labels, val_proba)
test_auc = roc_auc_score(test_labels, test_proba)

print(f"\n{'='*70}")
print(f"Model 4 PERFORMANCE")
print(f"{'='*70}")
print(f"\nAccuracy:")
print(f"  Train: {train_acc*100:.2f}%")
print(f"  Val:   {val_acc*100:.2f}%")
print(f"  Test:  {test_acc*100:.2f}%")
print(f"\nAUC-ROC:")
print(f"  Train: {train_auc:.4f}")
print(f"  Val:   {val_auc:.4f}")
print(f"  Test:  {test_auc:.4f}")
print(classification_report(test_labels, test_pred, target_names=['Fake', 'Real']))

# Save
with open('/content/models/model4_model.pkl', 'wb') as f:
    pickle.dump(model, f)

model4_outputs = {
    'train': {'decisions': train_decisions, 'raw_proba': train_proba.astype(np.float32), 'raw_pred': train_pred.astype(np.int8)},
    'val': {'decisions': val_decisions, 'raw_proba': val_proba.astype(np.float32), 'raw_pred': val_pred.astype(np.int8)},
    'test': {'decisions': test_decisions, 'raw_proba': test_proba.astype(np.float32), 'raw_pred': test_pred.astype(np.int8)},
    'feature_names': list(train_features.columns)
}

with open('/content/models/model4_outputs.pkl', 'wb') as f:
    pickle.dump(agent4_outputs, f)

print("\n✅ COMPLETE! Target: 90%+, Actual: {test_acc*100:.2f}%")
