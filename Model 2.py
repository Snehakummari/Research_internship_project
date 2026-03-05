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
print("MODEL 2: ADVANCED LINGUISTIC ANALYSIS (MODEL)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n[1/7] Loading spaCy model...")
try:
    nlp = spacy.load('en_core_web_sm')
    print("✓ spaCy loaded successfully")
except:
    print("❌ Error: spaCy model not found!")
    exit()

os.makedirs('/content/models', exist_ok=True)

print("\n[2/7] Defining comprehensive linguistic feature extraction...")

def extract_comprehensive_linguistic_features(text, max_chars=5000):
    features = {}
    text_truncated = text[:max_chars]
    doc = nlp(text_truncated)

    total_tokens = len(doc)
    if total_tokens == 0:
        return {f'feature_{i}': 0 for i in range(75)}

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

    sentences = list(doc.sents)
    sent_lengths = [len(sent) for sent in sentences]

    features['num_sentences'] = len(sentences)
    features['avg_sent_length'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sent_length_std'] = np.std(sent_lengths) if sent_lengths else 0
    features['max_sent_length'] = max(sent_lengths) if sent_lengths else 0
    features['min_sent_length'] = min(sent_lengths) if sent_lengths else 0
    features['sent_length_cv'] = (features['sent_length_std'] / max(features['avg_sent_length'], 1))
    features['sent_length_range'] = features['max_sent_length'] - features['min_sent_length']

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

    word_lengths = [len(token.text) for token in word_tokens]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
    features['word_length_std'] = np.std(word_lengths) if word_lengths else 0
    features['max_word_length'] = max(word_lengths) if word_lengths else 0

    long_words = [w for w in word_lengths if w > 6]
    features['long_word_ratio'] = len(long_words) / max(len(word_lengths), 1)

    very_long_words = [w for w in word_lengths if w > 10]
    features['very_long_word_ratio'] = len(very_long_words) / max(len(word_lengths), 1)

    passive_count = 0
    for i, token in enumerate(doc[:-1]):
        if token.lemma_ == 'be' and doc[i+1].tag_ in ['VBN', 'VBD']:
            passive_count += 1
    features['passive_voice_ratio'] = passive_count / max(len(sentences), 1)

    subordinate_conj = [token for token in doc if token.pos_ == 'SCONJ']
    features['subordinate_clause_ratio'] = len(subordinate_conj) / max(len(sentences), 1)

    prep_phrases = [token for token in doc if token.pos_ == 'ADP']
    features['prep_phrase_density'] = len(prep_phrases) / total_tokens

    relative_pronouns = [t for t in doc if t.text.lower() in ['who','which','that'] and t.dep_ in ['nsubj','dobj']]
    features['relative_clause_density'] = len(relative_pronouns) / max(len(sentences), 1)

    coord_conj = [token for token in doc if token.pos_ == 'CCONJ']
    features['clause_density'] = (len(coord_conj) + len(subordinate_conj)) / max(len(sentences), 1)

    features['punct_ratio'] = len([t for t in doc if t.is_punct]) / total_tokens
    features['comma_ratio'] = text_truncated.count(',') / total_tokens
    features['semicolon_ratio'] = text_truncated.count(';') / total_tokens
    features['colon_ratio'] = text_truncated.count(':') / total_tokens
    features['dash_ratio'] = (text_truncated.count('—') + text_truncated.count('--')) / total_tokens
    features['ellipsis_count'] = text_truncated.count('...') + text_truncated.count('…')
    features['parenthesis_ratio'] = (text_truncated.count('(') + text_truncated.count(')')) / total_tokens

    features['quote_count'] = (text_truncated.count('"') + text_truncated.count("'")) / 2
    features['quote_density'] = features['quote_count'] / max(len(sentences), 1)

    attribution_verbs = ['said','says','stated','states','claims','claimed','reported','reports','announced','declares','declared','according','noted','explained','confirmed']
    features['attribution_count'] = sum(1 for verb in attribution_verbs if verb in text_lower)
    features['attribution_density'] = features['attribution_count'] / max(len(sentences), 1)

    features['number_count'] = len([token for token in doc if token.like_num])
    features['number_density'] = features['number_count'] / total_tokens
    features['number_per_sentence'] = features['number_count'] / max(len(sentences), 1)

    features['question_ratio'] = len([s for s in sentences if '?' in s.text]) / max(len(sentences), 1)
    features['exclamation_ratio'] = len([s for s in sentences if '!' in s.text]) / max(len(sentences), 1)

    first_person = len([t for t in doc if t.text.lower() in ['i','we','me','us','my','our','mine','ours']])
    second_person = len([t for t in doc if t.text.lower() in ['you','your','yours']])
    third_person = len([t for t in doc if t.text.lower() in ['he','she','it','they','him','her','them','his','their']])

    features['first_person_ratio'] = first_person / total_tokens
    features['second_person_ratio'] = second_person / total_tokens
    features['third_person_ratio'] = third_person / total_tokens
    features['pronoun_perspective_diversity'] = len([x for x in [first_person,second_person,third_person] if x>0])

    features['coord_conjunction_ratio'] = len(coord_conj) / total_tokens
    features['subord_conjunction_ratio'] = len(subordinate_conj) / total_tokens

    return features

print("✓ Comprehensive linguistic feature extraction defined")
