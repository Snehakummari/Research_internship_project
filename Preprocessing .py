"""
STEP 1: Data Preprocessing for WELFake Dataset (RESEARCH-GRADE VERSION)
This script loads, cleans, and prepares data for Agent 1 training
Addresses: emoji preservation, label handling, duplicate detection, and more
"""

# Install required packages
!pip install pandas numpy scikit-learn tqdm

import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("=" * 60)
print("DATA PREPROCESSING PIPELINE - WELFAKE (RESEARCH-GRADE)")
print("=" * 60)

# ============================================================================
# Setup Output Directory
# ============================================================================
try:
    import google.colab
    OUTPUT_DIR = '/content'
    print("Detected: Google Colab environment")
except:
    OUTPUT_DIR = 'processed_data'
    print("Detected: Local environment")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# Load WELFake Dataset
# ============================================================================
print("\n[1/5] Loading WELFake dataset...")

try:
    df_welfake = pd.read_csv('WELFake_Dataset.csv')
    print(f"  ✓ WELFake loaded: {len(df_welfake)} samples")
    print(f"  Columns found: {df_welfake.columns.tolist()}")
except Exception as e:
    print(f"  ✗ Error loading WELFake: {e}")
    print("  Make sure 'WELFake_Dataset.csv' is in the current directory")
    exit(1)

# ============================================================================
# Data Cleaning Functions
# ============================================================================
print("\n[2/5] Defining cleaning functions...")

# Define emoji pattern for consistent detection
EMOJI_PATTERN = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]'

class LabelStandardizer:
    """Class to handle label standardization and tracking"""
    def __init__(self):
        self.unrecognized_labels = set()

    def standardize(self, label):
        """
        Robustly standardize labels to 0 (fake) and 1 (real)
        """
        if pd.isna(label):
            return None

        # Handle integer labels directly (common in WELFake)
        if isinstance(label, (int, np.integer)):
            if label == 0:
                return 0
            elif label == 1:
                return 1
            else:
                self.unrecognized_labels.add(str(label))
                return None

        # Handle string labels
        label_orig = str(label)
        label = label_orig.lower().strip()

        # Expanded label vocabulary for robustness
        fake_labels = ['fake', '0', 'false', 'f', 'unreliable', 'fake news',
                       'fakenews', 'not reliable', 'untrustworthy']
        real_labels = ['real', '1', 'true', 't', 'reliable', 'real news',
                       'realnews', 'trustworthy', 'verified']

        if label in fake_labels:
            return 0
        elif label in real_labels:
            return 1
        else:
            self.unrecognized_labels.add(label_orig)
            return None

def clean_text(text):
    """
    Clean and normalize text while PRESERVING stylistic signals

    Key improvements:
    - Keeps emojis (😂, 🔥, etc.) - important for fake news detection
    - Keeps special chars ($, %, #, @) - often used in sensational content
    - Keeps accented characters and Unicode symbols
    - Only removes control characters that break processing
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Convert to string and strip
    text = str(text).strip()

    # Remove URLs but keep placeholder for analysis
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)

    # Remove only control characters (invisible chars that break processing)
    # This preserves emojis, accents, symbols, etc.
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)

    # Normalize excessive whitespace but keep single spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

label_standardizer = LabelStandardizer()
print("✓ Cleaning functions defined")

# ============================================================================
# Process WELFake Dataset
# ============================================================================
print("\n[3/5] Processing WELFake dataset...")

# Identify text and label columns
text_col = None
label_col = None
title_col = None

# Check for specific WELFake columns first
if 'text' in df_welfake.columns:
    text_col = 'text'
if 'title' in df_welfake.columns:
    title_col = 'title'
if 'label' in df_welfake.columns:
    label_col = 'label'

# Fallback: search for text column
if text_col is None:
    text_candidates = [col for col in df_welfake.columns
                      if col.lower() in ['text', 'content', 'article', 'body', 'news']]
    if text_candidates:
        text_col = text_candidates[0]
        if len(text_candidates) > 1:
            print(f"  ⚠ Multiple text columns found: {text_candidates}")
            print(f"  Using: {text_col}")

# Fallback: search for label column
if label_col is None:
    label_candidates = [col for col in df_welfake.columns
                       if col.lower() in ['label', 'class', 'category', 'type']]
    if label_candidates:
        label_col = label_candidates[0]
    else:
        # Try to find binary column
        for col in df_welfake.columns:
            if df_welfake[col].nunique() == 2:
                label_col = col
                break

print(f"  Detected text column: {text_col}")
print(f"  Detected title column: {title_col}")
print(f"  Detected label column: {label_col}")

if text_col is None or label_col is None:
    print("✗ Could not identify text or label columns!")
    print("Available columns:", df_welfake.columns.tolist())
    exit(1)

# Intelligent title + text combination
if title_col and title_col in df_welfake.columns:
    non_empty_titles = df_welfake[title_col].notna().sum()
    title_coverage = non_empty_titles / len(df_welfake) * 100

    # Lower threshold: use titles if >5% have data
    if non_empty_titles > len(df_welfake) * 0.05:
        print(f"  Combining title and text ({non_empty_titles} titles, {title_coverage:.1f}% coverage)...")
        df_processed = pd.DataFrame({
            'text': df_welfake[title_col].fillna('') + ' ' + df_welfake[text_col].fillna(''),
            'label': df_welfake[label_col],
            'source': 'welfake'
        })
    else:
        print(f"  Title coverage too low ({title_coverage:.1f}%), using text only...")
        df_processed = pd.DataFrame({
            'text': df_welfake[text_col],
            'label': df_welfake[label_col],
            'source': 'welfake'
        })
else:
    print("  Using text column only...")
    df_processed = pd.DataFrame({
        'text': df_welfake[text_col],
        'label': df_welfake[label_col],
        'source': 'welfake'
    })

# Clean text (preserving stylistic signals)
print("  Cleaning text...")
df_processed['cleaned_text'] = df_processed['text'].apply(clean_text)

# Standardize labels
print("  Standardizing labels...")
original_labels = df_processed['label'].value_counts()
print(f"  Original label distribution: {original_labels.to_dict()}")
df_processed['label'] = df_processed['label'].apply(label_standardizer.standardize)

# Report unrecognized labels
if label_standardizer.unrecognized_labels:
    print(f"  ⚠ Unrecognized label(s): {label_standardizer.unrecognized_labels}")
    print(f"    These {len(label_standardizer.unrecognized_labels)} label(s) will be dropped")

# ============================================================================
# Remove rows with invalid data - IMPROVED TRACKING
# ============================================================================
initial_count = len(df_processed)

# Step 1: Remove null labels
before_null = len(df_processed)
df_processed = df_processed.dropna(subset=['label'])
null_removed = before_null - len(df_processed)

# Step 2: Remove empty strings
before_empty = len(df_processed)
df_processed = df_processed[df_processed['cleaned_text'].str.strip() != '']
empty_removed = before_empty - len(df_processed)

# Step 3: Remove short text (≤20 chars)
before_short = len(df_processed)
df_processed = df_processed[df_processed['cleaned_text'].str.len() > 20]
short_removed = before_short - len(df_processed)

print(f"  Removed invalid data:")
print(f"    - {null_removed} rows with null/unrecognized labels")
print(f"    - {empty_removed} rows with empty text")
print(f"    - {short_removed} rows with text ≤20 characters")

# Verify both label classes exist
unique_labels = df_processed['label'].unique()
if len(unique_labels) < 2:
    print(f"✗ Error: Only one label class found: {unique_labels}")
    print("Dataset must contain both fake (0) and real (1) labels")
    exit(1)

# ============================================================================
# Handle duplicates and conflicting labels - IMPROVED TRACKING
# ============================================================================
print("  Handling duplicates and conflicting labels...")
before_conflicts = len(df_processed)

# Step 1: Identify and remove texts with conflicting labels
text_label_counts = df_processed.groupby('cleaned_text')['label'].nunique()
conflicting_texts = text_label_counts[text_label_counts > 1].index

if len(conflicting_texts) > 0:
    print(f"    ⚠ Found {len(conflicting_texts)} texts with conflicting labels - removing all instances...")
    df_processed = df_processed[~df_processed['cleaned_text'].isin(conflicting_texts)]
    conflicts_removed = before_conflicts - len(df_processed)
else:
    conflicts_removed = 0

# Step 2: Remove exact duplicates (same text, same label)
before_dedup = len(df_processed)
df_processed = df_processed.drop_duplicates(subset=['cleaned_text'], keep='first')
duplicates_removed = before_dedup - len(df_processed)

if conflicts_removed > 0:
    print(f"    Removed {conflicts_removed} texts with conflicting labels")
if duplicates_removed > 0:
    print(f"    Removed {duplicates_removed} exact duplicate texts")

print(f"\n  ✓ Processed: {len(df_processed)} valid samples")
print(f"  ✓ Total removed: {initial_count - len(df_processed)} ({(initial_count - len(df_processed))/initial_count*100:.1f}%)")

# ============================================================================
# Calculate statistics AFTER all filtering - CRITICAL FIX
# ============================================================================
# ============================================================================
# Calculate statistics AFTER all filtering - CRITICAL FIX
# ============================================================================
# Check if we have any data left after filtering
if len(df_processed) == 0:
    print("\n✗ ERROR: No data remaining after filtering!")
    print("\nDiagnostic Information:")
    print(f"  - Started with: {initial_count} samples")
    print(f"  - Removed {null_removed} rows with null/unrecognized labels")
    print(f"  - Removed {empty_removed} rows with empty text")
    print(f"  - Removed {short_removed} rows with text ≤20 characters")
    print(f"  - Removed {conflicts_removed} rows with conflicting labels")
    print(f"  - Removed {duplicates_removed} duplicate rows")
    print(f"\n💡 Possible causes:")
    print("  1. Dataset has unrecognized label format")
    print("  2. All text is too short (≤20 chars)")
    print("  3. All rows have conflicting labels")
    print("  4. Wrong CSV file or column names")
    print(f"\n📋 Check your original data:")
    print(f"  Original label distribution: {original_labels.to_dict()}")
    exit(1)

fake_count = int((df_processed['label'] == 0).sum())
real_count = int((df_processed['label'] == 1).sum())
print(f"  Label distribution:")
print(f"    - Fake: {fake_count} ({fake_count/len(df_processed)*100:.1f}%)")
print(f"    - Real: {real_count} ({real_count/len(df_processed)*100:.1f}%)")

text_lengths = df_processed['cleaned_text'].str.len()
word_counts = df_processed['cleaned_text'].str.split().str.len()
print(f"  Text stats:")
print(f"    - Character length: min={text_lengths.min()}, max={text_lengths.max()}, mean={text_lengths.mean():.0f}")
print(f"    - Word count: min={word_counts.min()}, max={word_counts.max()}, mean={word_counts.mean():.0f}")

# Check for special characters preserved
emoji_count = df_processed['cleaned_text'].str.contains(EMOJI_PATTERN, regex=True).sum()
hashtag_count = df_processed['cleaned_text'].str.contains('#').sum()
dollar_count = df_processed['cleaned_text'].str.contains(r'\$').sum()
if emoji_count > 0:
    print(f"  ✓ Preserved {emoji_count} texts with emojis")
if hashtag_count > 0:
    print(f"  ✓ Preserved {hashtag_count} texts with hashtags")
if dollar_count > 0:
    print(f"  ✓ Preserved {dollar_count} texts with $ signs")
print(f"    - Fake: {fake_count} ({fake_count/len(df_processed)*100:.1f}%)")
print(f"    - Real: {real_count} ({real_count/len(df_processed)*100:.1f}%)")

text_lengths = df_processed['cleaned_text'].str.len()
word_counts = df_processed['cleaned_text'].str.split().str.len()
print(f"  Text stats:")
print(f"    - Character length: min={text_lengths.min()}, max={text_lengths.max()}, mean={text_lengths.mean():.0f}")
print(f"    - Word count: min={word_counts.min()}, max={word_counts.max()}, mean={word_counts.mean():.0f}")

# Check for special characters preserved
emoji_count = df_processed['cleaned_text'].str.contains(EMOJI_PATTERN, regex=True).sum()
hashtag_count = df_processed['cleaned_text'].str.contains('#').sum()
dollar_count = df_processed['cleaned_text'].str.contains(r'\$').sum()
if emoji_count > 0:
    print(f"  ✓ Preserved {emoji_count} texts with emojis")
if hashtag_count > 0:
    print(f"  ✓ Preserved {hashtag_count} texts with hashtags")
if dollar_count > 0:
    print(f"  ✓ Preserved {dollar_count} texts with $ signs")

# ============================================================================
# Keep only necessary columns - OPTIONAL CLEANUP
# ============================================================================
df_processed = df_processed[['cleaned_text', 'label', 'source']]
print(f"\n  ✓ Kept only necessary columns: {df_processed.columns.tolist()}")

# ============================================================================
# Create Train/Val/Test Splits
# ============================================================================
print("\n[4/5] Creating train/val/test splits...")

# Check minimum data requirements
if len(df_processed) < 30:
    print(f"✗ Error: Only {len(df_processed)} samples. Need at least 30 for proper splitting.")
    exit(1)
elif len(df_processed) < 100:
    print(f"⚠ Warning: Only {len(df_processed)} samples. Consider using more data.")

# Check for severe class imbalance
fake_ratio = fake_count / len(df_processed)
if fake_ratio > 0.9 or fake_ratio < 0.1:
    print(f"⚠ WARNING: Severe class imbalance ({fake_ratio*100:.1f}% fake)")
    print("  Consider class weights or resampling during training")

# Summary
print(f"\n✓ Dataset ready: {len(df_processed)} total samples")

# Shuffle the data
df_processed = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)

# Split: 70% train, 15% validation, 15% test with stratification
try:
    train_df, temp_df = train_test_split(
        df_processed,
        test_size=0.3,
        random_state=42,
        stratify=df_processed['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label']
    )
    print("✓ Stratified split successful")
except ValueError as e:
    print(f"⚠ Stratified split failed: {e}")
    print("  Using non-stratified split...")
    train_df, temp_df = train_test_split(
        df_processed,
        test_size=0.3,
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42
    )

print(f"\n✓ Data split complete:")
print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df_processed)*100:.1f}%)")
print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df_processed)*100:.1f}%)")
print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df_processed)*100:.1f}%)")

# Verify label distribution
print("\n  Label distribution per split:")
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    fake_count_split = sum(df['label'] == 0)
    real_count_split = sum(df['label'] == 1)
    fake_pct = fake_count_split / len(df) * 100
    real_pct = real_count_split / len(df) * 100
    print(f"    {name:5s} - Fake: {fake_count_split:5d} ({fake_pct:5.1f}%), Real: {real_count_split:5d} ({real_pct:5.1f}%)")

# Content-based leakage detection
train_texts = set(train_df['cleaned_text'].values)
val_texts = set(val_df['cleaned_text'].values)
test_texts = set(test_df['cleaned_text'].values)

train_val_overlap = len(train_texts & val_texts)
train_test_overlap = len(train_texts & test_texts)
val_test_overlap = len(val_texts & test_texts)

if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
    print(f"  ⚠ WARNING: Data leakage detected!")
    print(f"    Train-Val overlap: {train_val_overlap} texts")
    print(f"    Train-Test overlap: {train_test_overlap} texts")
    print(f"    Val-Test overlap: {val_test_overlap} texts")
else:
    print("  ✓ Verified: No data leakage between splits")

# ============================================================================
# Save Pickle Files
# ============================================================================
print("\n[5/5] Saving processed data...")

# Save splits
with open(f'{OUTPUT_DIR}/train.pkl', 'wb') as f:
    pickle.dump(train_df, f)
print(f"✓ Saved: {OUTPUT_DIR}/train.pkl ({len(train_df)} samples)")

with open(f'{OUTPUT_DIR}/val.pkl', 'wb') as f:
    pickle.dump(val_df, f)
print(f"✓ Saved: {OUTPUT_DIR}/val.pkl ({len(val_df)} samples)")

with open(f'{OUTPUT_DIR}/test.pkl', 'wb') as f:
    pickle.dump(test_df, f)
print(f"✓ Saved: {OUTPUT_DIR}/test.pkl ({len(test_df)} samples)")

with open(f'{OUTPUT_DIR}/combined_data.pkl', 'wb') as f:
    pickle.dump(df_processed, f)
print(f"✓ Saved: {OUTPUT_DIR}/combined_data.pkl ({len(df_processed)} samples)")

# Save comprehensive metadata
metadata = {
    'total_samples': len(df_processed),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'fake_count': fake_count,
    'real_count': real_count,
    'text_col': text_col,
    'label_col': label_col,
    'title_col': title_col,
    'avg_text_length': float(text_lengths.mean()),
    'avg_word_count': float(word_counts.mean()),
    'min_text_length': int(text_lengths.min()),
    'max_text_length': int(text_lengths.max()),
    'chars_threshold': 20,
    'null_removed': null_removed,
    'empty_removed': empty_removed,
    'short_removed': short_removed,
    'duplicates_removed': duplicates_removed,
    'conflicts_removed': conflicts_removed,
    'emoji_texts': int(emoji_count),
    'hashtag_texts': int(hashtag_count),
    'preprocessing_version': 'research_grade_v4_improved'
}

with open(f'{OUTPUT_DIR}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Saved: {OUTPUT_DIR}/metadata.pkl")

# ============================================================================
# Display Sample Data
# ============================================================================
print("\n" + "=" * 60)
print("SAMPLE DATA PREVIEW")
print("=" * 60)

print("\nFirst 3 samples from training set:")
for idx in range(min(3, len(train_df))):
    row = train_df.iloc[idx]
    label_text = "REAL" if row['label'] == 1 else "FAKE"
    text_preview = row['cleaned_text'][:200] + "..." if len(row['cleaned_text']) > 200 else row['cleaned_text']
    word_count = len(row['cleaned_text'].split())
    has_emoji = bool(re.search(EMOJI_PATTERN, row['cleaned_text']))
    has_special = bool(re.search(r'[#@$%]', row['cleaned_text']))

    print(f"\n[{idx+1}] Label: {label_text} | Words: {word_count} | Emoji: {has_emoji} | Special: {has_special}")
    print(f"    {text_preview}")

print("\n" + "=" * 60)
print("✅ RESEARCH-GRADE PREPROCESSING COMPLETE!")
print("=" * 60)
print("\n📊 Key Features:")
print("  ✓ Preserved emojis, hashtags, special characters")
print("  ✓ Lowered threshold to 20 chars (catches sensational headlines)")
print("  ✓ Removed conflicting labels (same text, different labels)")
print("  ✓ Removed exact duplicates after conflict removal")
print("  ✓ Robust label handling (int + expanded vocabulary)")
print("  ✓ Content-based leakage detection")
print("  ✓ Stratified splitting with fallback")
print("  ✓ Comprehensive metadata tracking")
print("  ✓ IMPROVED: Accurate counting at each filtering step")
print("  ✓ IMPROVED: Statistics calculated after all filtering")
print("  ✓ IMPROVED: Kept only necessary columns for training")
print("\n📁 Files saved:")
print(f"  {OUTPUT_DIR}/train.pkl ({len(train_df)} samples)")
print(f"  {OUTPUT_DIR}/val.pkl ({len(val_df)} samples)")
print(f"  {OUTPUT_DIR}/test.pkl ({len(test_df)} samples)")
print(f"  {OUTPUT_DIR}/combined_data.pkl ({len(df_processed)} samples)")
print(f"  {OUTPUT_DIR}/metadata.pkl")
print("\n" + "=" * 60)
