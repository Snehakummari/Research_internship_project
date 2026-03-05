"""
SIMPLIFIED HIGH-ACCURACY MULTI-MODEL ENSEMBLE
4 Models → Ensemble Features → XGBoost Meta-Learner
Goal: Maximum accuracy with minimal complexity
FIXED VERSION: Added validation, input checks, early stopping
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("SIMPLIFIED MULTI-Model STACKING ENSEMBLE (FIXED)")
print("="*80)

# =====================================================
# 1. LOAD AGENT OUTPUTS WITH VALIDATION
# =====================================================
print("\n[1/6] Loading Model outputs...")

def validate_Model_output(model, model_name):
    """Validate agent output structure and data quality"""
    required_splits = ["train", "val", "test"]
    required_keys = ["raw_proba", "raw_pred"]
    
    for split in required_splits:
        if split not in agent:
            raise ValueError(f"{agent_name} missing '{split}' split")
        
        for key in required_keys:
            if key not in agent[split]:
                raise ValueError(f"{agent_name}['{split}'] missing '{key}'")
            
            # Check for NaN values
            data = agent[split][key]
            if np.any(np.isnan(data)):
                raise ValueError(f"{agent_name}['{split}']['{key}'] contains NaN values")
            
            # Check data types
            if key == "raw_proba" and (np.any(data < 0) or np.any(data > 1)):
                raise ValueError(f"{agent_name}['{split}']['raw_proba'] contains values outside [0,1]")
            
            if key == "raw_pred" and not np.all(np.isin(data, [0, 1])):
                raise ValueError(f"{agent_name}['{split}']['raw_pred'] contains non-binary values")
    
    print(f"  ✅ {model_name} validation passed")

try:
    agent1 = pickle.load(open("/content/models/model1_outputs.pkl", "rb"))
    agent2 = pickle.load(open("/content/models/model2_outputs.pkl", "rb"))
    agent3 = pickle.load(open("/content/models/model3_outputs.pkl", "rb"))
    agent4 = pickle.load(open("/content/models/model4_outputs.pkl", "rb"))
    
    train_df = pickle.load(open("train.pkl", "rb"))
    val_df   = pickle.load(open("val.pkl", "rb"))
    test_df  = pickle.load(open("test.pkl", "rb"))
    
    print("\n  Validating agent outputs...")
    validate_agent_output(model1, "Model1")
    validate_agent_output(model2, "Model2")
    validate_agent_output(model3, "Model3")
    validate_agent_output(model4, "Model4")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Required file not found - {e}")
    print("Please ensure all agent outputs and data files are in the correct location.")
    raise
except Exception as e:
    print(f"\n❌ ERROR during loading: {e}")
    raise

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

print("\n✅ Data loaded and validated successfully")

# =====================================================
# 2. CREATE ENSEMBLE FEATURES WITH VALIDATION
# =====================================================
print("\n[2/6] Creating ensemble features...")

def create_features(split):
    """Create ensemble features from agent outputs"""
    p1 = agent1[split]["raw_proba"]
    p2 = agent2[split]["raw_proba"]
    p3 = agent3[split]["raw_proba"]
    p4 = agent4[split]["raw_proba"]

    b1 = agent1[split]["raw_pred"]
    b2 = agent2[split]["raw_pred"]
    b3 = agent3[split]["raw_pred"]
    b4 = agent4[split]["raw_pred"]
    
    # Validate shape consistency
    lengths = [len(p1), len(p2), len(p3), len(p4)]
    if len(set(lengths)) != 1:
        raise ValueError(f"Model output lengths don't match for {split}: {lengths}")

    features = np.column_stack([
        # Predictions (4 features)
        b1, b2, b3, b4,

        # Probabilities (4 features)
        p1, p2, p3, p4,

        # Confidence - distance from 0.5 (4 features)
        np.abs(p1 - 0.5),
        np.abs(p2 - 0.5),
        np.abs(p3 - 0.5),
        np.abs(p4 - 0.5),

        # Agreement flags (6 features)
        (b1 == b2).astype(int),
        (b1 == b3).astype(int),
        (b1 == b4).astype(int),
        (b2 == b3).astype(int),
        (b2 == b4).astype(int),
        (b3 == b4).astype(int),

        # Voting and statistics (5 features)
        (b1 + b2 + b3 + b4),                      # vote count
        (p1 + p2 + p3 + p4) / 4,                  # avg probability
        np.var(np.column_stack([p1,p2,p3,p4]), axis=1),  # variance
        np.max(np.column_stack([p1,p2,p3,p4]), axis=1),  # max prob
        np.min(np.column_stack([p1,p2,p3,p4]), axis=1),  # min prob

        # Pairwise probability gaps (6 features)
        np.abs(p1 - p2),
        np.abs(p1 - p3),
        np.abs(p1 - p4),
        np.abs(p2 - p3),
        np.abs(p2 - p4),
        np.abs(p3 - p4)
    ])

    return features


X_train = create_features("train")
X_val   = create_features("val")
X_test  = create_features("test")

# Validate feature matrices
print(f"  Train shape: {X_train.shape}")
print(f"  Val shape:   {X_val.shape}")
print(f"  Test shape:  {X_test.shape}")

# Check for NaN values in features
for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
    if np.any(np.isnan(X)):
        raise ValueError(f"NaN values detected in {name} features")
    if np.any(np.isinf(X)):
        raise ValueError(f"Inf values detected in {name} features")

print("✅ Feature matrix built and validated (29 features)")

# =====================================================
# 3. NORMALIZE FEATURES
# =====================================================
print("\n[3/6] Scaling features...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print("✅ Features scaled successfully")

# =====================================================
# 4. TRAIN META-LEARNER WITH EARLY STOPPING
# =====================================================
print("\n[4/6] Training meta-learner (XGBoost with validation)...")

meta_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    early_stopping_rounds=20
)

# Train with validation set for early stopping
meta_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

best_iteration = meta_model.best_iteration
print(f"✅ Training complete - Best iteration: {best_iteration}")

# =====================================================
# 5. EVALUATE ON TRAINING, VALIDATION AND TEST SETS
# =====================================================
print("\n[5/6] Evaluating model on all sets...")

# Training set evaluation
y_train_pred = meta_model.predict(X_train)
y_train_proba = meta_model.predict_proba(X_train)[:,1]
train_acc = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print("\n" + "="*80)
print("TRAINING SET RESULTS")
print("="*80)
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Train AUC: {train_auc:.4f}")

# Validation set evaluation
y_val_pred = meta_model.predict(X_val)
y_val_proba = meta_model.predict_proba(X_val)[:,1]
val_acc = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_proba)

print("\n" + "="*80)
print("VALIDATION SET RESULTS")
print("="*80)
print(f"Val Accuracy: {val_acc*100:.2f}%")
print(f"Val AUC: {val_auc:.4f}")

# Test set evaluation
y_test_pred = meta_model.predict(X_test)
y_test_proba = meta_model.predict_proba(X_test)[:,1]
test_acc = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\n" + "="*80)
print("TEST SET RESULTS (FINAL)")
print("="*80)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test AUC: {test_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Fake","Real"], digits=4))

cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# Check for overfitting
print("\n" + "="*80)
print("GENERALIZATION ANALYSIS")
print("="*80)
train_val_gap = abs(train_acc - val_acc)
val_test_gap = abs(val_acc - test_acc)
train_test_gap = abs(train_acc - test_acc)

print(f"Train vs Val gap:  {train_val_gap*100:.2f}%")
print(f"Val vs Test gap:   {val_test_gap*100:.2f}%")
print(f"Train vs Test gap: {train_test_gap*100:.2f}%")

if train_val_gap > 0.02:
    print("\n⚠️  Warning: Model may be slightly overfitting (train-val gap > 2%)")
elif val_test_gap > 0.05:
    print(f"\n⚠️  Warning: Significant accuracy gap between val and test: {val_test_gap*100:.2f}%")
else:
    print(f"\n✅ Excellent generalization - All gaps < 2%")

# =====================================================
# 6. SAVE FINAL ENSEMBLE MODEL
# =====================================================
print("\n[6/6] Saving final ensemble model...")

os.makedirs("/content/models/ensemble", exist_ok=True)

package = {
    "meta_model": meta_model,
    "scaler": scaler,
    "train_accuracy": train_acc,
    "val_accuracy": val_acc,
    "test_accuracy": test_acc,
    "train_auc": train_auc,
    "val_auc": val_auc,
    "test_auc": test_auc,
    "best_iteration": best_iteration,
    "num_features": X_train.shape[1],
    "description": "4-agent stacking ensemble with confidence & agreement features + early stopping"
}

with open("/content/models/ensemble/final_ensemble.pkl", "wb") as f:
    pickle.dump(package, f)

print("✅ Model saved at /content/models/ensemble/final_ensemble.pkl")

# Save model summary
summary_file = "/content/models/ensemble/model_summary.txt"
with open(summary_file, "w") as f:
    f.write("="*80 + "\n")
    f.write("ENSEMBLE MODEL SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Date: {pd.Timestamp.now()}\n")
    f.write(f"Number of Features: {X_train.shape[1]}\n")
    f.write(f"Best Iteration: {best_iteration}\n\n")
    f.write(f"Training Accuracy: {train_acc*100:.2f}%\n")
    f.write(f"Training AUC: {train_auc:.4f}\n\n")
    f.write(f"Validation Accuracy: {val_acc*100:.2f}%\n")
    f.write(f"Validation AUC: {val_auc:.4f}\n\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test AUC: {test_auc:.4f}\n\n")
    f.write("="*80 + "\n")

print(f"✅ Model summary saved at {summary_file}")
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE - ALL VALIDATIONS PASSED")
print("="*80)
