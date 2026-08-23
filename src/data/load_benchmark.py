import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

print("=" * 60)
print("📊 AI Act Benchmark Data Loader")
print("=" * 60)

# ============================================================
# STEP 1: Load the scenarios (with UTF-8 encoding)
# ============================================================
print("\n📂 Loading scenarios.json...")
with open("scenarios.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# The data is under the 'data' key
scenarios = data['data']
print(f"✅ Loaded {len(scenarios)} scenarios")

# Convert to DataFrame
df = pd.DataFrame(scenarios)
print(f"\n📋 Columns: {df.columns.tolist()}")
print(f"\n📊 Risk level distribution:")
print(df['risk_level'].value_counts())

# ============================================================
# STEP 2: Encode labels to numbers
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Encoding labels")
print("=" * 60)

label_encoder = LabelEncoder()
df['risk_label'] = label_encoder.fit_transform(df['risk_level'])
print(f"✅ Encoded labels:")
for i, label in enumerate(label_encoder.classes_):
    print(f"   {label} -> {i}")

# ============================================================
# STEP 3: Prepare text for classification
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Preparing text features")
print("=" * 60)

def create_feature_text(row):
    """Combine relevant text fields into one document"""
    text_parts = []
    
    # Combine these fields for classification
    fields = ['intended_use', 'system_type', 'input_data', 'domain']
    for field in fields:
        if field in row and pd.notna(row[field]):
            text_parts.append(str(row[field]))
    
    return " ".join(text_parts)

df['text'] = df.apply(create_feature_text, axis=1)
print(f"✅ Created text features for {len(df)} scenarios")

# Show example
print(f"\n📝 Example text:\n{df['text'].iloc[0][:200]}...")

# ============================================================
# STEP 4: Split data for training/testing
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Splitting train/test")
print("=" * 60)

X = df['text'].values
y = df['risk_label'].values  # Use numeric labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training: {len(X_train)} samples")
print(f"📊 Testing: {len(X_test)} samples")

# Show class distribution in train
print(f"\n📊 Training set distribution:")
train_dist = pd.Series(y_train).map(lambda x: label_encoder.inverse_transform([x])[0]).value_counts()
print(train_dist)

# ============================================================
# STEP 5: Build Model 1 - TF-IDF + XGBoost (Baseline)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Model 1 - TF-IDF + XGBoost (Baseline)")
print("=" * 60)

# Vectorize the text
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

print("🔄 Vectorizing text...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"✅ Vectorized: {X_train_vec.shape}")

# Train XGBoost
print("🔄 Training XGBoost...")
model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Model 1 Results:")
print(f"   Accuracy: {accuracy:.4f}")

# Convert predictions back to original labels for the classification report
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print(f"\n   Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# ============================================================
# STEP 6: Show per-class performance
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Per-class performance summary")
print("=" * 60)

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=[0, 1, 2, 3]
)

print("Per-class metrics:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"\n{class_name.upper()}:")
    print(f"   Precision: {precision[i]:.3f}")
    print(f"   Recall: {recall[i]:.3f}")
    print(f"   F1-score: {f1[i]:.3f}")
    print(f"   Support: {support[i]}")

# ============================================================
# STEP 7: Load QA pairs (optional)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Loading QA pairs")
print("=" * 60)

try:
    with open("qa_pairs.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"✅ Loaded QA data with keys: {list(qa_data.keys())}")
    if 'data' in qa_data:
        print(f"   Contains {len(qa_data['data'])} QA pairs")
except Exception as e:
    print(f"⚠️ Could not load QA pairs: {e}")

print("\n" + "=" * 60)
print("✅ Done! Ready for Models 2 and 3")
print("=" * 60)