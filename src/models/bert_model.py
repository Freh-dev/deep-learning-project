import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, BertConfig
import torch
from torch.utils.data import Dataset
import os

# Suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("=" * 60)
print("🤖 Model 2: BERT Fine-Tuning")
print("=" * 60)

# ============================================================
# STEP 1: Load and prepare data
# ============================================================
print("\n📂 Loading scenarios.json...")
with open("../scenarios.json", "r", encoding="utf-8") as f:
    data = json.load(f)

scenarios = data['data']
df = pd.DataFrame(scenarios)

# Encode labels
label_encoder = LabelEncoder()
df['risk_label'] = label_encoder.fit_transform(df['risk_level'])
print(f"✅ Labels: {list(label_encoder.classes_)}")
print(f"✅ Encoded: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Create text features
def create_text(row):
    fields = ['intended_use', 'system_type', 'input_data', 'domain']
    return " ".join([str(row[f]) for f in fields if pd.notna(row[f])])

df['text'] = df.apply(create_text, axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].values, df['risk_label'].values, 
    test_size=0.2, random_state=42, stratify=df['risk_label'].values
)

print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")

# ============================================================
# STEP 2: Create PyTorch Dataset
# ============================================================
class AIDataset(Dataset):
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
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# STEP 3: Load BERT model
# ============================================================
print("\n🔄 Loading BERT model...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with proper configuration
config = BertConfig.from_pretrained(
    model_name,
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    config=config
)

# Create datasets
train_dataset = AIDataset(X_train, y_train, tokenizer)
test_dataset = AIDataset(X_test, y_test, tokenizer)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# STEP 4: Setup training (FIXED - removed report_to)
# ============================================================
print("\n🔄 Setting up training...")
training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # report_to removed - uses default (tensorboard)
)

# ============================================================
# STEP 5: Custom metrics function
# ============================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# ============================================================
# STEP 6: Train the model
# ============================================================
print("\n🔄 Training BERT (this may take 5-10 minutes)...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ============================================================
# STEP 7: Evaluate
# ============================================================
print("\n📊 Evaluating BERT...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print(f"\n📊 Model 2 Results:")
print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Per-class metrics
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=[0, 1, 2, 3]
)

print("\n📊 Per-class performance summary:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"\n{class_name.upper()}:")
    print(f"   Precision: {precision[i]:.3f}")
    print(f"   Recall: {recall[i]:.3f}")
    print(f"   F1-score: {f1[i]:.3f}")
    print(f"   Support: {support[i]}")

print("\n✅ Model 2 Complete!")
print("=" * 60)