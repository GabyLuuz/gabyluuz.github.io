import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import shap
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

warnings.simplefilter(action='ignore', category=UserWarning)

# Set working directory (Modify as needed)
os.chdir(r'C:\Users\22760\OneDrive - purdue.edu\02 Courses\IP\CCAC\DATA')

# Load Data
train_df = pd.read_csv("bracket_training.csv")
test_df = pd.read_csv("bracket_test.csv")

# Encode categorical features
label_encoders = {}
for col in train_df.columns:
    if train_df[col].dtype == "object":
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        label_encoders[col] = le
        
# Apply the same transformation to test data
for col in test_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        all_classes = np.append(le.classes_, ["UNKNOWN"])
        le.classes_ = all_classes
        test_df[col] = test_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["UNKNOWN"])[0])

# **Enhanced Feature Engineering with LLM**
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Choose a categorical text column
text_col = "CustomerDMADescription"

# Encode text features efficiently
train_embeddings = embed_model.encode(train_df[text_col].astype(str).tolist(), batch_size=64, show_progress_bar=True)
test_embeddings = embed_model.encode(test_df[text_col].astype(str).tolist(), batch_size=64, show_progress_bar=True)

# Convert embeddings to DataFrame and merge with original data
embedding_cols = [f"text_feat_{i}" for i in range(train_embeddings.shape[1])]
train_embeddings_df = pd.DataFrame(train_embeddings, columns=embedding_cols)
test_embeddings_df = pd.DataFrame(test_embeddings, columns=embedding_cols)

train_df = pd.concat([train_df.reset_index(drop=True), train_embeddings_df.reset_index(drop=True)], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_embeddings_df.reset_index(drop=True)], axis=1)

# Drop the original text column
train_df.drop(columns=[text_col], inplace=True)
test_df.drop(columns=[text_col], inplace=True)

# Update feature list
features = [col for col in train_df.columns if col not in ["NationalChampion", "SemifinalWinner_East_West", "SemifinalWinner_South_Midwest", "BracketEntryId"]]

# Train separate models with best hyperparameters
best_params = {
    "SemifinalWinner_East_West": {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.03},
    "SemifinalWinner_South_Midwest": {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
    "NationalChampion": {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03},
}

test_predictions = test_df.copy()
label_encoders_targets = {}

for target in best_params:
    print(f"\n🚀 Training model for: {target}")

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(train_df[target])
    label_encoders_targets[target] = le

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[features], y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )

    # Train XGBoost model with best hyperparameters
    xgb_model = xgb.XGBClassifier(
        **best_params[target],
        random_state=42, 
        eval_metric="mlogloss", 
        tree_method="hist", 
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate
    y_val_pred = xgb_model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"✅ XGBoost Accuracy for {target}: {acc:.4f}")

    # Make predictions on test set
    y_test_pred = xgb_model.predict(test_df[features])
    test_predictions[target] = le.inverse_transform(y_test_pred)

# Save final predictions to CSV
submission = test_predictions[["BracketEntryId", "SemifinalWinner_East_West", "SemifinalWinner_South_Midwest", "NationalChampion"]]
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved.")
