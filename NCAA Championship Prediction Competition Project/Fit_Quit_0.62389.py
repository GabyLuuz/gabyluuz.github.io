import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import shap
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
# pip install lightgbm
# pip install shap
# pip install sentence-transformers

warnings.simplefilter(action='ignore', category=UserWarning)

# Set working directory (Modify as needed)
os.chdir(r'C:\Users\22760\OneDrive - purdue.edu\02 Courses\IP\CCAC\DATA')

# Load Data
train_df = pd.read_csv("bracket_training.csv")
test_df = pd.read_csv("bracket_test.csv")

# Encode categorical features
label_encoders = {}

# Convert categorical columns to numeric using Label Encoding
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

# **Feature Engineering with LLM**
# Use Sentence Transformers to create embeddings for categorical features
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Choose a categorical text column (modify based on your dataset)
text_col = "CustomerDMADescription"   # Example column
# Convert text column to a list (to process all at once)
train_texts = train_df["CustomerDMADescription"].astype(str).tolist()
test_texts = test_df["CustomerDMADescription"].astype(str).tolist()

# Encode all rows in **batches** instead of one-by-one
train_embeddings = embed_model.encode(train_texts, batch_size=32, show_progress_bar=True)
test_embeddings = embed_model.encode(test_texts, batch_size=32, show_progress_bar=True)

# Convert embeddings to DataFrame and merge with original data
train_embeddings_df = pd.DataFrame(train_embeddings, columns=[f"text_feat_{i}" for i in range(train_embeddings.shape[1])])
test_embeddings_df = pd.DataFrame(test_embeddings, columns=[f"text_feat_{i}" for i in range(test_embeddings.shape[1])])

# Merge embeddings back to train and test datasets
train_df = pd.concat([train_df.reset_index(drop=True), train_embeddings_df.reset_index(drop=True)], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_embeddings_df.reset_index(drop=True)], axis=1)

# Convert all values to string format to avoid 'int' error
train_df["CustomerDMADescription"] = train_df["CustomerDMADescription"].astype(str)
test_df["CustomerDMADescription"] = test_df["CustomerDMADescription"].astype(str)

# Encode with batch processing
train_embeddings = embed_model.encode(
    train_df["CustomerDMADescription"].tolist(),
    batch_size=32,
    show_progress_bar=True
)
# Convert test data text column into embeddings using batch processing
test_embeddings = embed_model.encode(
    test_df["CustomerDMADescription"].tolist(),  # Convert to list
    batch_size=32,  # Process multiple rows at once (Adjust for speed)
    show_progress_bar=True  # Show progress
)

# Convert list of embeddings into individual feature columns
test_embeddings_pca = np.array(test_embeddings)  # Convert to NumPy array

# Add embeddings as new feature columns in test_df
for i in range(test_embeddings_pca.shape[1]):  # Loop over embedding dimensions
    test_df[f"text_feat_{i}"] = test_embeddings_pca[:, i]

# Drop the original text column (optional)
test_df.drop(columns=["CustomerDMADescription"], inplace=True)

# Store predictions
test_predictions = test_df.copy()

# Update features list AFTER LLM embeddings are added
features = [col for col in train_df.columns if col not in ["NationalChampion", "SemifinalWinner_East_West", "SemifinalWinner_South_Midwest", "BracketEntryId", "CustomerDMADescription"]]

# Train a model for each target variable
targets = ["SemifinalWinner_East_West", "SemifinalWinner_South_Midwest", "NationalChampion"]
label_encoders_targets = {}

from sklearn.model_selection import RandomizedSearchCV
from collections import Counter

for target in targets:
    print(f"\n🚀 Training model for: {target}")

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(train_df[target])
    label_encoders_targets[target] = le

    # Split data (use a smaller training size for speed)
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[features], y_encoded, test_size=0.1, random_state=42, stratify=y_encoded  # Use 10% validation set
    )

    # **Optimized Hyperparameter Tuning**
    param_grid = {
       'n_estimators': [100, 200, 300],  
       'max_depth': [3, 5, 7],  
       'learning_rate': [0.03, 0.05, 0.1] 
       }

    xgb_model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric="mlogloss", 
        tree_method="hist",  # Use GPU if available
        n_jobs=-1  # Use all CPU cores
    )

    # **Use RandomizedSearchCV Instead of GridSearchCV**
    grid_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=5,  # Only search 5 combinations instead of full grid
        cv=2,  # Reduce cross-validation folds (from 3 to 2)
        scoring="accuracy", 
        verbose=1, 
        n_jobs=-1  # Parallel processing
    )
    grid_search.fit(X_train, y_train)
    best_xgb_model = grid_search.best_estimator_

    # Evaluate both models
    y_val_pred_xgb = best_xgb_model.predict(X_val)
    y_val_pred_lgb = lgb_model.predict(X_val)

    acc_xgb = accuracy_score(y_val, y_val_pred_xgb)
    acc_lgb = accuracy_score(y_val, y_val_pred_lgb)

    print(f"✅ XGBoost Accuracy for {target}: {acc_xgb:.4f}")
    print(f"✅ LightGBM Accuracy for {target}: {acc_lgb:.4f}")

    # Choose the best model and print which one was selected
    if acc_xgb > acc_lgb:
        best_model = best_xgb_model
        print(f"🔥 Best Model for {target}: **XGBoost** with params: {grid_search.best_params_}")
    else:
        best_model = lgb_model
        print(f"🔥 Best Model for {target}: **LightGBM** (default params)")

    # Make predictions on the test set
    y_test_pred = best_model.predict(test_df[features])

    # Decode predictions back to original labels
    test_predictions[target] = le.inverse_transform(y_test_pred)

# Save final predictions to CSV
save_path = r"C:\Users\22760\OneDrive - purdue.edu\02 Courses\IP\CCAC\DATA\submission.csv"
submission = test_predictions[["BracketEntryId", "SemifinalWinner_East_West", "SemifinalWinner_South_Midwest", "NationalChampion"]]
submission.to_csv(save_path, index=False)
print(f"✅ Submission file saved at: {save_path}")

'''
🚀 Training model for: SemifinalWinner_East_West
Fitting 2 folds for each of 5 candidates, totalling 10 fits
✅ XGBoost Accuracy for SemifinalWinner_East_West: 0.6854
✅ LightGBM Accuracy for SemifinalWinner_East_West: 0.0008
🔥 Best Model for SemifinalWinner_East_West: **XGBoost** with params: {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.03}

🚀 Training model for: SemifinalWinner_South_Midwest
Fitting 2 folds for each of 5 candidates, totalling 10 fits
✅ XGBoost Accuracy for SemifinalWinner_South_Midwest: 0.6406
✅ LightGBM Accuracy for SemifinalWinner_South_Midwest: 0.0022
🔥 Best Model for SemifinalWinner_South_Midwest: **XGBoost** with params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}

🚀 Training model for: NationalChampion
Fitting 2 folds for each of 5 candidates, totalling 10 fits
✅ XGBoost Accuracy for NationalChampion: 0.4683
✅ LightGBM Accuracy for NationalChampion: 0.1239
🔥 Best Model for NationalChampion: **XGBoost** with params: {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03}
'''