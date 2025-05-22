import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
# pip install xgboost

os.chdir(r'C:\Users\22760\OneDrive - purdue.edu\02 Courses\IP\CCAC\DATA')
# 1️⃣ Load Data
train_df = pd.read_csv("bracket_training.csv")
test_df = pd.read_csv("bracket_test.csv")

# 2️⃣ Exploratory Data Analysis (EDA)
print("Training Data Info:")
print(train_df.info())  # Check data types and missing values
print("\nTest Data Info:")
print(test_df.info())

# Check for missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum())

# Encoding categorical features
label_encoders = {}

# Convert categorical columns to numeric using Label Encoding
for col in train_df.columns:
    if train_df[col].dtype == "object":  # Check if column is categorical
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])  # Fit on train data
        label_encoders[col] = le  # Save encoder for test set transformation


# Apply encoding with an extended LabelEncoder that includes unseen labels
for col in test_df.columns:
    if col in label_encoders:
        le = label_encoders[col]  # Get the trained encoder
        
        # Extend encoder classes to include unseen labels
        all_classes = np.append(le.classes_, ["UNKNOWN"])  # Add a placeholder for new labels
        le.classes_ = all_classes  # Update the encoder with new classes
        
        # Transform test data (unseen labels become "UNKNOWN")
        test_df[col] = test_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["UNKNOWN"])[0])

# 4️⃣ Define Features and Target Variable
# Define features
features = [col for col in train_df.columns if col not in ["NationalChampion", "SemifinalWinner_East_West", "SemifinalWinner_South_Midwest", "BracketEntryId"]]

# Store predictions
test_predictions = test_df.copy()

# Train a model for each target variable
targets = ["SemifinalWinner_East_West", "SemifinalWinner_South_Midwest", "NationalChampion"]
label_encoders_targets = {}  # Store LabelEncoders for decoding predictions

for target in targets:
    print(f"\nTraining model for: {target}")

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(train_df[target])  # Convert non-sequential labels to 0, 1, 2...
    label_encoders_targets[target] = le  # Store for later decoding

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[features], y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy for {target}: {accuracy:.4f}")

    # Make Predictions for test set
    y_test_pred = model.predict(test_df[features])

    # Decode predictions back to original labels
    test_predictions[target] = le.inverse_transform(y_test_pred)

# Save final predictions to a CSV
# Define the save path
save_path = r"C:\Users\22760\OneDrive - purdue.edu\02 Courses\IP\CCAC\DATA\submission.csv"

# Save final predictions to CSV
submission.to_csv(save_path, index=False)
print(f"✅ Submission file saved at: {save_path}")
