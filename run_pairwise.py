import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.io as sio

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================
# Assuming 'data' and 'labels' are already loaded

main_path = "ds"

data_path = os.path.join(main_path, 'indiana_pines/indian_pines_corrected.mat')
labels_path = os.path.join(main_path, 'indiana_pines/indian_pines_gt.mat')

data = sio.loadmat(data_path)['indian_pines_corrected']
labels = sio.loadmat(labels_path)['indian_pines_gt']

print("Reshaping data...")
X = data.reshape(-1, data.shape[-1])
y = labels.reshape(-1)

# Ensure output directory exists
output_dir = "files/linear_models"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory established at: {output_dir}/")

# Filter out background (Class 0) to get our 16 real crop classes
valid_pixels = y != 0
X_crops = X[valid_pixels]
y_crops = y[valid_pixels]

# Get a list of the unique classes [1, 2, 3, ..., 16]
unique_classes = np.unique(y_crops)

# Generate all unique pairs (120 pairs for 16 classes)
crop_pairs = list(combinations(unique_classes, 2))
print(f"Generated {len(crop_pairs)} unique crop pairs for analysis.")

# Initialize a list to hold our results before saving to CSV
results_data = []

# ==========================================
# 2. EXECUTE PAIRWISE PIPELINE
# ==========================================
print("\nRunning Logistic Regression on all pairs...")

for idx, (class_a, class_b) in enumerate(crop_pairs, 1):
    # Print progress every 20 pairs so you know it hasn't frozen
    if idx % 20 == 0 or idx == 1:
        print(f"Processing pair {idx}/{len(crop_pairs)}: Class {class_a} vs Class {class_b}...")
        
    # Filter dataset to ONLY include these two classes
    mask = np.isin(y_crops, [class_a, class_b])
    X_pair = X_crops[mask]
    y_pair = y_crops[mask]
    
    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X_pair, y_pair, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression (Binary Classification)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy for this pair
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    # Extract absolute weights to find the most important bands
    # model.coef_ shape is (1, 200) for binary classification
    weights = np.abs(model.coef_[0])
    
    # Get the indices of the top 10 highest weights 
    # (Adding 1 because bands are usually 1-indexed, not 0-indexed)
    top_10_indices = np.argsort(weights)[::-1][:10] + 1
    top_10_bands = ", ".join(map(str, top_10_indices))
    
    # Append to our results list
    results_data.append({
        "Class_A": class_a,
        "Class_B": class_b,
        "Separation_Accuracy_%": round(acc * 100, 2),
        "Top_10_Important_Bands": top_10_bands
    })

# ==========================================
# 3. EXPORT TO CSV
# ==========================================
# Convert results to a pandas DataFrame and save
df_results = pd.DataFrame(results_data)
csv_path = os.path.join(output_dir, "pairwise_band_importance.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "="*50)
print(f"ANALYSIS COMPLETE. Results saved to:\n{csv_path}")
print("="*50)