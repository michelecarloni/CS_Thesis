import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1) Experiment Configuration
# ==========================================
USE_PCA = True                # Toggled ON for the baseline as requested
N_PCA_COMPONENTS = 30         # 30 is the standard academic baseline for Salinas

USE_FEATURE_SELECTION = False 
N_TOP_BANDS = 60              

# ==========================================
# 2) Setup GPU
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
print(f"PCA Enabled: {USE_PCA}")
print(f"Feature reduction Enabled: {USE_FEATURE_SELECTION}")

# ==========================================
# 3) Load and Filter the Data
# ==========================================
# Adjust these folder names if your dataset is saved elsewhere
main_path = os.path.join("..", "ds")

data_path = os.path.join(main_path, 'salinas_valley/salinas_valley_corrected.mat')
gt_path = os.path.join(main_path, 'salinas_valley/salinas_valley_gt.mat')

data = sio.loadmat(data_path)['salinas_corrected']
gt = sio.loadmat(gt_path)['salinas_gt']

# Flatten the data
h, w, c = data.shape
data_2d = data.reshape(-1, c)
gt_1d = gt.reshape(-1)

# Filter out the background (Class 0) to keep all 16 crop classes
print("Filtering out background pixels...")
valid_mask = (gt_1d != 0)
X_valid = data_2d[valid_mask]
y_valid_raw = gt_1d[valid_mask]

# CRITICAL PYTORCH STEP: Remap labels from 1-16 to 0-15
y_valid = y_valid_raw - 1

print(f"Total valid pixels for training: {X_valid.shape[0]}")

# ==========================================
# 4) Split Data (70/30 is more standard for full datasets)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.3, random_state=42)

# ==========================================
# 5) Preprocessing: Scaling and Optional PCA
# ==========================================
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

if USE_PCA:
    print(f"Applying PCA... Reducing from 204 to {N_PCA_COMPONENTS} components.")
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_train_processed = pca.fit_transform(X_train_processed)
    X_test_processed = pca.transform(X_test_processed)
    network_input_size = N_PCA_COMPONENTS
    
elif USE_FEATURE_SELECTION:
    print(f"\nRunning Random Forest to find the top {N_TOP_BANDS} physical bands...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_processed, y_train)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-N_TOP_BANDS:]
    top_indices = np.sort(top_indices)
    X_train_processed = X_train_processed[:, top_indices]
    X_test_processed = X_test_processed[:, top_indices]
    network_input_size = N_TOP_BANDS
else:
    print("Using all original bands.")
    network_input_size = 204 # Salinas usually has 204 bands after water absorption removal

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# ==========================================
# 6) Define the Neural Network
# ==========================================
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # Updated to handle 16 total classes
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model for 16 classes
model = SimpleClassifier(input_size=network_input_size, num_classes=16).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 7) Train the Model
# ==========================================
epochs = 150 # Bumped up slightly for the larger dataset
print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 30 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ==========================================
# 8) Evaluate the Model
# ==========================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test_t).sum().item()
    total = y_test_t.size(0)
    accuracy = (correct / total) * 100
    
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

y_true = y_test_t.cpu().numpy()
y_pred = predicted.cpu().numpy()

# ==========================================
# 9) Save Classification Report
# ==========================================
# Official 16 class names for Salinas
class_names = [
    'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
    'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 
    'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 
    'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 
    'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'
]

# Create output directory
output_dir = os.path.join("..", "files", "salinas_valley", "MLP")
os.makedirs(output_dir, exist_ok=True)

report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print("\n--- Classification Report ---")
print(report_str)

# Save to file
file_path = os.path.join(output_dir, "Salinas_1D_PCA_Baseline_Report.txt")
with open(file_path, "w") as f:
    f.write(f"Salinas Valley - 1D Pixel-Based Classification (No Locality)\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write("="*60 + "\n\n")
    f.write(report_str)

print(f"\n[SUCCESS] Classification report saved to: {file_path}")

# ==========================================
# 10) Plot Confusion Matrix
# ==========================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Salinas Valley (1D Baseline)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()