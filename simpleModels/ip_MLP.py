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

# 1) Experiment Configuration
# Toggle this to True or False to experiment!
USE_PCA = False
N_PCA_COMPONENTS = 30 

USE_FEATURE_SELECTION = True  # Toggle this to try keeping only the best physical bands
N_TOP_BANDS = 60              # How many physical bands to keep

# 2) Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
print(f"PCA Enabled: {USE_PCA}")
print(f"Feature reduction Enabled: {USE_FEATURE_SELECTION}")

# 3) Load and Filter the Data
main_path = os.path.join("..", "ds")

data_path = os.path.join(main_path, 'indiana_pines/indian_pines_corrected.mat')
gt_path = os.path.join(main_path, 'indiana_pines/indian_pines_gt.mat')

data = sio.loadmat(data_path)['indian_pines_corrected']
gt = sio.loadmat(gt_path)['indian_pines_gt']

# Flatten the data
h, w, c = data.shape
data_2d = data.reshape(-1, c)
gt_1d = gt.reshape(-1)

# Filter for ONLY Soybeans (Classes 10, 11, 12)
soybean_mask = (gt_1d == 10) | (gt_1d == 11) | (gt_1d == 12)
X_soy = data_2d[soybean_mask]
y_soy_raw = gt_1d[soybean_mask]

# Remap labels to 0, 1, 2 for PyTorch
y_soy = np.zeros_like(y_soy_raw)
y_soy[y_soy_raw == 10] = 0
y_soy[y_soy_raw == 11] = 1
y_soy[y_soy_raw == 12] = 2

# 4) Split Data FIRST (To prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(X_soy, y_soy, test_size=0.2, random_state=42)

# 5) Preprocessing: Scaling and Optional PCA
# Step A: Standardize the data (Fit on train, apply to train and test)
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

# Step B: Apply PCA or FEATURE SELECTION if toggled on
if USE_PCA:
    print(f"Applying PCA... Reducing from 200 to {N_PCA_COMPONENTS} components.")
    pca = PCA(n_components=N_PCA_COMPONENTS)
    # Fit PCA ONLY on training data to learn the transformation
    X_train_processed = pca.fit_transform(X_train_processed)
    # Apply that learned transformation to the test data
    X_test_processed = pca.transform(X_test_processed)
    
    # The neural network input layer must match the number of PCA components
    network_input_size = N_PCA_COMPONENTS
if USE_FEATURE_SELECTION:
    print(f"\nRunning Random Forest to find the top {N_TOP_BANDS} physical bands...")
    # Train the "Scout" model on the training data
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_processed, y_train)
    
    # Get the scorecard and find the indices of the highest-scoring bands
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-N_TOP_BANDS:]
    
    # Sort them so they remain in physical wavelength order (left to right)
    top_indices = np.sort(top_indices)
    print(f"Selected Physical Band Indices:\n{top_indices}")
    
    # SLICE the data to keep ONLY those physical columns
    X_train_processed = X_train_processed[:, top_indices]
    X_test_processed = X_test_processed[:, top_indices]
    
    network_input_size = N_TOP_BANDS
else:
    print("Skipping PCA and Feature selection... Using all 200 original bands.")
    # The neural network input layer must match the original 200 bands
    network_input_size = 200

# Convert to PyTorch Tensors and move to GPU
X_train_t = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# 6) Define the Neural Network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model with DYNAMIC input size
model = SimpleClassifier(input_size=network_input_size, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7) Train the Model
epochs = 100
print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 8) Evaluate the Model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test_t).sum().item()
    total = y_test_t.size(0)
    accuracy = (correct / total) * 100
    
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")


# PyTorch tensors are on the GPU. Scikit-learn requires NumPy arrays on the CPU.
y_true = y_test_t.cpu().numpy()
y_pred = predicted.cpu().numpy()

# Define our class names for the labels
class_names = ['Soybean-notill', 'Soybean-mintill', 'Soybean-clean']

# 9) Print Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# 10) Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
# annot=True puts the numbers inside the boxes, fmt='d' ensures they are whole numbers
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Label (What the model guessed)')
plt.ylabel('True Label (The actual ground truth)')
plt.title('Confusion Matrix: Soybean Tillage Classification')
plt.show()