import os
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1) Experiment Configuration
# ==========================================
N_PCA_COMPONENTS = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==========================================
# 2) Load Both Datasets
# ==========================================
main_path = os.path.join("..", "ds")

print("\nLoading Pavia Datasets...")
# Pavia University (SOURCE DOMAIN - Training)
data_u_path = os.path.join(main_path, 'pavia_university/pavia_university.mat')
gt_u_path = os.path.join(main_path, 'pavia_university/pavia_university_gt.mat')
data_U = sio.loadmat(data_u_path)['paviaU']
gt_U = sio.loadmat(gt_u_path)['paviaU_gt']

# Pavia Centre (TARGET DOMAIN - Testing)
data_c_path = os.path.join(main_path, 'pavia_center/pavia_center.mat')
gt_c_path = os.path.join(main_path, 'pavia_center/pavia_center_gt.mat')
data_C = sio.loadmat(data_c_path)['pavia']
gt_C = sio.loadmat(gt_c_path)['pavia_gt']

# ==========================================
# 3) Band Alignment (103 vs 102)
# ==========================================
print(f"Original bands -> Pavia U: {data_U.shape[2]}, Pavia C: {data_C.shape[2]}")
# Truncate Pavia U to match Pavia Centre's 102 bands
data_U = data_U[:, :, :102]
print(f"Aligned bands  -> Pavia U: {data_U.shape[2]}, Pavia C: {data_C.shape[2]}")

# Flatten data
X_U = data_U.reshape(-1, data_U.shape[2])
y_U_raw = gt_U.reshape(-1)

X_C = data_C.reshape(-1, data_C.shape[2])
y_C_raw = gt_C.reshape(-1)

# ==========================================
# 4) Class Mapping (Isolating Shared Classes)
# ==========================================
# We map the specific dataset labels to a unified 0-6 scale
unified_class_names = ['Asphalt', 'Meadows', 'Trees', 'Bare Soil', 'Bitumen', 'Bricks', 'Shadows']

# Format: {Original_Label: Unified_Label}
map_U = {1:0, 2:1, 4:2, 6:3, 7:4, 8:5, 9:6} # Pavia U mappings
map_C = {3:0, 8:1, 2:2, 9:3, 5:4, 4:5, 7:6} # Pavia Centre mappings

def filter_and_map(X, y_raw, mapping_dict):
    """Filters out unshared classes and remaps the rest to 0-6."""
    mask = np.isin(y_raw, list(mapping_dict.keys()))
    X_filtered = X[mask]
    y_filtered_raw = y_raw[mask]
    
    # Apply the mapping
    y_mapped = np.vectorize(mapping_dict.get)(y_filtered_raw)
    return X_filtered, y_mapped

print("\nIsolating shared classes...")
X_train, y_train = filter_and_map(X_U, y_U_raw, map_U) # Train strictly on University
X_test, y_test = filter_and_map(X_C, y_C_raw, map_C)   # Test strictly on Centre

print(f"Training samples (Pavia U): {X_train.shape[0]}")
print(f"Testing samples (Pavia C):  {X_test.shape[0]}")

# ==========================================
# 5) Pure ML Preprocessing (Fit on Train ONLY)
# ==========================================
print("\nScaling and applying PCA (Strictly fit on Pavia U)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

# Apply the exact same mathematical transformations to the unseen Target Domain
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Convert to Tensors
X_train_t = torch.tensor(X_train_pca, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test_pca, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# ==========================================
# 6) Define the 1D Neural Network
# ==========================================
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

# 7 Classes total
model = SimpleClassifier(input_size=N_PCA_COMPONENTS, num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 7) Train the Model (on Source Domain)
# ==========================================
epochs = 150
print("\nStarting Training on Pavia University...")
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
# 8) Evaluate Model (on Target Domain)
# ==========================================
print("\nEvaluating on unseen Pavia Centre...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test_t).sum().item()
    total = y_test_t.size(0)
    accuracy = (correct / total) * 100
    
    print(f"\nFinal Domain Adaptation Accuracy: {accuracy:.2f}%")

y_true = y_test_t.cpu().numpy()
y_pred = predicted.cpu().numpy()

# ==========================================
# 9) Save Performance
# ==========================================
output_dir = os.path.join("..", "files", "pavia_domain_adaptation", "MLP")
os.makedirs(output_dir, exist_ok=True)
report_str = classification_report(y_true, y_pred, target_names=unified_class_names, zero_division=0)

file_path = os.path.join(output_dir, "Pavia_DA_1D_MLP_Report.txt")
with open(file_path, "w") as f:
    f.write(f"DOMAIN ADAPTATION EXPERIMENT - 1D MLP\n")
    f.write(f"Source Training: Pavia University\n")
    f.write(f"Target Testing:  Pavia Centre\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write("="*60 + "\n\n")
    f.write(report_str)

print(f"\n[SUCCESS] Domain Adaptation report saved to: {file_path}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unified_class_names, yticklabels=unified_class_names)
plt.xlabel('Predicted Label (Pavia Centre)')
plt.ylabel('True Label (Pavia Centre)')
plt.title('Domain Adaptation: Train on University, Test on Centre')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()