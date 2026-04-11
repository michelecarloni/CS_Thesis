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
WINDOW_SIZE = 7
N_PCA_COMPONENTS = 30
BUFFER_ZONE = 4 # Creates an 8-pixel dead zone (4 on each side of the median line)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==========================================
# 2) Load Data
# ==========================================
main_path = os.path.join("..", "ds")

data_path = os.path.join(main_path, 'salinas_valley/salinas_valley_corrected.mat')
gt_path = os.path.join(main_path, 'salinas_valley/salinas_valley_gt.mat')

data = sio.loadmat(data_path)['salinas_corrected']
gt = sio.loadmat(gt_path)['salinas_gt']

h, w, c = data.shape

# ==========================================
# 3) Create Disjoint Spatial Masks (No Overlap)
# ==========================================
print("Creating physically disjoint spatial splits...")
train_mask = np.zeros((h, w), dtype=bool)
test_mask = np.zeros((h, w), dtype=bool)

for class_idx in range(1, 17): # Classes 1 through 16
    rows, cols = np.where(gt == class_idx)
    if len(rows) == 0: continue
    
    # Check which physical axis of the field is longer to make a clean cut
    row_spread = rows.max() - rows.min()
    col_spread = cols.max() - cols.min()
    
    if row_spread > col_spread:
        # Field is tall: Cut horizontally
        median_line = np.median(rows)
        for r, col in zip(rows, cols):
            if r < median_line - BUFFER_ZONE:
                train_mask[r, col] = True
            elif r > median_line + BUFFER_ZONE:
                test_mask[r, col] = True
    else:
        # Field is wide: Cut vertically
        median_line = np.median(cols)
        for r, col in zip(rows, cols):
            if col < median_line - BUFFER_ZONE:
                train_mask[r, col] = True
            elif col > median_line + BUFFER_ZONE:
                test_mask[r, col] = True

print(f"Training pixels assigned: {np.sum(train_mask)}")
print(f"Testing pixels assigned:  {np.sum(test_mask)}")
print(f"Dead zone (discarded):    {np.sum(gt != 0) - np.sum(train_mask) - np.sum(test_mask)}")

# ==========================================
# 4) Pure ML Preprocessing (Fit on Train ONLY)
# ==========================================
print("\nScaling and applying PCA (Strictly fit on training data)...")
data_2d = data.reshape(-1, c)
train_mask_flat = train_mask.flatten()

# Fit scaler and PCA ONLY on the training pixels
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(data_2d[train_mask_flat])

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
pca.fit(train_data_scaled)

# Now apply the completely trained transformations to the entire map
data_scaled_full = scaler.transform(data_2d)
data_pca_full = pca.transform(data_scaled_full)

# Reshape back to 3D map
data_3d = data_pca_full.reshape(h, w, N_PCA_COMPONENTS)

# ==========================================
# 5) Extract Patches safely using Masks
# ==========================================
print(f"Extracting {WINDOW_SIZE}x{WINDOW_SIZE} spatial patches...")
margin = int((WINDOW_SIZE - 1) / 2)
padded_X = np.pad(data_3d, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

def extract_from_mask(mask):
    patches, labels = [], []
    for r, c_idx in zip(*np.where(mask)):
        # Original coordinates r, c map to r:r+WINDOW_SIZE in the padded array
        patch = padded_X[r : r + WINDOW_SIZE, c_idx : c_idx + WINDOW_SIZE]
        patches.append(patch)
        labels.append(gt[r, c_idx] - 1) # Remap to 0-15
    return np.array(patches), np.array(labels)

X_train, y_train = extract_from_mask(train_mask)
X_test, y_test = extract_from_mask(test_mask)

# Reshape for PyTorch (Batch, Channels, Height, Width)
X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)

# Convert to Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# ==========================================
# 6) Define the 2D CNN Architecture
# ==========================================
class SalinasCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SalinasCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.reshape(x.size(0), -1) # Flattened safely
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SalinasCNN(in_channels=N_PCA_COMPONENTS, num_classes=16).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 7) Train the Model
# ==========================================
epochs = 150
print("\nStarting Disjoint CNN Training...")
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
# 8) Evaluate & Save
# ==========================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test_t).sum().item()
    total = y_test_t.size(0)
    accuracy = (correct / total) * 100
    
    print(f"\nFinal Disjoint CNN Test Accuracy: {accuracy:.2f}%")

y_true = y_test_t.cpu().numpy()
y_pred = predicted.cpu().numpy()

class_names = [
    'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
    'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 
    'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 
    'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 
    'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'
]

output_dir = os.path.join("..", "files", "salinas_valley", "CNN")
os.makedirs(output_dir, exist_ok=True)
report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

file_path = os.path.join(output_dir, "Salinas_2D_CNN_DISJOINT_Report.txt")
with open(file_path, "w") as f:
    f.write(f"Salinas Valley - 2D CNN Classification (DISJOINT SPATIAL SPLIT)\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write("="*60 + "\n\n")
    f.write(report_str)

print(f"\n[SUCCESS] Disjoint report saved to: {file_path}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Salinas Valley (Disjoint Spatial Split)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()