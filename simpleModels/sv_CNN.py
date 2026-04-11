import os
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1) Experiment Configuration
# ==========================================
WINDOW_SIZE = 7               # Creates a 7x7 spatial patch around each pixel
N_PCA_COMPONENTS = 30         # Compress chemistry to 30 bands
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==========================================
# 2) Helper Function: Patch Extraction
# ==========================================
def create_patches(X, y, window_size=7):
    """Pads the image and extracts a window_size x window_size patch per pixel."""
    margin = int((window_size - 1) / 2)
    # Pad the spatial dimensions of the image with zeros
    zero_padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    
    patches = []
    labels = []
    
    print(f"Extracting {window_size}x{window_size} spatial patches...")
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            label = y[r - margin, c - margin]
            if label != 0: # Only extract patches for valid crop pixels
                patch = zero_padded_X[r - margin : r + margin + 1, c - margin : c + margin + 1]
                patches.append(patch)
                labels.append(label)
                
    return np.array(patches), np.array(labels)

# ==========================================
# 3) Load Data & Preprocess (Scale + PCA)
# ==========================================
main_path = os.path.join("..", "ds")

data_path = os.path.join(main_path, 'salinas_valley/salinas_valley_corrected.mat')
gt_path = os.path.join(main_path, 'salinas_valley/salinas_valley_gt.mat')

data = sio.loadmat(data_path)['salinas_corrected']
gt = sio.loadmat(gt_path)['salinas_gt']

h, w, c = data.shape

# Step A: Flatten to 2D to apply Scaling and PCA
data_2d = data.reshape(-1, c)

print("Scaling data and applying PCA...")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_2d)

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
data_pca = pca.fit_transform(data_scaled)

# Step B: Reshape back to 3D image cube (Height x Width x PCA_Channels)
data_3d = data_pca.reshape(h, w, N_PCA_COMPONENTS)

# ==========================================
# 4) Extract Patches & Split Data
# ==========================================
# Extract the patches
X_patches, y_labels_raw = create_patches(data_3d, gt, window_size=WINDOW_SIZE)

# CRITICAL PYTORCH STEP: Remap labels from 1-16 to 0-15
y_labels = y_labels_raw - 1

print(f"Total valid patches extracted: {X_patches.shape[0]}")

# Split 70/30 (Random Split)
X_train, X_test, y_train, y_test = train_test_split(X_patches, y_labels, test_size=0.3, random_state=42)

# PyTorch expects dimensions: (Batch, Channels, Height, Width)
# Current shape: (Batch, Height, Width, Channels) -> We must transpose
X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)

# Convert to Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# ==========================================
# 5) Define the 2D CNN Architecture
# ==========================================
class SalinasCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SalinasCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # Reduces 7x7 to 3x3
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # No pooling here to preserve the 3x3 spatial grid
        
        # Fully Connected Block
        # 128 channels * 3 height * 3 width = 1152 flattened features
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4) # Helps prevent overfitting on local patches
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SalinasCNN(in_channels=N_PCA_COMPONENTS, num_classes=16).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 6) Train the Model
# ==========================================
epochs = 150
print("\nStarting CNN Training...")
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
# 7) Evaluate & Save
# ==========================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test_t).sum().item()
    total = y_test_t.size(0)
    accuracy = (correct / total) * 100
    
    print(f"\nFinal CNN Test Accuracy: {accuracy:.2f}%")

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

file_path = os.path.join(output_dir, "Salinas_2D_CNN_Report.txt")
with open(file_path, "w") as f:
    f.write(f"Salinas Valley - 2D CNN Classification (Locality via 7x7 Patches)\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write("="*60 + "\n\n")
    f.write(report_str)

print(f"\n[SUCCESS] CNN report saved to: {file_path}")

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Salinas Valley (2D CNN Spatial Context)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()