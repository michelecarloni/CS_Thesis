"""
This script considers 4 scenarios:
- all classes with background
- all classes without background (only crops)
- kinds of crop (Corn-notill, Corn-mintill, Corn) class 2, 3, 4
- kinds of soybean (Soybean-notill, Soybean-mintill, Soybean) class 10, 11, 12

For all these scenarios 4 algorithms have been tried:
- Decison tree
- Random forest
- Logistic regression
- Linear SVM

The performance and plot data can be found respectively at the following paths:

files/linear_models/performance
files/linear_models/plot

"""



import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import  scipy.io as sio
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================
# Assuming 'data' and 'labels' are already loaded in your environment
# X = data.reshape(-1, data.shape[-1])
# y = labels.reshape(-1)

# Load Data

main_path = "ds"

data_path = os.path.join(main_path, 'indiana_pines/indian_pines_corrected.mat')
labels_path = os.path.join(main_path, 'indiana_pines/indian_pines_gt.mat')

data = sio.loadmat(data_path)['indian_pines_corrected']
labels = sio.loadmat(labels_path)['indian_pines_gt']


print("Reshaping data...")
X = data.reshape(-1, data.shape[-1])
y = labels.reshape(-1)

# Create the separated output directories
perf_dir = "files/linear_models/performance"
plot_dir = "files/linear_models/plot"
os.makedirs(perf_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
print(f"Directories established:\n - {perf_dir}/\n - {plot_dir}/")

# ==========================================
# 2. HELPER FUNCTION: PLOT WEIGHTS
# ==========================================
def plot_and_save_weights(model, model_name, scenario_name, output_folder):
    """Extracts weights from linear models and saves a bar chart."""
    mean_abs_weights = np.mean(np.abs(model.coef_), axis=0)
    bands = np.arange(1, len(mean_abs_weights) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.bar(bands, mean_abs_weights, color='steelblue')
    
    clean_scenario = scenario_name.replace('_', ' ') 
    plt.title(f'Overall Band Importance: {model_name}\n({clean_scenario})', fontsize=14)
    plt.xlabel('Hyperspectral Band Number', fontsize=12)
    plt.ylabel('Mean Absolute Coefficient', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 201)
    
    safe_model_name = model_name.replace(' ', '_')
    filename = f"{scenario_name}_{safe_model_name}_Bands.png"
    filepath = os.path.join(output_folder, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close() 

# ==========================================
# 3. DEFINE SCENARIOS & MODELS
# ==========================================
scenarios = {
    "1_All_Classes_With_Background": lambda labels: np.ones_like(labels, dtype=bool),
    "2_All_Crops_No_Background": lambda labels: labels != 0,
    "3_Corn_Tillage_Only": lambda labels: np.isin(labels, [2, 3, 4]),
    "4_Soybean_Tillage_Only": lambda labels: np.isin(labels, [10, 11, 12])
}

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Linear SVM": LinearSVC(max_iter=2000, dual=False, random_state=42)
}

# ==========================================
# 4. EXECUTE PIPELINE
# ==========================================
print("\nStarting automated experiment pipeline...")

for scenario_name, condition in scenarios.items():
    print(f"\n>>> Running Scenario: {scenario_name}")
    
    mask = condition(y)
    X_subset = X[mask]
    y_subset = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the text report to the 'performance' directory
    file_path = os.path.join(perf_dir, f"{scenario_name}_results.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"EXPERIMENT SCENARIO: {scenario_name}\n")
        f.write(f"Total Samples: {X_subset.shape[0]}\n")
        f.write("="*50 + "\n\n")
        
        for model_name, model in models.items():
            print(f"    -> Training {model_name}...")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            
            f.write(f"--- {model_name.upper()} ---\n")
            f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n" + "-"*50 + "\n\n")
            
            # Save the PNG plot to the 'plot' directory
            if model_name in ["Logistic Regression", "Linear SVM"]:
                plot_and_save_weights(model, model_name, scenario_name, plot_dir)
                
    print(f"    [SUCCESS] Results and plots saved for {scenario_name}")

print("\n" + "="*50)
print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")
print("="*50)