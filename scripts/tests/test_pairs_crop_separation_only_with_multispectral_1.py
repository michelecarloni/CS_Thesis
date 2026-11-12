import os
import sys
import glob
import pandas as pd
import numpy as np
import itertools

# Setup project root path
project_root = os.path.abspath('../../')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.data_structures import h2crop_taxonomy_dict, h2crop_band_mapping_no_water_absorption_bands

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
splits = [1, 2, 3]
taxonomy = 3

ml_results_base = "/run/media/michele/T7/feature_importance_results/feature_importance_unique_subset_standard_ML_results_bar_chart/hyperspectral"
unet_results_base = "/run/media/michele/T7/feature_importance_results/feature_importance_unique_subset_unet_results_bar_chart/hyperspectral"

out_dir = "../../final_analysis_results/separation_multispectral_pairs"
os.makedirs(out_dir, exist_ok=True)

models = {
    "decision_tree": {"base": ml_results_base, "type": "ml"},
    "random_forest": {"base": ml_results_base, "type": "ml"},
    "logistic_regression": {"base": ml_results_base, "type": "ml"},
    "linear_svm": {"base": ml_results_base, "type": "ml"},
    "unet_resnet18": {"base": unet_results_base, "type": "unet"}
}

# 10 classes in total, EXCLUDING background (0) leaves 9 classes.
subset_classes = [1, 2, 8, 10, 11, 23, 29, 30, 38, 50]
subset_classes.sort()

taxonomy_key = f'Taxonomy_{taxonomy}'
class_names = {c: h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") for c in subset_classes}

# ==========================================
# SPECTRAL MACRO-GROUP MAPPING
# ==========================================
def get_macro_group(wavelength_nm):
    """Maps a specific wavelength to its macroscopic physical group."""
    if wavelength_nm < 520:
        return "Blue"
    elif wavelength_nm < 600:
        return "Green"
    elif wavelength_nm < 680:
        return "Red"
    elif wavelength_nm < 720:
        return "Lower Red Edge"
    elif wavelength_nm < 760:
        return "Upper Red Edge"
    elif wavelength_nm < 950:
        return "NIR-1"
    elif wavelength_nm < 1400:
        return "NIR-2"
    elif wavelength_nm < 1800:
        return "SWIR-1"
    else:
        return "SWIR-2"


if __name__ == "__main__":
    print(f"{'='*70}\nSTARTING THEORETICAL MULTISPECTRAL SEPARABILITY ANALYSIS\n{'='*70}")
    
    for model_name, info in models.items():
        base_path = info["base"]
        model_type = info["type"]
        
        for split in splits:
            print(f"  -> Analyzing {model_name.upper()} | Split {split}...")
            
            # Locate the model's directory for this split
            target_dir = os.path.join(base_path, f"split_{split}", model_name)
            if not os.path.exists(target_dir):
                print(f"     [Warning] Directory not found: {target_dir}")
                continue
            
            crop_top_features = {}
            
            # 1. Read the Top 2 Bands for each crop class
            for cls in subset_classes:
                csv_path = os.path.join(target_dir, f"importance_class_{cls}_sorted.csv")
                
                if not os.path.exists(csv_path):
                    print(f"     [Warning] Missing CSV for class {cls} in {model_name} split {split}")
                    continue
                    
                df = pd.read_csv(csv_path)
                top_2_bands = df['band'].head(2).tolist()
                
                band_details = []
                macro_groups = set()
                
                for b in top_2_bands:
                    wl = h2crop_band_mapping_no_water_absorption_bands.get(b, b)
                    if isinstance(wl, (int, float)):
                        group = get_macro_group(wl)
                        band_details.append({"index": b, "wavelength": f"{wl:.1f}nm", "group": group})
                        macro_groups.add(group)
                    else:
                        band_details.append({"index": b, "wavelength": "Unknown", "group": "Unknown"})
                        macro_groups.add("Unknown")
                        
                crop_top_features[cls] = {
                    "name": class_names[cls],
                    "details": band_details,
                    "groups": macro_groups
                }
            
            if len(crop_top_features) < 2:
                continue
                
            # 2. Analyze Pairwise Separability
            separable_pairs = []
            hyperspectral_pairs = []
            
            # Generate all unique crop-to-crop combinations (36 pairs for 9 classes)
            for cls_a, cls_b in itertools.combinations(crop_top_features.keys(), 2):
                data_a = crop_top_features[cls_a]
                data_b = crop_top_features[cls_b]
                
                # THEORETICAL RULE: 
                # If they rely on the exact same set of macro groups, MS blurs them. Hyperspectral required.
                # If they don't perfectly overlap, MS has distinct filters to capture the variance.
                if data_a["groups"] == data_b["groups"]:
                    hyperspectral_pairs.append((data_a, data_b))
                else:
                    separable_pairs.append((data_a, data_b))
            
            # 3. Generate the Text Report
            report_path = os.path.join(out_dir, f"{model_name}_split_{split}_theoretical_separability.txt")
            
            with open(report_path, "w") as f:
                f.write(f"THEORETICAL SEPARABILITY REPORT\n")
                f.write(f"Model: {model_name.upper()}\n")
                f.write(f"Split: {split}\n")
                f.write("="*70 + "\n\n")
                
                f.write("--- 1. EASILY SEPARABLE BY MULTISPECTRAL (Distinct Macro-Groups) ---\n")
                f.write("These crops rely on at least one completely different physical spectral range.\n\n")
                
                if not separable_pairs:
                    f.write("  None found.\n\n")
                for data_a, data_b in separable_pairs:
                    f.write(f">> {data_a['name']}  vs  {data_b['name']}\n")
                    
                    bands_a_str = " | ".join([f"{b['wavelength']} ({b['group']})" for b in data_a['details']])
                    bands_b_str = " | ".join([f"{b['wavelength']} ({b['group']})" for b in data_b['details']])
                    
                    f.write(f"     {data_a['name']:<30} Top Bands: {bands_a_str}\n")
                    f.write(f"     {data_b['name']:<30} Top Bands: {bands_b_str}\n\n")
                    
                f.write("\n" + "="*70 + "\n")
                f.write("--- 2. STRICTLY REQUIRE HYPERSPECTRAL (Overlapping Macro-Groups) ---\n")
                f.write("These crops rely on the exact same spectral ranges. Multispectral broad bands will likely blur them together.\n\n")
                
                if not hyperspectral_pairs:
                    f.write("  None found.\n\n")
                for data_a, data_b in hyperspectral_pairs:
                    f.write(f">> {data_a['name']}  vs  {data_b['name']}\n")
                    
                    bands_a_str = " | ".join([f"{b['wavelength']} ({b['group']})" for b in data_a['details']])
                    bands_b_str = " | ".join([f"{b['wavelength']} ({b['group']})" for b in data_b['details']])
                    
                    f.write(f"     {data_a['name']:<30} Top Bands: {bands_a_str}\n")
                    f.write(f"     {data_b['name']:<30} Top Bands: {bands_b_str}\n\n")

    print(f"\n{'='*70}\nAnalysis complete! Reports saved to: {out_dir}\n{'='*70}")