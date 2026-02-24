"""
Master script to run complete DREAMER SHAP and PDI analysis pipeline
This will take approximately 10-12 hours for full LOSO with 2 feature methods
"""

import subprocess
import sys
from pathlib import Path
import time

def check_shap_installation():
    """Check if SHAP is installed"""
    try:
        import shap
        print(" SHAP is installed")
        return True
    except ImportError:

        print("\nInstalling SHAP...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
            print(" SHAP installed successfully")
            return True
        except:

            print("Please install manually: pip install shap")
            return False

def run_command(cmd, description):
    """Run a command and handle errors"""

    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n Completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n ailed with error code {e.returncode}")
        return False

def main():
  
    print("\nThis pipeline will:")
    print("1. Check/install SHAP library")
    print("2. Run DREAMER LOSO with model and prediction saving (~10 hours)")
    print("3. Generate SHAP and PDI visualizations\n")
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Check SHAP
     
    print("STEP 1: Checking SHAP Installation")
     
    if not check_shap_installation():
        print("\n SHAP is required. Please install it and try again.")
        return
    
    # Step 2: Run DREAMER LOSO with SHAP

    print("  Models, predictions, and SHAP values will be saved")
    print(" Using: wavelet + lorentzian features, 5 graph types each\n")
    
    response = input("Start training? [y/N]: ")
    if response.lower() != 'y':
        print("Skipping training. You can run manually with:")
        print("python scripts/run_dreamer_with_shap.py")
    else:
        cmd = [
            sys.executable,
            "scripts/run_dreamer_with_shap.py",
            "--data_path", "data/DREAMER/DREAMER.mat",
            "--features", "wavelet", "lorentzian",
            "--graphs", "plv", "coherence", "correlation", "mi", "aec",
            "--hidden_dim", "64",
            "--epochs", "100",
            "--device", "cuda",
            "--out_dir", "dreamer_with_shap",
            "--seed", "42"
        ]
        
        if not run_command(cmd, "Running DREAMER LOSO with SHAP"):
            print("\n Training failed. Check errors above.")
            return
    

    
    # Check if output directory exists
    if not Path("dreamer_with_shap").exists():

        return
    
    cmd = [
        sys.executable,
        "scripts/visualize_dreamer_shap_pdi.py",
        "--dreamer_dir", "dreamer_with_shap",
        "--output_dir", "figures/dreamer"
    ]
    
    if not run_command(cmd, "Generating visualizations"):

        return
    
    # Step 4: Summary

    print("\nOutput locations:")
    print("  - Models: dreamer_with_shap/*/models/")
    print("  - Predictions: dreamer_with_shap/*/predictions_*.npy")
    print("  - SHAP values: dreamer_with_shap/*/shap/")
    print("  - Figures: figures/dreamer/")
    print("\nKey figures generated:")
    print("  - pdi_heatmap_*.png - PDI matrices for each feature method")
    print("  - pdi_distribution_*.png - PDI value distributions")
    print("  - shap_summary_*.png - SHAP feature importance")
    print("  - accuracy_vs_pdi_*.png - Accuracy vs interpretability trade-off")
    print("  - dreamer_cross_feature_comparison.png - Cross-method comparison")
    print("\nNext steps:")
    print("1. Review figures in figures/dreamer/")
    print("2. Compare with DEAP results (if available)")
    print("3. Update LaTeX paper with DREAMER results")
    
if __name__ == '__main__':
    main()
