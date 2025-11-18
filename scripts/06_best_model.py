"""
Script: 06_best_model.py
Description: Comprehensive evaluation and exploration of the best performing TDA model on test data.
             Parameters are taken from Rank 1 of data/results/grid_search_mfcc.csv
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve
)
from persim import wasserstein

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model3 import BinaryClassificationMFCC
import src.utils as ut

# Configuration for the brdt Model based on run_grid_search.py
BEST_PARAMS = {
    'distance': wasserstein,
    'weights': (1, 1),
    'n_mfcc': 10,
    'sr': 40.0,
    'win_length_sec': 0.2888888888888889,
    'hop_length_sec': 0.5,
    'sample': 13,
    'thresh': 1666.6666666666667,
    'alpha': 0.4666666666666667,
    'max_points': 250
}

# The grid search used seed=28, so we should use it here for reproducibility
MODEL_SEED = 28 

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "best_model_analysis"

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Noise', 'Earthquake'],
                yticklabels=['Noise', 'Earthquake'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_proba, auc_score, save_path):
    """Generate and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ ROC curve saved to {save_path}")

def analyze_errors(X_test, y_test, y_pred, y_proba):
    """Perform analysis on misclassified samples."""
    # Identify indices
    false_positives = np.where((y_test == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]
    
    print(f"\n{'=' * 40}")
    print("ERROR ANALYSIS")
    print(f"{'=' * 40}")
    print(f"False Positives (Noise predicted as Earthquake): {len(false_positives)}")
    print(f"False Negatives (Earthquake predicted as Noise): {len(false_negatives)}")
    
    if len(false_positives) > 0:
        print("\nTop False Positives (highest confidence wrong predictions):")
        # Sort by probability of being positive
        fp_probs = y_proba[false_positives]
        sorted_indices = np.argsort(fp_probs)[::-1][:3] # Top 3
        
        for idx in sorted_indices:
            orig_idx = false_positives[idx]
            prob = fp_probs[idx]
            sig = X_test[orig_idx]
            print(f"  - Index {orig_idx}: Prob(Earthquake)={prob:.4f}, Signal Mean={np.mean(sig):.2e}, Std={np.std(sig):.2e}")

    if len(false_negatives) > 0:
        print("\nTop False Negatives (lowest confidence wrong predictions):")
        # Sort by probability of being positive (we want low prob here, i.e., confident it's negative)
        fn_probs = y_proba[false_negatives]
        sorted_indices = np.argsort(fn_probs)[:3] # Bottom 3
        
        for idx in sorted_indices:
            orig_idx = false_negatives[idx]
            prob = fn_probs[idx]
            sig = X_test[orig_idx]
            print(f"  - Index {orig_idx}: Prob(Earthquake)={prob:.4f}, Signal Mean={np.mean(sig):.2e}, Std={np.std(sig):.2e}")

def main():
    """Main execution flow."""
    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    
    print("=" * 70)
    print("BEST MODEL EVALUATION & EXPLORATION")
    print("=" * 70)
    print("Using parameters from Rank 1 of grid_search_mfcc.csv")
    print(f"Params: {BEST_PARAMS}")
    
    # 1. Load Data
    X_train, y_train, X_test, y_test = ut.load_datasets(max_samples=None) # Using full dataset for final evaluation
    
    # 2. Initialize Model
    print(f"\nInitializing BinaryClassificationMFCC with optimal parameters...")
    model = BinaryClassificationMFCC(**BEST_PARAMS, seed=MODEL_SEED)
    
    # 3. Train
    print(f"\n{'─' * 70}")
    print("Training Model...")
    start_time = time.time()
    model.fit(X_train, y_train, verbose=True)
    train_time = time.time() - start_time
    print(f"✓ Training completed in {train_time:.2f} seconds")
    
    # 4. Evaluation
    print(f"\n{'─' * 70}")
    print("Evaluating on Test Data...")
    start_time = time.time()
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba_all = model.predict_proba(X_test)
    y_proba = y_proba_all[:, 1]  # Probability of positive class
    
    eval_time = time.time() - start_time
    print(f"✓ Prediction completed in {eval_time:.2f} seconds")
    
    # 5. Metrics Calculation
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=['Noise', 'Earthquake'])
    
    # Print Metrics
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    # 6. Visualizations
    print(f"\n{'=' * 70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 70}")
    
    plot_confusion_matrix(y_test, y_pred, RESULTS_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, auc, RESULTS_DIR / "roc_curve.png")
    
    # 7. Error Analysis
    analyze_errors(X_test, y_test, y_pred, y_proba)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Evaluation Complete. Results saved to: {RESULTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()