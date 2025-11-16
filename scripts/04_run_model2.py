"""
Script to train and evaluate BinaryClassificationTE model (Takens Embedding approach).
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model2 import BinaryClassificationTE
import src.utils as ut
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from persim import wasserstein, bottleneck

def main():
    """Main execution function."""
    print("=" * 70)
    print("BinaryClassificationTE Model Evaluation")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(28)
    
    # Load datasets
    X_train, y_train, X_test, y_test = ut.load_datasets(max_samples=100)
    
    # Initialize model with best configuration
    print(f"\n{'=' * 70}")
    print("Model Configuration")
    print(f"{'=' * 70}")
    
    model = BinaryClassificationTE(
        distance=wasserstein,
        weights=(1,),
        dim=4,
        tau=5,
        stride=1,
        sample=20,
        thresh=np.inf,
        alpha=0.5,
        max_points=100
    )
    
    # Training
    print(f"\n{'─' * 70}")
    print("Training...")
    start_time = time.time()
    model.fit(X_train, y_train, verbose=True)
    train_time = time.time() - start_time
    print(f"✓ Training completed in {train_time/60:.2f} minutes")
    
    # Evaluation
    print(f"\n{'─' * 70}")
    print("Evaluating...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    eval_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"✓ Evaluation completed in {eval_time/60:.2f} minutes")
    
    # Results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Event', 'Event']))
    
    print("\n" + "=" * 70)
    print("✓ Model evaluation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
