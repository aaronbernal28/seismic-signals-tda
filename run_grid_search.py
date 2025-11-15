"""
Randomized Grid Search with Cross-Validation for both BinaryClassificationTE and BinaryClassificationMFCC models.
Main metric: ROC AUC
Uses sklearn.model_selection.RandomizedSearchCV
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))  # Add repository root to path
import numpy as np
import pandas as pd
from src.models.model2 import BinaryClassificationTE
from src.models.model3 import BinaryClassificationMFCC
import src.utils as ut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from persim import wasserstein, bottleneck

def randomized_search_te(X, y, param_distributions, n_iter=10, cv=3, random_state=28, n_jobs=1):
    """Randomized search for BinaryClassificationTE using sklearn's RandomizedSearchCV.
    
    Args:
        X: Training data (list of arrays)
        y: Training labels (numpy array)
        param_distributions: Dictionary with parameter distributions to sample from
        n_iter: Number of random combinations to try
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        RandomizedSearchCV: Fitted search object with results
    """
    print("=" * 80)
    print("RANDOMIZED GRID SEARCH: BinaryClassificationTE")
    print("=" * 80)
    print(f"Parameters to search: {list(param_distributions.keys())}")
    print(f"Number of iterations: {n_iter}")
    print(f"Cross-validation folds: {cv}")
    print("=" * 80)
    
    # Initialize base model
    base_model = BinaryClassificationTE()
    
    # Create scorer for ROC AUC
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=auc_scorer,
        random_state=random_state,
        verbose=2,
        n_jobs=n_jobs,
        return_train_score=True
    )
    
    # Fit the random search
    print("\nStarting randomized search...\n")
    random_search.fit(X, y)
    
    return random_search


def randomized_search_mfcc(X, y, param_distributions, n_iter=10, cv=3, random_state=28, n_jobs=1):
    """Randomized search for BinaryClassificationMFCC using sklearn's RandomizedSearchCV.
    
    Args:
        X: Training data (list of arrays)
        y: Training labels (numpy array)
        param_distributions: Dictionary with parameter distributions to sample from
        n_iter: Number of random combinations to try
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        RandomizedSearchCV: Fitted search object with results
    """
    print("=" * 80)
    print("RANDOMIZED GRID SEARCH: BinaryClassificationMFCC")
    print("=" * 80)
    print(f"Parameters to search: {list(param_distributions.keys())}")
    print(f"Number of iterations: {n_iter}")
    print(f"Cross-validation folds: {cv}")
    print("=" * 80)
    
    # Initialize base model
    base_model = BinaryClassificationMFCC()
    
    # Create scorer for ROC AUC
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=auc_scorer,
        random_state=random_state,
        verbose=2,
        n_jobs=n_jobs,
        return_train_score=True
    )
    
    # Fit the random search
    print("\nStarting randomized search...\n")
    random_search.fit(X, y)
    
    return random_search


def display_results(search, model_name):
    """Display and format results from RandomizedSearchCV.
    
    Args:
        search: Fitted RandomizedSearchCV object
        model_name: Name of the model for display
        
    Returns:
        pd.DataFrame: Results sorted by mean test score
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(search.cv_results_)
    
    # Select relevant columns
    columns_to_keep = [col for col in results_df.columns if col.startswith('param_') or 
                       col in ['mean_test_score', 'std_test_score', 'rank_test_score', 
                               'mean_train_score', 'std_train_score', 'mean_fit_time']]
    results_df = results_df[columns_to_keep]
    
    # Rename columns for clarity
    results_df = results_df.rename(columns={
        'mean_test_score': 'mean_auc',
        'std_test_score': 'std_auc',
        'mean_train_score': 'mean_train_auc',
        'std_train_score': 'std_train_auc',
        'mean_fit_time': 'fit_time',
        'rank_test_score': 'rank'
    })
    
    # Remove 'param_' prefix from parameter columns
    results_df.columns = [col.replace('param_', '') if col.startswith('param_') else col 
                          for col in results_df.columns]
    
    # Sort by mean AUC
    results_df = results_df.sort_values('rank').reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {model_name}")
    print("=" * 80)
    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best AUC score: {search.best_score_:.4f}")
    print("\nTop 5 configurations:")
    print(results_df.head(5).to_string())
    
    return results_df


def main():
    """Main execution function."""
    print("=" * 80)
    print("RANDOMIZED GRID SEARCH WITH CROSS-VALIDATION")
    print("Main Metric: ROC AUC")
    print("=" * 80)
    
    # Set random seed
    np.random.seed(28)
    
    # Load datasets
    X_train, y_train, X_test, y_test = ut.load_datasets()
    
    # =========================================================================
    # HYPERPARAMETER DISTRIBUTIONS - TO BE COMPLETED
    # =========================================================================
    
    # Parameter distributions for BinaryClassificationTE (Takens Embedding)
    param_distributions = {
        'distance': [wasserstein, bottleneck],  # distance metric between persistence diagrams
        'weights': [[1], [2, 1]],  # Weights for distance calculation, the length depends on homology dimensions used
        'sample': [10, 20, 30, None],  # Number of diagrams to sample
        'thresh': [np.inf, 5000, 10000],  # Threshold
        'alpha': [0.25, 0.5, 0.75, 1.0],  # FPS subsampling proportion
        'max_points': [100, 200, 300, np.inf],  # Maximum number of Takens points
        'seed': [28]  # Random seed for reproducibility
    }

    # Parameter distributions for BinaryClassificationTE (Takens Embedding)
    param_distributions_te = {
        **param_distributions,
        'dim': [3, 4, 5, 10, 20],  # Takens embedding dimension
        'tau': [3, 4, 5, 10],  # Time delay
        'stride': [1, 2],  # Stride
    }
    
    # Parameter distributions for BinaryClassificationMFCC (MFCC Features)
    param_distributions_mfcc = {
        **param_distributions,
        'n_mfcc': [13, 20, 40, 60],  # Number of MFCC coefficients
        'win_length_sec': [0.2, 0.3, 0.4],  # Window length win_length = int(sr * win_length_sec)
        'hop_length_sec': [0.1, 0.15, 0.2],  # Hop length hop_length = int(sr * hop_length_sec)
    }
    
    # =========================================================================
    # GRID SEARCH PARAMETERS
    # =========================================================================
    n_iter = 10  # Number of random parameter combinations to try per model
    cv_folds = 3  # Number of cross-validation folds
    random_state = 28  # Random seed for reproducibility
    n_jobs = -1  # Number of parallel jobs for RandomizedSearchCV
    
    # =========================================================================
    # RUN RANDOMIZED SEARCH FOR MODEL 2 (Takens Embedding)
    # =========================================================================
    print("\n")
    search_te = randomized_search_te(
        X_train, y_train, 
        param_distributions=param_distributions_te,
        n_iter=n_iter,
        cv=cv_folds,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Display and save results
    results_te = display_results(search_te, "BinaryClassificationTE")
    
    output_path_te = Path("data/results/grid_search_te.csv")
    output_path_te.parent.mkdir(exist_ok=True)
    results_te.to_csv(output_path_te, index=False)
    print(f"\n✓ Results saved to: {output_path_te}")
    
    # =========================================================================
    # RUN RANDOMIZED SEARCH FOR MODEL 3 (MFCC)
    # =========================================================================
    print("\n\n")
    search_mfcc = randomized_search_mfcc(
        X_train, y_train,
        param_distributions=param_distributions_mfcc,
        n_iter=n_iter,
        cv=cv_folds,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Display and save results
    results_mfcc = display_results(search_mfcc, "BinaryClassificationMFCC")
    
    output_path_mfcc = Path("data/results/grid_search_mfcc.csv")
    output_path_mfcc.parent.mkdir(exist_ok=True)
    results_mfcc.to_csv(output_path_mfcc, index=False)
    print(f"\n✓ Results saved to: {output_path_mfcc}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("RANDOMIZED SEARCH COMPLETED")
    print("=" * 80)
    print(f"\nBest BinaryClassificationTE:")
    print(f"  AUC: {search_te.best_score_:.4f}")
    print(f"  Params: {search_te.best_params_}")
    
    print(f"\nBest BinaryClassificationMFCC:")
    print(f"  AUC: {search_mfcc.best_score_:.4f}")
    print(f"  Params: {search_mfcc.best_params_}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
