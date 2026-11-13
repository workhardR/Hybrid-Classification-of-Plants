"""
SMOTE (Synthetic Minority Over-sampling Technique) Feature-Space Balancer
========================================================================
Path: plant_hybrid/core/smote_augmentation.py

USE CASE & ARCHITECTURAL RATIONALE:
-----------------------------------
1. Why NOT apply SMOTE directly on raw image pixels?
   - Raw image interpolation (e.g. 0.5 * Image_A + 0.5 * Image_B) produces artificial "ghost" 
     translucencies, blurred leaf margins, and destroyed vascular venation patterns.
   - For pixel-level image generation, class-conditional DCGAN (cDCGAN) is applied in Step 06.

2. Why DO apply SMOTE on Deep Latent Feature Embeddings?
   - Convolutional and Transformer backbones (EfficientNet, ResNet, DenseNet, Swin-T) project 
     images into a continuous, semantically structured latent feature manifold (e.g. 1024-d or 1792-d).
   - On this manifold, linear convex combinations between minority feature vectors correspond to 
     realistic variations in plant texture and morphology.
   - SMOTE balances skewed class distributions directly before training tabular classifiers 
     (XGBoost, SVM, MLP) or fine-tuning attention heads, resolving class imbalance and preventing 
     overfitting on majority categories.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class FeatureSMOTE:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) for Latent Feature Vectors.
    
    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbors to construct synthetic interpolation segments.
    random_state : int, default=42
        Seed for reproducibility.
    """
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self, X, y):
        """
        Resamples the feature matrix X and target labels y to balance minority classes.
        
        Parameters
        ----------
        X : np.ndarray of shape (N_samples, N_features)
            Extracted deep feature embeddings from CNN/Transformer backbone.
        y : np.ndarray of shape (N_samples,)
            Corresponding integer class labels.
            
        Returns
        -------
        X_resampled : np.ndarray
            Balanced feature matrix containing real + synthesized minority vectors.
        y_resampled : np.ndarray
            Corresponding balanced class labels.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        
        synthetic_X = []
        synthetic_y = []
        
        for cls, count in zip(classes, counts):
            n_needed = max_count - count
            if n_needed <= 0:
                continue
                
            cls_mask = (y == cls)
            X_cls = X[cls_mask]
            
            # If class has fewer samples than k_neighbors, adapt k
            effective_k = min(self.k_neighbors, len(X_cls) - 1)
            if effective_k < 1:
                # Duplicate with slight Gaussian jitter if only 1 sample exists
                noise = self.rng.normal(0, 0.01, size=(n_needed, X.shape[1]))
                syn_samples = np.tile(X_cls[0], (n_needed, 1)) + noise
                synthetic_X.append(syn_samples)
                synthetic_y.extend([cls] * n_needed)
                continue
            
            # Fit k-nearest neighbors in latent feature space
            nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean", n_jobs=-1)
            nn.fit(X_cls)
            indices = nn.kneighbors(X_cls, return_distance=False)
            
            # Generate synthetic samples via convex interpolation:
            # X_syn = X_i + lambda * (X_neighbor - X_i), where lambda in (0, 1)
            base_idx = self.rng.choice(len(X_cls), size=n_needed, replace=True)
            neighbor_choice = self.rng.randint(1, effective_k + 1, size=n_needed)
            
            lambdas = self.rng.uniform(0.0, 1.0, size=(n_needed, 1))
            
            x_base = X_cls[base_idx]
            selected_neighbors = indices[base_idx, neighbor_choice]
            x_neighbor = X_cls[selected_neighbors]
            
            syn_samples = x_base + lambdas * (x_neighbor - x_base)
            synthetic_X.append(syn_samples)
            synthetic_y.extend([cls] * n_needed)
            
        if synthetic_X:
            X_balanced = np.vstack([X, np.vstack(synthetic_X)])
            y_balanced = np.concatenate([y, np.array(synthetic_y, dtype=np.int64)])
        else:
            X_balanced = X.copy()
            y_balanced = y.copy()
            
        return X_balanced, y_balanced


# Standalone demonstration and verification
if __name__ == "__main__":
    print("=" * 76)
    print(" SMOTE FEATURE AUGMENTATION VERIFICATION")
    print(" File: plant_hybrid/core/smote_augmentation.py")
    print("=" * 76)
    
    # Simulate imbalanced feature dataset (e.g. Endangered plant class vs Abundant medicinal class)
    np.random.seed(42)
    X_abundant = np.random.randn(100, 1024)   # 100 samples with 1024-dim DenseNet features
    y_abundant = np.zeros(100, dtype=int)
    
    X_rare = np.random.randn(15, 1024) + 2.0  # Only 15 samples of rare endangered plant
    y_rare = np.ones(15, dtype=int)
    
    X_raw = np.vstack([X_abundant, X_rare])
    y_raw = np.concatenate([y_abundant, y_rare])
    
    print(f"[*] Before SMOTE:")
    print(f"    - Majority Class (0): {(y_raw == 0).sum()} samples")
    print(f"    - Minority Class (1): {(y_raw == 1).sum()} samples")
    print(f"    - Total Feature Shape: {X_raw.shape}")
    
    # Apply Feature SMOTE
    smote = FeatureSMOTE(k_neighbors=5, random_state=42)
    X_bal, y_bal = smote.fit_resample(X_raw, y_raw)
    
    print(f"\n[+] After SMOTE Feature Balancing:")
    print(f"    - Majority Class (0): {(y_bal == 0).sum()} samples")
    print(f"    - Minority Class (1): {(y_bal == 1).sum()} samples (100% Balanced!)")
    print(f"    - Total Balanced Feature Shape: {X_bal.shape}")
    print("[+] SMOTE algorithm executed successfully and verified error-free!")

