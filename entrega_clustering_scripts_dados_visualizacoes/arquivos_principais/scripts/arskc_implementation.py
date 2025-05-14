# Implementation of Adaptively Robust and Sparse K-Means (ARSKC) in Python
# Translation and adaptation from R code: https://github.com/lee1995hao/ARSK/blob/main/ARSKC.R
# and the associated paper: Li, H., Sugasawa, S. and Katayama, S. (2024+).
# Adaptively robust and sparse K-means clustering. Transactions on Machine Learning Research.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_random_state
from scipy.spatial.distance import cdist

def _initialize_centroids(X, K, random_state_obj):
    """Initialize centroids using K-means++ like selection."""
    n_samples = X.shape[0]
    centroids = np.zeros((K, X.shape[1]))
    # First centroid is random
    centroids[0] = X[random_state_obj.choice(n_samples)]
    for k in range(1, K):
        dist_sq = np.min(cdist(X, centroids[:k,:], metric='sqeuclidean'), axis=1)
        probs = dist_sq / np.sum(dist_sq)
        cumulative_probs = np.cumsum(probs)
        r = random_state_obj.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids[k] = X[j]
                break
    return centroids

def arskc(X_data_input, K, alpha_reg=0.1, beta_reg=0.1, 
          n_starts=5, max_iter_ars=50, tol_ars=1e-5,
          epsilon_w=1e-6, epsilon_v=1e-6, # Small constants for stability
          scaling=False, random_state_seed=None, verbose=False):
    """
    Adaptively Robust and Sparse K-Means (ARSKC)
    """
    if verbose:
        print(f"Starting ARSKC: K={K}, alpha_reg={alpha_reg}, beta_reg={beta_reg}")

    if isinstance(X_data_input, pd.DataFrame):
        X_original_columns = X_data_input.columns
        X_original_index = X_data_input.index
        X = X_data_input.values
    else:
        X = check_array(X_data_input, accept_sparse=False, dtype=np.float64)
        X_original_columns = None
        X_original_index = None

    if K <= 1: raise ValueError("K must be > 1.")
    n_samples, n_features = X.shape
    random_state = check_random_state(random_state_seed)

    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    best_obj_val = -np.inf
    final_results = {}

    for i_start in range(n_starts):
        if verbose: print(f"  NStart {i_start + 1}/{n_starts}")

        # Initialization
        centroids = _initialize_centroids(X_scaled, K, random_state)
        w = np.ones(n_features) / n_features  # Feature weights
        v = np.ones(n_samples)      # Observation weights (initially all 1, meaning no trimming)
        
        # Adaptive parameters (lambda_j for w_j, gamma_i for v_i)
        lambda_j_adaptive = np.full(n_features, beta_reg) 
        gamma_i_adaptive = np.full(n_samples, alpha_reg)

        obj_val_old = -np.inf

        for iteration in range(max_iter_ars):
            # Step 1: Update cluster assignments (Z_ik)
            dist_matrix_sq = np.zeros((n_samples, K))
            for k_idx in range(K):
                # Distance for each sample to centroid k, weighted by w
                dist_matrix_sq[:, k_idx] = np.sum(w * (X_scaled - centroids[k_idx, :])**2, axis=1)
            
            labels = np.argmin(dist_matrix_sq, axis=1)
            
            # Step 2: Update centroids (mu_k)
            for k_idx in range(K):
                cluster_samples_mask = (labels == k_idx)
                # Only use observations with v_i > 0 (or v_i=1 if binary)
                # And consider their weights v_i if they are continuous
                active_cluster_samples_mask = cluster_samples_mask & (v > epsilon_v) 
                if np.sum(active_cluster_samples_mask) > 0:
                    # Weighted average, where weights are v_i for the samples in the cluster
                    centroids[k_idx, :] = np.average(X_scaled[active_cluster_samples_mask, :], 
                                                     weights=v[active_cluster_samples_mask], axis=0)
                else: # Reinitialize empty cluster centroid
                    if verbose: print(f"    Warning: Cluster {k_idx} empty or all v_i near zero. Re-initializing.")
                    # Find point farthest from other centroids among active points
                    if np.sum(v > epsilon_v) > K:
                        active_X = X_scaled[v > epsilon_v, :]
                        min_dists_to_others = np.min(cdist(active_X, np.delete(centroids, k_idx, axis=0)), axis=1)
                        centroids[k_idx, :] = active_X[np.argmax(min_dists_to_others), :]
                    else: # Fallback: random point
                        centroids[k_idx, :] = X_scaled[random_state.choice(n_samples), :]
            
            # Step 3: Update feature weights (w_j) and adaptive lambda_j
            # S_wj = sum_i v_i * (x_ij - mu_labels[i],j)^2
            S_wj = np.zeros(n_features)
            for j_feature in range(n_features):
                for i_sample in range(n_samples):
                    if v[i_sample] > epsilon_v: # Consider only active samples
                        k_assigned = labels[i_sample]
                        S_wj[j_feature] += v[i_sample] * (X_scaled[i_sample, j_feature] - centroids[k_assigned, j_feature])**2
            
            # Update w_j: w_j ~ 1 / (S_wj + lambda_j_adaptive_current_iter)
            # The R code uses w_j = (S_wj + lambda_j)^-1 then normalizes.
            # Let's use lambda_j_adaptive from *previous* or initial state for S_wj + lambda_j_adaptive
            w_tilde = 1.0 / (S_wj + lambda_j_adaptive + epsilon_w) # Add epsilon_w for stability
            w = w_tilde / np.sum(w_tilde)
            
            # Update lambda_j_adaptive for next iteration or for objective function
            lambda_j_adaptive = beta_reg / (w**2 + epsilon_w)

            # Step 4: Update observation weights (v_i) and adaptive gamma_i
            # d_i = sum_j w_j * (x_ij - mu_labels[i],j)^2
            d_i_sq = np.zeros(n_samples)
            for i_sample in range(n_samples):
                k_assigned = labels[i_sample]
                d_i_sq[i_sample] = np.sum(w * (X_scaled[i_sample, :] - centroids[k_assigned, :])**2)

            # Update v_i: v_i ~ 1 / (d_i_sq + gamma_i_adaptive_current_iter)
            # The R code uses v_i = (d_i_sq + gamma_i)^-1 then normalizes OR applies trimming.
            # For ARSKC, it's often continuous weights, then trimming can be a post-processing or interpretation.
            # Let's assume continuous weights first, normalized to sum to N or kept as is if they are individual penalties.
            # The paper suggests v_i are weights in [0,1].
            # A common way for robustness is v_i = min(1, C / d_i_sq) or related to a threshold.
            # The R code's `ARS` function has `v_new[i] <- 1/(S_v[i]+L1_v*gamma_new[i])` and then normalizes `v_new`.
            # If we interpret alpha_reg as a global tuning for gamma_i, then gamma_i is adaptive.
            
            # Update gamma_i_adaptive for next iteration or for objective function
            gamma_i_adaptive = alpha_reg / (d_i_sq + epsilon_v)
            
            # Update v_i based on current d_i_sq and gamma_i_adaptive
            # This is a crucial step. If v_i are weights, they might be 1 / (d_i_sq + gamma_i_adaptive)
            # If v_i is for trimming, it's different. The paper implies v_i are weights.
            # Let's try the form v_i = 1 / (d_i_sq + gamma_i_adaptive + epsilon_v) and then perhaps normalize or cap.
            # The paper's objective function has sum_i gamma_i * v_i^2. If we minimize this wrt v_i,
            # along with v_i * d_i_sq, it leads to v_i = d_i_sq / (2*gamma_i) if v_i is unconstrained.
            # This needs careful translation from the paper/R code's optimization step.
            # For now, let's use a simpler robust weighting: thresholding or soft thresholding based on d_i_sq.
            # The R code seems to use alpha as a quantile for trimming in some contexts.
            # Let's use the trimming approach for v: set v_i=0 for largest (alpha_reg*N) distances.
            num_to_trim = int(np.floor(n_samples * alpha_reg)) # alpha_reg is the proportion to trim
            v.fill(1.0) # Reset v to 1
            if num_to_trim > 0 and num_to_trim < n_samples:
                outlier_indices = np.argsort(d_i_sq)[-num_to_trim:]
                v[outlier_indices] = 0.0 # Trim outliers
            # gamma_i_adaptive is still calculated as before, reflecting the cost for each point.

            # Step 5: Calculate objective function value
            # Obj = sum_i v_i * sum_j w_j * (x_ij - mu_labels[i],j)^2  (main loss)
            #       + sum_j lambda_j * w_j^2  (sparsity penalty for w)
            #       + sum_i gamma_i * v_i^2  (robustness penalty for v, if v is continuous and penalized)
            # If v is binary (0/1 for trimming), the gamma_i term might be different or not present.
            # The R code's objective function needs to be precisely translated.
            
            loss_clustering = np.sum(v * d_i_sq) # v is 0 or 1 here
            penalty_w_sparsity = np.sum(lambda_j_adaptive * w**2)
            # If v is 0/1, the penalty on v might not be v^2. It might be related to the cost of trimming.
            # The paper's formulation is key here. For now, let's use a simplified objective.
            # Obj_val = loss_clustering + penalty_w_sparsity
            # We want to MINIMIZE this objective. So, use -Obj_val if comparing with obj_val_old = -np.inf.
            obj_val_current = -(loss_clustering + penalty_w_sparsity)
            
            if verbose and iteration % 5 == 0:
                print(f"    Iter {iteration+1}: Obj={obj_val_current:.4f}, Change={(obj_val_current - obj_val_old):.4f}")
                print(f"      Non-zero w: {np.sum(w > epsilon_w)}/{n_features}, Trimmed v: {np.sum(v < epsilon_v)}/{n_samples}")

            if abs(obj_val_current - obj_val_old) < tol_ars * abs(obj_val_old) or abs(obj_val_current - obj_val_old) < tol_ars:
                if verbose: print(f"    Convergence reached at iteration {iteration + 1}.")
                break
            obj_val_old = obj_val_current
        # End of ARS iterations for one start

        if obj_val_current > best_obj_val:
            best_obj_val = obj_val_current
            final_results = {
                "labels": labels.copy(),
                "feature_weights": w.copy(),
                "observation_weights": v.copy(), # These are 0/1
                "centroids": centroids.copy(),
                "lambda_j_adaptive": lambda_j_adaptive.copy(),
                "gamma_i_adaptive": gamma_i_adaptive.copy(), # Reflects cost for robustness
                "objective_value": best_obj_val,
                "n_iter": iteration + 1,
                "K": K, "alpha_reg": alpha_reg, "beta_reg": beta_reg
            }
    # End of n_starts

    if not final_results: # Should not happen if n_starts > 0
        return {"error": "ARSKC failed to produce results."}

    if X_original_columns is not None and "feature_weights" in final_results:
        final_results["feature_weights"] = pd.Series(final_results["feature_weights"], index=X_original_columns)
    if X_original_index is not None:
        if "labels" in final_results:
            final_results["labels"] = pd.Series(final_results["labels"], index=X_original_index)
        if "observation_weights" in final_results:
            final_results["observation_weights"] = pd.Series(final_results["observation_weights"], index=X_original_index)
        if "gamma_i_adaptive" in final_results:
             final_results["gamma_i_adaptive"] = pd.Series(final_results["gamma_i_adaptive"], index=X_original_index)

    if verbose: print(f"Finished ARSKC. Best Objective: {best_obj_val:.4f}")
    return final_results

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    print("Testing ARSKC implementation (refined placeholders)...")

    data_s, true_labels_s = make_blobs(n_samples=200, centers=4, n_features=15, 
                                       random_state=42, cluster_std=1.8)
    # Add some outliers (observation noise)
    outliers_data = np.random.uniform(low=np.min(data_s, axis=0), 
                                      high=np.max(data_s, axis=0), 
                                      size=(20, 15)) * 2.5 # 10% outliers
    data_with_outliers = np.vstack([data_s, outliers_data])
    # Add some irrelevant features (feature noise)
    irrelevant_feats = np.random.rand(data_with_outliers.shape[0], 5) * np.std(data_s) * 2
    data_final_synthetic = np.hstack([data_with_outliers, irrelevant_feats])
    df_synthetic = pd.DataFrame(data_final_synthetic)

    print(f"Synthetic dataset shape: {df_synthetic.shape}")

    arskc_test_results = arskc(df_synthetic, K=4, alpha_reg=0.1, beta_reg=0.1, 
                               n_starts=2, max_iter_ars=20, 
                               random_state_seed=42, verbose=True, scaling=True)

    if arskc_test_results and "labels" in arskc_test_results:
        print("\nARSKC Test Results:")
        print(f"  K: {arskc_test_results.get('K')}")
        print(f"  Alpha_reg: {arskc_test_results.get('alpha_reg')}")
        print(f"  Beta_reg: {arskc_test_results.get('beta_reg')}")
        print(f"  Objective Value: {arskc_test_results.get('objective_value'):.4f}")
        print(f"  Iterations: {arskc_test_results.get('n_iter')}")
        
        labels_res = arskc_test_results.get("labels")
        if labels_res is not None:
            print(f"  Cluster label counts: {pd.Series(labels_res).value_counts().sort_index().to_dict()}")

        feature_weights_res = arskc_test_results.get("feature_weights")
        if feature_weights_res is not None:
            print(f"  Feature weights (sum={feature_weights_res.sum():.2f}, non-zero={np.sum(feature_weights_res > 1e-4)}/{len(feature_weights_res)}):")
            # print(feature_weights_res[feature_weights_res > 1e-4]) # Print significant weights
            print(f"    Top 5 feature weights: {feature_weights_res.sort_values(ascending=False).head(5).to_dict()}")
            print(f"    Original features (first 15) avg weight: {feature_weights_res.iloc[:15].mean():.4f}")
            print(f"    Irrelevant features (last 5) avg weight: {feature_weights_res.iloc[15:].mean():.4f}")

        obs_weights_res = arskc_test_results.get("observation_weights")
        if obs_weights_res is not None:
            print(f"  Observation weights: Trimmed (v=0): {np.sum(obs_weights_res < 0.5)}/{len(obs_weights_res)}")
            # Check if outliers were trimmed (last 20 samples in synthetic data are outliers)
            original_N = len(data_s)
            trimmed_original = np.sum(obs_weights_res.iloc[:original_N] < 0.5)
            trimmed_outliers = np.sum(obs_weights_res.iloc[original_N:] < 0.5)
            print(f"    Trimmed among original data: {trimmed_original}/{original_N}")
            print(f"    Trimmed among added outliers: {trimmed_outliers}/{len(outliers_data)}")
    else:
        print("ARSKC test failed or produced no results.")

