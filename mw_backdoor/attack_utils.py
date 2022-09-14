import os
import numpy as np

def get_poisoning_candidate_samples(dataset,original_model, X_test, y_test):
    if dataset == 'ember':
        if os.path.exists(f'./dataset/x_mw_poisoning_candidates.npy'):
            # print('Found poisoning_candidate_samples')
            return np.load(f'./dataset/x_mw_poisoning_candidates.npy'),np.load(f'./dataset/x_mw_poisoning_candidates_idx.npy')
        else:
            X_test = X_test[y_test == 1]
            y = original_model.predict(X_test)
            if y.ndim > 1:
                y = y.flatten()
            correct_ids = y > 0.5
            X_mw_poisoning_candidates = X_test[correct_ids]
            return X_mw_poisoning_candidates, correct_ids
    else:
        X_test = X_test[y_test == 1]
        y = original_model.predict(X_test)
        if y.ndim > 1:
            y = y.flatten()
        correct_ids = y > 0.5
        X_mw_poisoning_candidates = X_test[correct_ids]
        return X_mw_poisoning_candidates, correct_ids