import time
import hdbscan
import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import MinMaxScaler

def standardize_data(data_mat, feature_range=(-1, 1)):
    std_data = MinMaxScaler(feature_range=feature_range).fit_transform(data_mat)

    return std_data

def compute_silhouettes(data_mat, labels, metric='euclidean', save_dir=''):
    start_time = time.time()
    clus_silh = silhouette_samples(data_mat, labels, metric=metric)
    clus_avg_silh = {k: np.mean(clus_silh[labels == k]) for k in set(labels)}
    print('       Computing silhouettes took: {}'.format(time.time() - start_time))

    return clus_silh, clus_avg_silh


# # noinspection PyUnresolvedReferences
def eval_cluster(labels, cluster_id, is_clean):
    cluster = labels == cluster_id
    identified = 0

    for i in range(len(is_clean)):
        if cluster[i] and is_clean[i] == 0:
            identified += 1

    return identified


def eval_clustering(labels, is_clean):
    return {k: eval_cluster(labels, k, is_clean) for k in set(labels)}


def cluster_hdbscan(data_mat, metric='euclidean', min_clus_size=5, min_samples=None, n_jobs=32, save_dir=''):
    start_time = time.time()
    hdb = hdbscan.HDBSCAN(
        metric=metric,
        core_dist_n_jobs=n_jobs,
        min_cluster_size=min_clus_size,
        min_samples=min_samples
    )
    hdb.fit(data_mat)
    print('       Clustering took: {}'.format(time.time() - start_time))

    hdb_labs = hdb.labels_
    return hdb, hdb_labs

def show_clustering(labels, is_clean, print_mc=10, print_ev=10, avg_silh=None):
    cluster_sizes = Counter(labels)

    print('       Total number of clusters: {}'.format(len(set(labels))))

    if print_mc:
        print('       {} most common cluster sizes:'.format(print_mc))
        print(f'       {cluster_sizes.most_common(print_mc)}')
        print()

    evals = eval_clustering(labels, is_clean)
    return cluster_sizes, evals

def spectral_sign_paper(data_mat):
    clus_avg = np.average(data_mat, axis=0)  # R-hat
    clus_centered = data_mat - clus_avg  # M

    u, s, v = np.linalg.svd(clus_centered, full_matrices=False)

    # From https://github.com/MadryLab/backdoor_data_poisoning/blob/master/compute_corr.py
    eigs = v[0:1]
    corrs = np.matmul(eigs, np.transpose(
        clus_centered))  # shape num_top, num_active_indices
    scores = np.linalg.norm(corrs, axis=0)  # shape num_active_indices
    score_percentile = np.percentile(scores, 85)  # Discard top 15%
    top_scores = np.where(scores > score_percentile)[0]
    # make bitmap with samples to remove
    to_remove = np.zeros(shape=data_mat.shape[0])
    to_remove[top_scores] = 1
    top_scores_indices = set(top_scores.flatten().tolist())

    return to_remove, top_scores, top_scores_indices


def spectral_remove_lists(x_gw_sel_std, bdr_indices):
    to_remove_pa, top_scores_pa, top_scores_indices_pa = spectral_sign_paper(
        x_gw_sel_std)
    found_pa = top_scores_indices_pa.intersection(bdr_indices)
    return to_remove_pa, found_pa
