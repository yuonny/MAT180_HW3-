import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import random
from pathlib import Path
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt

def heirichal_clustering(files):

    X = torch.stack([torch.load(f, map_location="cpu").flatten() for f in files], dim=0)
    X = X.detach().cpu().numpy()   # shape should be (10, 4096)

    h_link = linkage(X, method='average', metric='cosine')
    
    plt.figure(figsize=(10, 5))
    dendrogram(h_link)
    plt.title('Hierarchical Clustering)')
    plt.xlabel('Sample')
    plt.ylabel('Distance')
    plt.show()



def main():
    gram_neg_hosts = Path("evo2/gram_neg/hosts")
    gram_pos_hosts = Path("evo2/gram_pos/hosts")

    gnh_tensor_files = sorted(gram_neg_hosts.glob("*.pt"))
    gph_tensor_files = sorted(gram_pos_hosts.glob("*.pt"))
    
    print("Starting Hierichal Clustering")
    
    random.seed(0)
    gnh_samp_h = random.sample(gnh_tensor_files, k=5)
    gph_samp_h = random.sample(gph_tensor_files, k=5)

    h_files = gnh_samp_h + gph_samp_h
    heirichal_clustering(files = h_files)
    
    print("Starting K-Means")
    
    random.seed(1)
    gnh_samp_k = random.sample(gnh_tensor_files, k=5)
    gph_samp_k = random.sample(gph_tensor_files, k=5)

    k_files = gnh_samp_h + gnh_samp_k +  gph_samp_h + gph_samp_k
    labels = (["gram_neg"] * 10) + (["gram_pos"] * 10)
    
    X = torch.stack([torch.load(f, map_location="cpu").flatten() for f in k_files], dim=0)
    X = X.detach().cpu().numpy() 
    
    K_means = KMeans(n_clusters = 2, random_state = 0)
    K_means.fit(X)

    
    pred_k = K_means.labels_
    true = np.array([0]*10 + [1]*10)  # 0=neg, 1=pos

    acc1_k = np.mean(pred_k == true)
    acc2_k = np.mean((1 - pred_k) == true)
    print("K-Means best accuracy:", max(acc1_k, acc2_k))
    print("K-Means Predicted labels:", pred_k)

    
    print("Starting Spectral Clustering")
    clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(X)
    
    pred_s = clustering.labels_
    
    acc1_s = np.mean(pred_s == true)
    acc2_s = np.mean((1 - pred_s) == true)
    print("Spectral Clustering best accuracy:", max(acc1_s, acc2_s))
    
    pred_s_flipped = 1 - pred_s
    print("Spectral Clustering Predicted labels:", pred_s_flipped)
    
    print("true labels:", labels)
    
if __name__ == "__main__":
    main()