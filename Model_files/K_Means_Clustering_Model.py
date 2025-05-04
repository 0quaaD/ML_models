import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=5000,
                  centers=[
                      [0,1],
                      [3,5],
                      [-4,-6],
                      [2,-1],],
                  cluster_std=0.9)

k_means = KMeans(n_clusters=4,
                 init='k-means++',
                 n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

fig = plt.figure(figsize=(10,6))
colors = plt.cm.tab10(np.linspace(0,1,len(set(k_means_labels))))
ax = fig.add_subplot(1,1,1)

for k, col in zip(range(len([[0,1],[3,5],[-4,-6],[2,-1]])),colors):
  my_members = (k_means_labels == k)
  cluster_center = k_means_cluster_centers[k]
  ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
  ax.plot(cluster_center[0],cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.show()
