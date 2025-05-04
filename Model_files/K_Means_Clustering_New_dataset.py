import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("customerSegmentation.csv")
df = df.drop(columns='Address',axis=1)

if(df.isnull().any().any()):
  df = df.dropna()
  print("Data has been cleaned.\n")
else:
  print("Data is already clean.\n")



X = df.values[:,1:]
clust_dataset = StandardScaler().fit_transform(X)
k_means = KMeans(n_clusters=3,
                 init='k-means++',
                 n_init=12)
k_means.fit(clust_dataset)
labels = k_means.labels_
df['CLUS_KM'] = labels

df.groupby("CLUS_KM").mean()

# 2D plot for K-Means Clustering
area = np.pi * (X[:,1])**2
plt.scatter(X[:,0], X[:,3], s=area, c=labels.astype(float), cmap='tab10', ec='k', alpha = 0.5)
plt.title("2D Clustering")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()
print("\n\n")

# 3D plot for K-Means Clustering
fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=k_means_labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot

fig.show()
