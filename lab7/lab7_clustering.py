import random

import matplotlib.pyplot as plt
import numpy as np
from lab7_dendrogram import plot_dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import calinski_harabaz_score

digits = load_digits()
data = digits.data

print(data.shape)

print(data[0])
print(digits.target[0])
plt.gray()
plt.figure(1, figsize=(3, 3))
plt.matshow(digits.images[0])
plt.show()

rnd = random.randint(0,10)

print(data[rnd])
print(digits.target[rnd])
plt.gray()
plt.figure(1, figsize=(3, 3))
plt.matshow(digits.images[rnd])
plt.show()

for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++')
    km.fit(data)
    sc = calinski_harabaz_score(data, km.labels_)
    print("Checking  k=" + str(k), "CHS:", sc)

n_clust = 10
kmeans = KMeans(init='k-means++', n_clusters=n_clust)
fit = kmeans.fit(data)
indices = {i: np.where(fit.labels_ == i)[0] for i in range(fit.n_clusters)}
labels = digits.target


def get_info():
    global index, values, label, val
    for index in range(fit.n_clusters):
        values = indices[int(index)]
        label = [labels[val] for val in values]
        print(50 * "-")
        print("Cluster %s Information" % index)
        for val in set(label):
            print("Value:", str(val), "Count:", str(label.count(val)))


get_info()

hac = AgglomerativeClustering(n_clusters=n_clust)
# fit = hac.fit(data)
fit = hac.fit(data[:50])
plot_dendrogram(fit)
plt.show()
indices = {i: np.where(fit.labels_ == i)[0] for i in range(fit.n_clusters)}
labels = digits.target
get_info()
