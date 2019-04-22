import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans

data = np.loadtxt("lab10_linear.txt")
plt.scatter(data[:, 0], data[:, 1])
plt.show()

k = 4
km = KMeans(n_clusters=k, init='k-means++')
km.fit(data)
labels = km.labels_
plt.scatter(data[:, 0], data[:, 1], c=km.labels_)

clf = svm.SVC(kernel='linear')
fit = clf.fit(data, labels)

s_vec = fit.support_vectors_
print("support vectors...")
print(fit.support_vectors_)
# plt.scatter(s_vec[:, 0], s_vec[:, 1], s=150, c='r')
# plt.show()

# # create a mesh to plot in
# X = data
# y = labels
# h = .02
# x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
# y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
#
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
# plt.title("Linear Separators")
# plt.show()

print("predictions")
vals = [
    [12.0, - 15.0],
    [-12.0, -15.0],
    [-10.0, 20.0],
    [2.9, 5.3],
    [8.7, 0.28]
]
vals = np.array(vals)
for v in vals:
    print(fit.predict([v]))

plt.scatter(vals[:, 0], vals[:, 1], s=150, c='k')

plt.show()
