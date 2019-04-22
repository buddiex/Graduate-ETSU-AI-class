import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris

#
iris = load_iris()
data = iris.data  # Contains the four features for each flower (150 records)
target = iris.target  # Contains the type of flower
clf = svm.SVC(kernel='rbf', gamma=0.7)

# s_length = data[:, 0]
# s_width = data[:, 1]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(s_length, s_width, c=target)
# ax1.set_xlabel("Sepal Length (cm)")
# ax1.set_ylabel("Sepal Width (cm)")
# # plt.show()
#
#
# fit = clf.fit(data[:, :2], target)
# s_vec = fit.support_vectors_
# print("support vectors...")
# # print(s_vec)
# ax1.scatter(s_vec[:, 0], s_vec[:, 1], cmap='viridis', marker='s', edgecolor='k', s=100)
# plt.show()
#
#
# figure = plt.figure()
# dataplot = figure.add_subplot(111, projection='3d')
#
# for i in range(len(data)):
#     x = data[i][0]
#     y = data[i][1]
#     z = target[i]
#     color = ''
#     if (z == 0):
#         color = 'r'
#     elif (z == 1):
#         color = 'b'
#     else:
#         color = 'g'
#     dataplot.scatter(x, y, z, marker='s', c=color)


X = data
y = target
h = .02
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
fit = clf.fit(data[:, :2], target)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("RBF Separators")
plt.show()


print("predictions")
vals =  [[ 7.4, 3.4], [ 5.0, 1.5], [ 4.75, 4.0], [5.9, 3.5], [3.5, 1.1]]
vals = np.array(vals)
for v in vals:
    print(fit.predict([v]))

plt.scatter(vals[:, 0], vals[:, 1], s=150, c='k')

plt.show()