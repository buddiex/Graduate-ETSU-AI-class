import random

import matplotlib.pyplot as plt
import pydotplus
from sklearn import datasets
from sklearn import tree, metrics
from sklearn.naive_bayes import GaussianNB

# import some data to play with
iris = datasets.load_iris()
data = iris.data
target = iris.target
X = data[:, 2:]  # we only take the first two features.
y = target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(1, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Petal length')
plt.ylabel('Pepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

gnb = GaussianNB()
fit = gnb.fit(iris.data, iris.target)
prediction = fit.predict(iris.data)
for n, pred in enumerate(prediction):
    if y[n] != pred:
        print("Error in prediction of flower {}: {}".format(n, data[n,]))
        print("-- predicted: {} actual: {}".format(pred, y[n]))

print(iris.target_names[fit.predict([[7.2, 3.1, 4.8, 1.5]])][0])
print(iris.target_names[fit.predict([[3.6, 2.8, 1.8, 0.5]])][0])
print(iris.target_names[fit.predict([[5.5, 3.8, 2.8, 1.2]])][0])
print(iris.target_names[fit.predict([[7.8, 1.9, 5.9, 2.1]])][0])
print(iris.target_names[fit.predict([[18.2, 9.1, 15.4, 5.5]])][0])
print(iris.target_names[fit.predict([[0.5, 0.25, 0.9, 0.3]])][0])
print("*" * 150 + "\n")

training_size = 10
training_sizes = [random.randint(10, 130) for _ in range(30)]
result = []

for training_size in sorted(training_sizes):
    X_train = data[:training_size]  # Change the numeric value for training and testing on each line
    X_test = data[training_size:]
    Y_train = target[:training_size]
    Y_test = target[training_size:]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    score = metrics.accuracy_score(Y_test, y_pred)
    # print("training size:{0:.0f}, Accuracy:{0:.3f}".format(training_size,metrics.accuracy_score(Y_test,y_pred)))
    result.append((training_size, score))

out = sorted(result, key=lambda x: x[1], reverse=True)
best = out[0]
worse = out[-1]
print(best, worse)

training_size = best[0]
X_train = data[:training_size]  # Change the numeric value for training and testing on each line
X_test = data[training_size:]
Y_train = target[:training_size]
Y_test = target[training_size:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

with open("lab8.dot", 'w') as f:
    tree.export_graphviz(clf, out_file=f,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
dot_data = ""
with open("lab8.dot", 'r') as f:
    dot_data = f.read().replace('\n', '')

graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("lab8_tree.pdf")
# (129, 0.9047619047619048) (44, 0.05660377358490566)
print(iris.target_names)

print(clf.predict_proba([[7.2, 3.1, 4.8, 1.5]])[0])
print(clf.predict_proba([[3.6, 2.8, 1.8, 0.5]])[0])
print(clf.predict_proba([[5.5, 3.8, 2.8, 1.2]])[0])
print(clf.predict_proba([[7.8, 1.9, 5.9, 2.1]])[0])
print(clf.predict_proba([[18.2, 9.1, 15.4, 5.5]])[0])
print(clf.predict_proba([[0.5, 0.25, 0.9, 0.3]])[0])