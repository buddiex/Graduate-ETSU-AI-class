import sys

import matplotlib.pyplot as plt
import numpy as np
from project5_perceptron import Perceptron


def perceptron_learning(X, y):
    ### Plot specifically for the iris data set

    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    #
    # plt.xlabel('sepal length [cm]')
    # plt.ylabel('sepal width [cm]')
    # plt.legend(loc='upper left')
    # plt.show()
    #
    # plt.scatter(X[:50, 2], X[:50, 3], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 2], X[50:100, 3], color='blue', marker='x', label='versicolor')
    #
    # plt.xlabel('petal length [cm]')
    # plt.ylabel('petal width [cm]')
    # plt.legend(loc='upper left')
    # plt.show()

    ### Plot for the 3-d data set
    # figure = plt.figure()
    # dataplot = figure.add_subplot(111,projection='3d')
    #
    # for i in range(0, len(X)):
    #     theX = X[i,0]
    #     theY = X[i,1]
    #     theZ = X[i,2]s
    #     color = ''
    #     if y[i] == -1:
    #         color = 'r'
    #     else:
    #         color = 'b'
    #     dataplot.scatter(theX, theY, theZ, marker='s', c=color)
    # plt.show()

    ### Run the perceptron learning
    ppn = Perceptron(n_inputs=len(y), learning_rate=0.01, iterations=10)

    ppn.train(X, y)

    ### Plot the errors across epochs
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Misclassifications')
    plt.show()

    ### Print the prediction for a given vector
    # # Iris Data
    # print(ppn.activate([[7.2, 3.1, 4.8, 1.5]]))
    print(ppn.activate([3.6, 2.8, 1.8, 0.5]))
    print(ppn.activate([5.5, 3.8, 2.8, 1.2]))
    print(ppn.activate([7.8, 1.9, 5.9, 2.1]))
    print(ppn.activate([18.2, 9.1, 15.4, 5.5]))
    print(ppn.activate([0.5, 0.25, 0.9, 0.3]))
    # 3D Data
    # print(ppn.activate([0,0,0]))
    # print(ppn.activate([60,60,60]))
    # print(ppn.activate([100,100,100]))


def main():
    filenameX = ""
    filenameY = ""
    if len(sys.argv) == 3:
        filenameX = sys.argv[1]
        filenameY = sys.argv[2]
    else:
        sys.exit(-1)

    X = np.load(filenameX)
    y = np.load(filenameY)

    # Convert 0's in the result data to -1
    for x1 in range(len(y)):
        if y[x1] == 0:
            y[x1] = -1

    perceptron_learning(X, y)


main()
