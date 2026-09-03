from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
print(iris.data.shape)
X = iris.data[:, [2, 3]]
print(X)

y = iris.target
print('Class labels', np.unique(y))
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Label counts in y:', np.bincount(y))
print('Lbelel count in y_train', np.bincount(y_train))
print('label count in y_test', np.bincount(y_test))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

print(X_train[0:5, :])

sc.fit(X_train)

print(sc.mean_)
print(sc.scale_)

X_train_std = sc.transform(X_train)
print("after transform")
print(X_train_std[0:5, :])
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_descision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],marker=markers[idx], label=f'Class {cl}', edgecolors='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black', alpha=1.0, linewidths=1, marker='o', s=100, label='Test set')
        

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_std = np.hstack((y_train, y_test))

plot_descision_regions(X=X_combined_std, y=y_combined_std, classifier=ppn, test_idx=range(105, 150))

plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
