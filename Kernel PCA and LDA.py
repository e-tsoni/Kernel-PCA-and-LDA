# -*- coding: utf-8 -*-
"""
@author: Eftychia Tsoni

"""

#making the necessary imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from datetime import datetime
import pandas as pd
np.set_printoptions(precision=4)
import seaborn as sns
sns.set()
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score


"""
Kernel PCA with make_moons dataset

"""


#making the dataset
X, y = make_moons(n_samples = 2000, noise = 0.04, random_state = 1)
color1=(0.69411766529083252, 0.3490196168422699, 0.15686275064945221)
color2=(0.65098041296005249, 0.80784314870834351, 0.89019608497619629)
colormap = np.array([color1,color2])
plt.title("Original data") 
plt.scatter(X[:, 0], X[:, 1], c=colormap[y])
plt.show() 

#Kernel PCA
kpca = KernelPCA(kernel ='rbf', gamma=10) 
X_kpca = kpca.fit_transform(X)  
plt.title("Kernel PCA") 
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c = colormap[y]) 
plt.show() 

#train an SVM after apllying kernelPCA
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_kpca, y, test_size=0.40, random_state=0)
model = SVC(kernel='linear', C=100)
t0 = datetime.now()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))
t1 = datetime.now()
print("Time = ",t1-t0)
print("\n")


"""
LDA with iris dataset

"""


#Importing the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)

#divide data into labels and feature set
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# divide data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Performing LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#Training and Making Predictions
classifier = SVC(kernel='linear', C=10, gamma='auto')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Evaluating the Performance
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))

