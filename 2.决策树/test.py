# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:38:05 2018

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np

import time
import logging
from sklearn.datasets import fetch_olivetti_faces
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

data_home='datasets/'

faces = fetch_olivetti_faces(data_home=data_home)

X = faces.data
y = faces.target
targets = np.unique(faces.target)
target_names = np.array(["c%d" % t for t in targets])
n_targets = target_names.shape[0]
n_samples, h, w = faces.images.shape
print('Sample count: {}\nTarget count: {}'.format(n_samples, n_targets))
print('Image size: {}x{}\nDataset shape: {}\n'.format(w, h, X.shape))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)

from sklearn.svm import SVC

start = time.clock()
print('Fitting train datasets ...')
clf = SVC(class_weight='balanced')
clf.fit(X_train, y_train)
print('Done in {0:.2f}s'.format(time.clock()-start))

start = time.clock()
print("Predicting test dataset ...")
y_pred = clf.predict(X_test)
print('Done in {0:.2f}s'.format(time.clock()-start))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print("confusion matrix:\n")
np.set_printoptions(threshold=np.nan)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.decomposition import PCA

print("Exploring explained variance ratio for dataset ...")
candidate_components = range(10, 110, 10)
explained_ratios = []
start = time.clock()
for c in candidate_components:
    pca = PCA(n_components=c)
    X_pca = pca.fit_transform(X)
    explained_ratios.append(np.sum(pca.explained_variance_ratio_))
print('Done in {0:.2f}s'.format(time.clock()-start))

plt.figure(figsize=(10, 6), dpi=80)
plt.grid()
plt.plot(candidate_components, explained_ratios)
plt.xlabel('弱分类器数量', fontproperties=font)
plt.ylabel('AD-BP准确率', fontproperties=font)
plt.title('准确率和弱分类器数量关系', fontproperties=font)
plt.yticks(np.arange(0.5, 1.05, .05))
plt.xticks(np.arange(0, 110, 10));
plt.xlim (10, 100)
