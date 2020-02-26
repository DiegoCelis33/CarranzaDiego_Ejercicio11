#!/usr/bin/env python
# coding: utf-8

# In[52]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


# In[72]:


numeros = skdata.load_digits()

imagenes = numeros['images']

target = numeros['target']
n_imagenes = len(target)
target[target!=1]=0



# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))



# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#numero = 1
#dd = y_train==numero
#nn = y_train!=numero

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]



clf = LinearDiscriminantAnalysis()

y_prediccion = []
f_1_score = []

y_prediccion_test = []
f_1_score_test = []
number_components = []

y_pred = 0
f_1 = 0

for i in range(3,40):
    
    
    clf.fit(x_train[:,:i], y_train)
    
    y_pred = clf.predict(x_train[:,:i])
    y_pred_test = clf.predict(x_test[:,:i])
    
    f_1 = f1_score(y_train, y_pred)
    
    f_1_test = f1_score(y_test, y_pred_test)
    
    
    y_prediccion.append(y_pred)
    f_1_score.append(f_1)
    
    y_prediccion_test.append(y_pred_test)
    f_1_score_test.append(f_1_test)
    
    number_components.append(i)
    
y_prediccion_o = []
f_1_score_o = []

y_prediccion_test_o = []
f_1_score_test_o = []
number_components_o = []
    
for i in range(3,40):
    
    
    clf.fit(x_train[:,:i], y_train)
    
    y_pred = clf.predict(x_train[:,:i])
    y_pred_test = clf.predict(x_test[:,:i])
    
    f_1 = f1_score(y_train, y_pred, pos_label = 0)
    
    f_1_test = f1_score(y_test, y_pred_test, pos_label = 0)
    
    
    y_prediccion_o.append(y_pred)
    f_1_score_o.append(f_1)
    
    y_prediccion_test_o.append(y_pred_test)
    f_1_score_test_o.append(f_1_test)
    
    number_components_o.append(i)
    
plt.figure()
plt.subplot(1,2,1)
plt.scatter(number_components,f_1_score)
plt.scatter(number_components,f_1_score_test)
plt.ylabel("f_1")
plt.xlabel("number")

plt.subplot(1,2,2)
plt.scatter(number_components_o,f_1_score_o)
plt.scatter(number_components_o,f_1_score_test_o)
plt.ylabel("f_1")
plt.xlabel("number")

plt.tight_layout()

plt.savefig("F1_score_LinearDiscriminantAnalysis.png")

    
    

