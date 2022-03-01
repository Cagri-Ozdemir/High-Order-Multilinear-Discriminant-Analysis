import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from cosine import tSVDdct,tproddct,ttransx
from scipy.fftpack import dct, idct
def stack_diag(A):
    d0,d1,d2 = A.shape
    B = np.zeros((d2,d0))
    for i in range(d2):
        B[i,:] = A[:,i,i]
    return B
import pickle
data_frame = pickle.load(open("data_frame3.pkl","rb"))
#####################train&test sets
import random
k1=0
k2=32
num_class = 27
train_set_size = 25# change the number based on K_fold. For example first set 25/7, however for the third set 26/6!!!!
test_set_size = 7
Train_set = {}
Test_set = {}
k = 0
kk = 0
clss = 0
K_fold = pickle.load(open("Kfold_s.pkl","rb"))
for i in range(num_class):
    sel = K_fold[0][0]# 5 training and testing sets. For the first set K_fold[0][0]. For the second: K_fold[1][0] and so on
    for ii in range(k1,k2):
        exists = ii in sel
        if exists == False:
            Test_set[k] = data_frame[ii+clss]
            k += 1
        else:
            continue
    for ii in range(k1,k2):
        exists = ii in sel
        if exists == True:
            Train_set[kk] = data_frame[ii+clss]
            kk += 1
        else:
            continue
    clss += 32
######create labels of training & testing sets#####
k = 0
Y_train = np.zeros((len(Train_set)))
for i in range(num_class):
    for ii in range(train_set_size):
        Y_train[k] = i
        k+= 1
k = 0
Y_test = np.zeros((len(Test_set)))
for i in range(num_class):
    for ii in range(test_set_size):
        Y_test[k] = i
        k+= 1
########################
eig = 5
distance = np.zeros((len(Train_set)),dtype="float")
predicted_idx = np.zeros((len(Test_set)))
for jj in range(len(Test_set)):
  for ii in range(len(Train_set)):
   ###test data
   ut,st,vt = tSVDdct(Test_set[jj])
   utt = ut[:,:,:eig]
   ###train data
   u,s,v = tSVDdct(Train_set[ii])
   u0 = u[:,:,:eig]
   uu, ss, vv = tSVDdct(tproddct(ttransx(utt),u0))
   ss1 = dct(ss, type=1,axis=0, norm='ortho')
   ss1[ss1 < 0] = 0
   ss1[ss1 > 1] = 1
   eigentuples = stack_diag(ss1)
   cos_thetas = np.arccos(eigentuples)**2
   d1 = np.linalg.norm(np.sqrt(np.sum(cos_thetas, axis=0)))
   #d1 = np.product(np.sqrt(np.sum(cos_thetas, axis=0)))
   distance[ii] = d1
  predicted_idx[jj] = np.argmin(distance)

get_predicted_idx = Y_train[predicted_idx.astype(int)]
tru_false_predictions = get_predicted_idx ==Y_test
true_count = sum(tru_false_predictions)
print("recognition accuracy",(np.sum(true_count)*100)/len(Test_set))

###plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
cm=confusion_matrix(Y_test, get_predicted_idx)
x_axis_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
y_axis_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
sn.heatmap(cm,annot=True,cbar=False,cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # font size
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()