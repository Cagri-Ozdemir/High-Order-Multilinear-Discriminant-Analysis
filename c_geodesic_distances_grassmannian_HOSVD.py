import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from tensorly.decomposition import tucker
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
distance2 = np.zeros((len(Train_set)),dtype="float")
predicted_idx2 = np.zeros((len(Test_set)))
for jj in range(len(Test_set)):
  for ii in range(len(Train_set)):
   d0,d1,d2 = Test_set[jj].shape
   ###test data
   test_core,test_factors = tucker(Test_set[jj],rank=[eig,eig,d2])
   ###train data
   d00, d12, d22 = Train_set[ii].shape
   train_core,train_factors = tucker(Train_set[ii],rank=[eig,eig,d22])

   uu0, ss0, vv0 = np.linalg.svd(train_factors[0].T@test_factors[0])
   ss0[ss0 < 0] = 0
   ss0[ss0 > 1] = 1
   uu1, ss1, vv1 = np.linalg.svd(train_factors[1].T@test_factors[1])
   ss1[ss1 < 0] = 0
   ss1[ss1 > 1] = 1
   cos_thetas0 = np.linalg.norm(np.arccos(ss0))
   cos_thetas1 = np.linalg.norm(np.arccos(ss1))
   d1 = cos_thetas0*cos_thetas1
   distance2[ii] = d1
  predicted_idx2[jj] = np.argmin(distance2)
get_predicted_idx2 = Y_train[predicted_idx2.astype(int)]
tru_false_predictions2 = get_predicted_idx2 == Y_test
true_count2 = sum(tru_false_predictions2)
print("recognition accuracy_HOSVD", (np.sum(true_count2) * 100) / len(Test_set))

###plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
cm=confusion_matrix(Y_test, get_predicted_idx2)
x_axis_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
y_axis_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
sn.heatmap(cm,annot=True,cbar=False,cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # font size
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
