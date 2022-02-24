import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from fft_fft import tSVD,tinvx,tprod,ttrans,fronorm
def stack_diag(A):
    d0,d1,d2 = A.shape
    B = np.zeros((d2,d0))
    for i in range(d2):
        B[i,:] = A[:,i,i]
    return B
import pickle
class_1 = pickle.load(open("class_1.pkl", "rb"))
class_2 = pickle.load(open("class_2.pkl", "rb"))
class_3 = pickle.load(open("class_3.pkl", "rb"))
class_4 = pickle.load(open("class_4.pkl", "rb"))
class_5 = pickle.load(open("class_5.pkl", "rb"))
class_6 = pickle.load(open("class_6.pkl", "rb"))
class_7 = pickle.load(open("class_7.pkl", "rb"))
class_8 = pickle.load(open("class_8.pkl", "rb"))
class_9 = pickle.load(open("class_9.pkl", "rb"))
Data_frame = [class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9]
#####################train&test sets
import random
k1=0
k2=100
num_class = 9
train_set_size = 80
test_set_size = 20
Train_set = {}
Test_set = {}
k = 0
kk = 0
K_fold = pickle.load(open("Kfold_s.pkl","rb"))
for i in range(num_class):
    #sel = random.sample(range(k1, k2), train_set_size)
    sel = K_fold[1][0]
    for ii in range(k1,k2):
        exists = ii in sel
        if exists == False:
            Test_set[k] = Data_frame[i][ii]
            k += 1
        else:
            continue
    for ii in range(k1,k2):
        exists = ii in sel
        if exists == True:
            Train_set[kk] = Data_frame[i][ii]
            kk += 1
        else:
            continue


######
#subspaces = {}
#k = 0
#for j in range(9):
#    d0,d1,d2 = Train_set[0+k].shape
#    tensor = np.zeros((d0,d1,d2))
#    tensor = Train_set[0+k]
#    for ii in range(train_set_size-1):
#        tensor = np.concatenate((tensor,Train_set[ii+1+k]),axis=2)
#    u,s,v = tSVD(tensor)
#    subspaces[j] = u
#    k+=80

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
distance = np.zeros((len(Train_set)),dtype="float")
predicted_idx = np.zeros((len(Test_set)))
for jj in range(len(Test_set)):
  for ii in range(len(Train_set)):
   ###test data
   ut,st,vt = tSVD(Test_set[jj])
   utt = ut[:,:,:eig]
   ###train data
   u,s,v = tSVD(Train_set[ii])
   u0 = u[:,:,:eig]
   uu, ss, vv = tSVD(tprod(ttrans(utt),u0))
   ss1 = np.fft.fft(ss,axis=0).real
   #coeffs = pywt.dwt(ss, 'haar', axis=0)
   #cA, cD = coeffs
   #ss1 = np.concatenate((cA, cD), axis=0)
   ss1[ss1 < 0] = 0
   ss1[ss1 > 1] = 1
   eigentuples = stack_diag(ss1)
   cos_thetas = np.arccos(eigentuples)**2
   #d1 = np.linalg.norm(np.sqrt(np.sum(cos_thetas, axis=0)))
   d1 = np.product(np.sqrt(np.sum(cos_thetas, axis=0)))
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
x_axis_labels = ["FL","FR","FC","SL","SR","SC","VL","VR","VC"]
y_axis_labels = ["FL","FR","FC","SL","SR","SC","VL","VR","VC"]
sn.heatmap(cm,annot=True,cbar=False,cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # font size
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()