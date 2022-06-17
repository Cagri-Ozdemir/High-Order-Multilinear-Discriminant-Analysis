import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from sklearn.model_selection import KFold
from FFT import pred4
from DCT import teigdct4,ttransdct4,tinvdct4,Class_scatters_dctcomp4,tproddct4,updated_Sw

import cv2
irisdf = scipy.io.loadmat('IRIS.mat')
iris = irisdf['AD']
label = np.zeros((225))
d0,d1,d2,d3 = iris.shape
k = 0
for i in range(45):
    for ii in range(5):
        label[k] = i+1
        k+=1

import pickle
K_fold_dic = pickle.load(open("k_fold_dic_MMU.pkl","rb"))
##########
accuracy_dct_clm = np.zeros((5))
accuracy_dct_row = np.zeros((5))

eig = 10
for i in range(5):
    s = K_fold_dic[i]
    Train_set = iris[:,:,:,s[0]]
    Test_set = iris[:,:,:,s[1]]
    Ytrn = label[s[0]]
    Ytst = label[s[1]]

    ###################SET1#########
    #######################################################~FOURIER TRANSFORM~capitalize column pixels#########################################
    Sw1, Sb1 = Class_scatters_dctcomp4(45, Train_set, Ytrn)
    S1 = tproddct4(tinvdct4(Sw1), Sb1)
    SS1, UU1 = teigdct4(S1)
    u1 = UU1[:, :, :, 0:eig]
    pro_df_trn1 = tproddct4(ttransdct4(u1), Train_set)
    pro_df_tst1 = tproddct4(ttransdct4(u1), Test_set)
    pre1, accuracy1, pRed1 = pred4(pro_df_trn1, pro_df_tst1, Ytst, Ytrn)
    accuracy_dct_clm[i] = accuracy1
    print("ac1:", accuracy1)
    ############
    Sww1, k = updated_Sw(Sw1)
    S1_u = tproddct4(tinvdct4(Sww1), Sb1)
    # S1_u = updated_Sw(S1_u)
    SS1_u, UU1_u = teigdct4(S1_u)
    u1_u = UU1_u[:, :, :, 0:eig]
    pro_df_trn1_u = tproddct4(ttransdct4(u1_u), Train_set)
    pro_df_tst1_u = tproddct4(ttransdct4(u1_u), Test_set)
    pre1_up, accuracy1_up, pRed1_up = pred4(pro_df_trn1_u, pro_df_tst1_u, Ytst, Ytrn)
    accuracy_dct_row[i] = accuracy1_up
    print("ac1_up", accuracy1_up)

