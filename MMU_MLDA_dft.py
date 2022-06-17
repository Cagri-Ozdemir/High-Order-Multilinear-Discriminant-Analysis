import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from sklearn.model_selection import KFold
from FFT import teig4,ttrans4,tinv4,Class_scatters_dftcomp4,pred4,tprod4,updated_Sw_dft

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
# 5-fold cross validation
#x = np.zeros((d3))
#kf = KFold(n_splits=5, shuffle=True)
#kf.get_n_splits(x)
#s1,s2,s3,s4,s5 = kf.split(x)
#K_fold_dic_MMU = [s1,s2,s3,s4,s5]
#import pickle
#a_file = open("K_fold_dic_MMU.pkl", "wb")
#pickle.dump(K_fold_dic_MMU, a_file)
#a_file.close()
import pickle
K_fold_dic = pickle.load(open("k_fold_dic_MMU.pkl","rb"))
##########
accuracy_fft_clm = np.zeros((5))
accuracy_fft_row = np.zeros((5))

eig = 10
for i in range(5):
    s = K_fold_dic[i]
    Train_set = iris[:,:,:,s[0]]
    Test_set = iris[:,:,:,s[1]]
    Ytrn = label[s[0]]
    Ytst = label[s[1]]

    ###################SET1#########
    #######################################################~FOURIER TRANSFORM~capitalize column pixels#########################################
    Sw1, Sb1 = Class_scatters_dftcomp4(45, Train_set, Ytrn)
    S1 = tprod4(tinv4(Sw1), Sb1)
    SS1, UU1 = teig4(S1)
    u1 = UU1[:, :, :, 0:eig]
    pro_df_trn1 = tprod4(ttrans4(u1), Train_set)
    pro_df_tst1 = tprod4(ttrans4(u1), Test_set)
    pre1, accuracy1, pRed1 = pred4(pro_df_trn1, pro_df_tst1, Ytst, Ytrn)
    accuracy_fft_clm[i] = accuracy1
    print("ac1:", accuracy1)
    #######################################################~FOURIER TRANSFORM~capitalize column pixels#########################################
    Sww1, k = updated_Sw_dft(Sw1)
    S1_u = tprod4(tinv4(Sww1), Sb1)
    SS1_u, UU1_u = teig4(S1_u)
    u1_u = UU1_u[:, :, :, 0:eig]
    pro_df_trn1_u = tprod4(ttrans4(u1_u), Train_set)
    pro_df_tst1_u = tprod4(ttrans4(u1_u), Test_set)
    pre1_up, accuracy1_up, pRed1_up = pred4(pro_df_trn1_u, pro_df_tst1_u, Ytst, Ytrn)
    print("ac1_up", accuracy1_up)
    accuracy_fft_row[i] = accuracy1_up

print("dft_row_mean",np.mean(accuracy_fft_row))
print("dft_row_std",np.std(accuracy_fft_row))

print("dft_clm_mean",np.mean(accuracy_fft_clm))
print("dft_clm_std",np.std(accuracy_fft_clm))