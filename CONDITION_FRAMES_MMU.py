import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from FFT import Class_scatters_dftcomp4,condition_dft
from DCT import Class_scatters_dctcomp4,condition_dct
from DWT import Class_scatters_dwtcomp4,condition_dwt
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

eig = 10
for i in range(1):
    s = K_fold_dic[1]
    Train_set = iris[:,:,:,s[0]]
    Test_set = iris[:,:,:,s[1]]
    Ytrn = label[s[0]]
    Ytst = label[s[1]]

    ###################SET1#########
    #######################################################~FOURIER TRANSFORM~capitalize column pixels#########################################
    Sw1, Sb1 = Class_scatters_dftcomp4(45, Train_set, Ytrn)
    Sw2, Sb2 = Class_scatters_dctcomp4(45, Train_set, Ytrn)
    Sw3, Sb3 = Class_scatters_dwtcomp4(45, Train_set, Ytrn)
    ########
    cond_fft = condition_dft(Sw1)
    cond_dct = condition_dct(Sw2)
    cond_dwt = condition_dwt(Sw3)
    ########

th = np.log(np.ones((192))*1.e+5)
x = np.arange(0,192)
f,ax = plt.subplots(1,3,figsize=(16, 4.5))
ax[0].plot(x,np.log(cond_fft),"go",ms=4)
ax[0].plot(th,"k--")
ax[0].set_title("$\mathcal{W}$ (DFT)", fontsize='large')
ax[0].set_ylim([2,14])
#ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0),useLocale=None, useMathText=True)

ax[1].plot(x,np.log(cond_dct),"rs",ms=4)
ax[1].set_yticks([])
ax[1].plot(th,"k--")
ax[1].set_title("$\mathcal{W}$ (DCT)", fontsize='large')
ax[1].set_ylim([2,14])
#ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0),useLocale=None, useMathText=True)

ax[2].plot(x,np.log(cond_dwt),"^b",ms=4)
ax[2].set_yticks([])
ax[2].plot(th,"k--")
ax[2].set_title("$\mathcal{W}$ (DWT)", fontsize='large')
ax[2].set_ylim([2,14])
#ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0),useLocale=None, useMathText=True)
f.supxlabel('Frontal slices')
f.supylabel('$log_{10}$(condition number)')
f.suptitle("The MMU Iris Database")

