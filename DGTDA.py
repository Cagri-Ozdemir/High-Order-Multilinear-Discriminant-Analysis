import numpy as np
from tensorly import unfold
#arr = np.random.rand(5,10,2)
#x1 = unfold(arr, 1)
#sh = arr.shape
#AR = fold(x1,1,sh)

def Class_scatters_dgtda(Tensor_train,y_train):
    n0,n1,n2,n3 = Tensor_train.shape
    num_class = int(np.amax(y_train))
    mean_tensor_train = np.zeros((n0,n1,n2,num_class))
    Sw0 = 0
    Sw1 = 0
    Sw2 = 0
    Mean_tensor = np.zeros((n0,n1, n2, 1))
    Mean_tensor[:,:,:,0] = (Tensor_train.sum(axis=3))/n3
    for i in range(num_class):
      Sa0 = 0
      Sa1 = 0
      Sa2 = 0
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,:,i] = (Tensor_train[:,:,:,idx].sum(axis=3))/occurrences
      for j in idx:
          a = Tensor_train[:,:,:,j]-mean_tensor_train[:,:,:,i]
          Sa0 = Sa0 + (unfold(a,0)@unfold(a,0).T)
          Sa1 = Sa1 + (unfold(a, 1) @ unfold(a, 1).T)
          Sa2 = Sa2 + (unfold(a, 2) @ unfold(a, 2).T)
      Sw0 = Sw0 + Sa0
      Sw1 = Sw1 + Sa1
      Sw2 = Sw2 + Sa2
    Sw = [Sw0,Sw1,Sw2]
    Sb0 = 0
    Sb1 = 0
    Sb2 = 0
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b = mean_tensor_train[:,:,:,i] - Mean_tensor[:,:,:,0]
        Sb0 = ((unfold(b,0)@(unfold(b,0)).T)*occurrences) + Sb0
        Sb1 = ((unfold(b,1)@(unfold(b,1)).T)*occurrences) + Sb1
        Sb2 = ((unfold(b,2)@(unfold(b,2)).T)*occurrences) + Sb2
    Sb = [Sb0, Sb1, Sb2]
    return Sw,Sb

