import numpy as np
from tensorly import unfold,tenalg
def Class_scatters_cmda(Tensor_train,y_train,i0,i1,i2,Tmax):
  n0,n1,n2,n3 = Tensor_train.shape
  num_class = int(np.amax(y_train))
  mean_tensor_train = np.zeros((n0,n1,n2,num_class))
  Mean_tensor = np.zeros((n0,n1, n2, 1))
  Mean_tensor[:,:,:,0] = (Tensor_train.sum(axis=3))/n3
  u0 = np.ones((n0,i0))
  u1 = np.ones((n1,i1))
  u2 = np.ones((n2,i2))
  err = np.zeros((Tmax))
  tol = 10**-6
  for ii in range(Tmax):
     Sw0 = 0
     Sw1 = 0
     Sw2 = 0
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
           sa0 = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u1,1,transpose=True),u2,2,transpose=True),0)
           sa0t = (unfold(tenalg.mode_dot(tenalg.mode_dot(a,u1,1,transpose=True),u2,2,transpose=True),0)).T
           Sa0 = (sa0@sa0t) + Sa0
           sa1 = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u2,2,transpose=True),1)
           sa1t = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u2,2,transpose=True),1).T
           Sa1 = (sa1@sa1t) + Sa1
           sa2 = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u1,1,transpose=True),2)
           sa2t = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u1,1,transpose=True),2).T
           Sa2 = (sa2@sa2t) + Sa2
       Sw0 = Sw0 + Sa0
       Sw1 = Sw1 + Sa1
       Sw2 = Sw2 + Sa2
     Sb0 = 0
     Sb1 = 0
     Sb2 = 0
     for i in range(num_class):
         occurrences = np.count_nonzero(y_train == i + 1)
         b = mean_tensor_train[:,:,:,i] - Mean_tensor[:,:,:,0]
         sb0 = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u1, 1, transpose=True), u2, 2, transpose=True), 0)
         sb0t = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u1, 1, transpose=True), u2, 2, transpose=True), 0).T
         Sb0 = (sb0 @ sb0t) * occurrences + Sb0
         sb1 = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u2, 2, transpose=True), 1)
         sb1t = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u2, 2, transpose=True), 1).T
         Sb1 = (sb1 @ sb1t) * occurrences + Sb1
         sb2 = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u1, 1, transpose=True), 2)
         sb2t = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u1, 1, transpose=True), 2).T
         Sb2 = (sb2 @ sb2t) * occurrences + Sb2
     U0, S0, V0 = np.linalg.svd(np.linalg.inv(Sw0) @ Sb0)
     U1, S1, V1 = np.linalg.svd(np.linalg.inv(Sw1) @ Sb1)
     U2, S2, V2 = np.linalg.svd(np.linalg.inv(Sw2) @ Sb2)

     err0 = np.linalg.norm(U0[:,0:i0] @ u0.T - np.eye(n0))
     err1 = np.linalg.norm(U1[:, 0:i1] @ u1.T - np.eye(n1))
     err2 = np.linalg.norm(U2[:, 0:i2] @ u2.T - np.eye(n2))
     #err0 = np.linalg.norm((U0 - u0)/ u0)**2
     #err1 = np.linalg.norm((U1[:, 0:i1] - u1)/u1)
     #err2 = np.linalg.norm((U2[:, 0:i2] - u2) / u2)
     err[ii] = err0 + err1 + err2
     if err[ii] <= tol:
         break
     u0 = U0[:, 0:i0]
     u1 = U1[:, 0:i1]
     u2 = U2[:, 0:i2]
  return u0,u1,u2


def Class_scatters_cmda2(Tensor_train,y_train,i0,i1,i2,Tmax):
  n0,n1,n2,n3 = Tensor_train.shape
  num_class = int(np.amax(y_train))
  mean_tensor_train = np.zeros((n0,n1,n2,num_class))
  Mean_tensor = np.zeros((n0,n1, n2, 1))
  Mean_tensor[:,:,:,0] = (Tensor_train.sum(axis=3))/n3
  u0 = np.eye(n0,i0)
  u1 = np.eye(n1,i1)
  u2 = np.eye(n2,i2)
  err = np.zeros((Tmax))
  tol = 10**-6
  for ii in range(Tmax):
     Sw0 = 0
     Sw1 = 0
     Sw2 = 0
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
           sa0 = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u1,1,transpose=True),u2,2,transpose=True),0)
           sa0t = (unfold(tenalg.mode_dot(tenalg.mode_dot(a,u1,1,transpose=True),u2,2,transpose=True),0)).T
           Sa0 = (sa0@sa0t) + Sa0
           sa1 = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u2,2,transpose=True),1)
           sa1t = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u2,2,transpose=True),1).T
           Sa1 = (sa1@sa1t) + Sa1
           sa2 = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u1,1,transpose=True),2)
           sa2t = unfold(tenalg.mode_dot(tenalg.mode_dot(a,u0,0,transpose=True),u1,1,transpose=True),2).T
           Sa2 = (sa2@sa2t) + Sa2
       Sw0 = Sw0 + Sa0
       Sw1 = Sw1 + Sa1
       Sw2 = Sw2 + Sa2
     Sb0 = 0
     Sb1 = 0
     Sb2 = 0
     for i in range(num_class):
         occurrences = np.count_nonzero(y_train == i + 1)
         b = mean_tensor_train[:,:,:,i] - Mean_tensor[:,:,:,0]
         sb0 = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u1, 1, transpose=True), u2, 2, transpose=True), 0)
         sb0t = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u1, 1, transpose=True), u2, 2, transpose=True), 0).T
         Sb0 = (sb0 @ sb0t) * occurrences + Sb0
         sb1 = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u2, 2, transpose=True), 1)
         sb1t = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u2, 2, transpose=True), 1).T
         Sb1 = (sb1 @ sb1t) * occurrences + Sb1
         sb2 = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u1, 1, transpose=True), 2)
         sb2t = unfold(tenalg.mode_dot(tenalg.mode_dot(b, u0, 0, transpose=True), u1, 1, transpose=True), 2).T
         Sb2 = (sb2 @ sb2t) * occurrences + Sb2
     S0, U0 = np.linalg.eig(np.linalg.inv(Sw0) @ Sb0)
     a0,a1 = U0.shape
     idx = np.argsort(S0)
     idx = idx[::-1][:a0]
     S0 = S0[idx].real
     U0 = U0[:, idx].real
     S1, U1 = np.linalg.eig(np.linalg.inv(Sw1) @ Sb1)
     a0, a1 = U1.shape
     idx = np.argsort(S1)
     idx = idx[::-1][:a0]
     S1 = S1[idx].real
     U1 = U1[:, idx].real
     S2, U2 = np.linalg.eig(np.linalg.inv(Sw2) @ Sb2)
     a0, a1 = U2.shape
     idx = np.argsort(S2)
     idx = idx[::-1][:a0]
     S2 = S2[idx].real
     U2 = U2[:, idx].real

     err0 = np.linalg.norm((U0[:,0:i0] - u0))**2
     err1 = np.linalg.norm((U1[:, 0:i1] - u1))**2
     err2 = np.linalg.norm((U2[:, 0:i2] - u2))**2
     err[ii] = err0 + err1 + err2
     if err[ii] <= tol:
         break
     u0 = U0[:, 0:i0]
     u1 = U1[:, 0:i1]
     u2 = U2[:, 0:i2]
  return U0,U1,U2
