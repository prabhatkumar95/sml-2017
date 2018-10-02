import numpy as np
from scipy.io import savemat
class PCA(object):
    def __init__(self):
        self.eigen_vect=[]
        self.eigen_val=[]

    def calc_vector_space(self,X):
        mean = np.mean(X,axis=0)
        X_centered = X-mean
        cov = np.matmul(X_centered.T,X_centered)
        cov = cov/float(len(X))
        # savemat('cov.mat',mdict={'arr':cov})
        e_val,e_vect = np.linalg.eigh(cov)
        print(e_vect.shape)
        for i in np.argsort(-e_val):
            self.eigen_val.append(e_val[i])
            self.eigen_vect.append(e_vect[:,i])

    def get_projection_vector(self,eigen_energy):
        total = np.sum(self.eigen_val)
        print(total)
        if (total==0):
            return None
        for i in range(1,len(self.eigen_val)):
            num = np.sum(self.eigen_val[:-i])
            # print(num)
            if((num/float(total))<eigen_energy):
                print(i)
                return self.eigen_vect[:-i+1]
        return None


