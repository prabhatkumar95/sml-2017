import numpy as np
import math
class NB(object):
    def __init__(self):
        self.mean={}
        self.prior={}
        self.std_dev={}
        self.index={}

    def fit(self,X,Y):
        class_sep = {}
        count=0
        for i in range(0, len(X)):
            temp = Y[i]
            # print type(Y[i])
            if (temp not in class_sep.keys()):
                class_sep[Y[i]] =[]
                self.index[Y[i]]=count
                count=count+1

            class_sep[Y[i]].append(X[i])

        for i in class_sep.keys():
            t=np.array(class_sep[i])
            class_sep[i]=t
            # print(class_sep[i].shape)
            self.mean[i]=np.mean(class_sep[i],axis=0)

            self.std_dev[i]=np.std(class_sep[i],axis=0)
            self.std_dev[i]=self.std_dev[i]+0.00001
            self.prior[i]=len(class_sep[i])/float(len(X))
            # print(self.prior[i])
        # print(np.mean(self.mean[3]))
    def predict(self,X):
        Y = np.zeros(len(X))
        Score=np.zeros(len(X)*len(self.index.keys())).reshape(len(self.index.keys()),len(X))
        for t in range(0,len(X)):
            result=0
            max_liklihood=float('-inf')
            # print(self.prior.keys())
            # print(self.mean.keys())
            # print((self.mean[5].shape))
            for i in self.prior.keys():
                log_liklihood = 0.0000
                # print(transform_X.shape)
                for j in range(0,len(X[t])):
                    # print(transform_X[j])
                    log_liklihood+=(-0.5*(((X[t][j]-self.mean[i][j])/float(self.std_dev[i][j]))**2)+math.log(1/(float(math.sqrt(2*math.pi)*(self.std_dev[i][j])))))
                log_liklihood+=math.log(self.prior[i])
                Score[self.index[i]][t]=log_liklihood
                if(log_liklihood>max_liklihood):
                    max_liklihood=log_liklihood
                    result=int(i)

            Y[t]=int(result)
        return Y,Score