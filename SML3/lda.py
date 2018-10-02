import numpy as np

class LDA(object):
    def __init__(self):
        self.sw=None
        self.sb=None
        self.class_mean={}
        self.class_no={}

    def calculate(self,X,Y):
        class_sep = {}
        for i in range(0,len(X)):
            if Y[i] not in class_sep.keys():
                class_sep[Y[i]]=[]

            class_sep[Y[i]].append(X[i])

        for i in class_sep.keys():
            class_sep[i]=np.array(class_sep[i])
            self.class_no[i]=class_sep[i].shape[0]
            self.class_mean[i]=np.mean(class_sep[i],axis=0)

        self.sw = np.zeros(X.shape[1]*X.shape[1]).reshape(X.shape[1],X.shape[1])
        self.sb = np.zeros(X.shape[1] * X.shape[1]).reshape(X.shape[1], X.shape[1])

        for i in class_sep.keys():
            mean = self.class_mean[i]
            # print(mean.shape)
            X_centered = class_sep[i] - mean
            # print(X_centered.shape)
            cov = np.matmul(X_centered.T, X_centered)
            # cov = np.cov(class_sep[i].T,bias=True)
            self.sw = self.sw+cov
            # print("SW",np.linalg.matrix_rank(self.sw))


        mean  = np.mean(X,axis=0)
        # print(mean.shape)
        # t = []
        for i in class_sep.keys():
            temp = self.class_mean[i]-mean
            temp=temp.reshape(temp.shape[0],1)
            # print(1,temp.shape)
            temp = np.matmul(temp,temp.T)
            # print(2,temp.shape)
            temp = self.class_no[i] * temp
            # print(3,temp.shape)
            self.sb = self.sb + temp
            # t.append(self.class_mean[i])
            # print("SB", np.linalg.matrix_rank(self.sb))
        # t=np.array(t)
        # print(t.shape)
        # self.sb=np.cov(t.T,bias=True)

    def project(self):
        return np.matmul(np.linalg.inv(self.sw),self.sb)
        # return np.matmul(np.linalg.inv(self.sw),self.sb)







