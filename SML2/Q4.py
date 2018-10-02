import numpy as np
import math






data = np.loadtxt('book_data.txt',delimiter=" ")
y=data[:,-1]
d=np.zeros(3*10*3).reshape(3,10,3)
d[0]=data[:10,:-1]
d[1]=data[10:20,:-1]
d[2]=data[20:30,:-1]
mean=np.zeros(3*3).reshape(3,3)
mean[0]=np.mean(d[0],axis=0)
mean[1]=np.mean(d[1],axis=0)
mean[2]=np.mean(d[2],axis=0)

cov=np.zeros(3*3*3).reshape(3,3,3)
cov[0]=np.cov(d[0].T)
cov[1]=np.cov(d[1].T)
cov[2]=np.cov(d[2].T)

prior=[0.8,0.1,0.1]


data=[[1,2,1],[5,3,2],[0,0,0],[1,0,0]]
data=np.array(data)
print(data)

def m_distance(mean,cov,data):
    dist = np.zeros(3*data.shape[0]).reshape(3,data.shape[0])
    for i in range(0,3):
        m=np.subtract(data,mean[i])
        c=cov[i]
        dist[i]=np.diag(np.matmul(np.matmul(m,np.linalg.inv(c)),m.T))
    return np.sqrt(dist)

def discriminant(m, cov, prior, data):
    discr = np.zeros(3 * data.shape[0]).reshape(3, data.shape[0])
    for i in range(0, 3):
        c=cov[i]
        # cov = np.array([[c]])
        # print(x.shape,cov.shape)
        sub=np.subtract(data,m[i])
        term1 = 0.5 * np.matmul(np.matmul(sub, np.linalg.inv(c)),sub.T)
        term2 = 0.5 * np.log(np.linalg.det(c))
        term3 = math.log(prior[i])
        discr[i]=np.diag(term3 - term1 - term2)
    return discr


dist=m_distance(mean,cov,data)
print(dist)
print(np.argmin(dist,axis=0))

dist=discriminant(mean,cov,prior,data)
print(np.argmax(dist,axis=0))

