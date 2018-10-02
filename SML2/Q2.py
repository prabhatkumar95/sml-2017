import numpy as np
import math


def discriminant(m, cov, prior, x):
    mean = np.array(m)
    # cov = np.array([[c]])
    # print(x.shape,cov.shape)
    term1 = 0.5 * np.matmul(np.matmul((x - mean), np.linalg.inv(cov)), (x - mean))
    term2 = 0.5 * np.log(np.linalg.det(cov))
    term3 = math.log(prior)
    return term3 - term1 - term2


def b_bound(m,cov,prior,feature):
    grid = np.ix_(feature, feature)
    cov0 = cov[0][grid]
    cov1 = cov[1][grid]
    mean0 = [m[0][i] for i in feature]
    mean1 = [m[1][i] for i in feature]

    mean_diff = np.subtract(mean1,mean0)
    std_sum_mean = (cov0 + cov1) / float(2)
    term1 = 0.125*np.matmul(np.matmul(mean_diff,np.linalg.inv(std_sum_mean)),mean_diff.T)
    term2 = 0.5*np.log(np.linalg.det(std_sum_mean)/float(math.sqrt(np.linalg.det(cov1)*np.linalg.det(cov0))))

    k_final = term1+term2
    # print(mean_diff,std_sum_mean)
    # print(-(0.125*math.pow(mean_diff,2)*1/float(std_sum_mean)))
    # print(0.5*math.log(abs((std_sum_mean))/float(math.sqrt(abs(var_1*var_2)))))
    # # print(math.pow(math.e,-0.125*math.pow(mean_diff,2)*1/float(std_sum_mean)+0.5*math.log(abs((std_sum_mean))/float(math.sqrt(abs(var_1*var_2))),base=math.e)))
    return math.sqrt(prior[0] * prior[1]) * math.pow(math.e,-k_final)


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
cov[0].shape
print(cov)

grid=np.ix_([0,1], [0,1])
cov[0][grid]
prior = [0.5,0.5,0]

def decision(m,c,prior,data,feature):
    grid=np.ix_(feature,feature)
    cov0=c[0][grid]
    cov1=c[1][grid]
    mean0=[m[0][i] for i in feature]
    mean1=[m[1][i] for i in feature]
    print(mean0,mean1)
    result0 = [0 if discriminant(mean0,cov0,prior[0],d[0][i][0:3])>discriminant(mean1,cov1,prior[1],data[0][i][0:3]) else 1 for i in range(0,10)]
    result1= [1 if discriminant(mean0,cov0,prior[0],d[1][i][0:3])<discriminant(mean1,cov1,prior[1],data[1][i][0:3]) else 0 for i in range(0,10)]
    result=result0+result1
    return result



def score(result,label):
    count=0
    for i in range(0,len(result)):
        if(result[i]==label[i]):
            count=count+1
    return count/float(len(result))




result=decision(mean,cov,prior,d,[0,1,2])
print(mean)

error=1-score(result,y[:20])
print(error)

print(b_bound(mean,cov,prior,[0]))

