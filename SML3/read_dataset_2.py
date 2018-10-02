import numpy as np
import os
import matplotlib as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from pca import PCA
from lda import LDA


def projectmatrix(X,p):
    result=[]
    for i in range(len(X)):
        result.append(np.matmul(X[i],p))
    return np.array(result)

#CIFAR Reference
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

t= unpickle("CIFAR/cifar-10-batches-py/data_batch_1")
data=t[b'data']
label = list(t[b'labels'])

for i in range(2,6):
    t= unpickle("CIFAR/cifar-10-batches-py/data_batch_"+str(i))
    temp= t[b'data']
    templ=t[b'labels']
    data = np.concatenate((data,temp),axis=0)
    label= label+list(templ)

t=unpickle("CIFAR/cifar-10-batches-py/test_batch")
temp = t[b'data']
templ = t[b'labels']
data = np.concatenate((data, temp), axis=0)
label = label + list(templ)

print(len(label))
print(data.shape)
result=[]
for fold in (0,5):
    print(fold)
    f=open('dataset2_lda.txt',mode='a')
    trainx,testx,trainy,testy=train_test_split(data,label,shuffle=True,test_size=0.1666)
    #
    # pca = PCA()
    # pca.calc_vector_space(trainx)
    # project_pca = pca.get_projection_vector(0.9)
    # project_pca=np.array(project_pca)
    # proj_trainx = np.matmul(project_pca,trainx.T).T
    # proj_testx = np.matmul(project_pca,testx.T).T
    #



    #
    lda = LDA()
    lda.calculate(trainx,trainy)
    project = lda.project()
    # print(np.linalg.matrix_rank(project))
    #
    # print(proj_trainx.shape)
    # print(np.linalg.matrix_rank(project))

    eigv,eigvect = np.linalg.eigh(project)
    eigvect= eigvect[:,-9:]
    # eigv,eigvect = eigsh(project,k=10)
    # print(np.argsort(eigv))
    # print(eigv[-10:])
    #
    #
    import matplotlib.pyplot as plt
    from skimage.util import invert
    # plot Images
    # for i in range(0,10):
    #     t = eigvect[:,i]
    #     plt.imshow(img_as_float(t.reshape(50,50)))
    #     plt.colorbar()
    #     plt.savefig("plots/plt_"+str(i+1)+"_green.png")
    #     plt.clf()
    #     # plt.show()

    #classifier
    from sklearn.linear_model import LogisticRegression
    # from sklearn.svm import LinearSVC


    # trans_trainx = np.matmul(trainx,eigvect)
    trans_trainx = projectmatrix(trainx,eigvect)
    m = LogisticRegression()
    # m = LinearSVC()
    # m.fit(trans_trainx,trainy)
    m.fit(trans_trainx,trainy)
    # m_t.fit(trans_trainx,trainy)
    # print(m.score(trans_testx,testy))
    # print(m.score(trans_testx,testy))
# print(m_t.score(trans_testx,testy))
    del trans_trainx
    trans_testx = np.matmul(testx, eigvect)

    score=m.score(trans_testx,testy)
    result.append(score)
    # print(m_t.score(trans_testx,testy))
    #
    f.write(str(fold+1)+" : "+str(score)+"\n")
    f.close()
    del lda
    # del trans_trainx
    del trans_testx
    #
f=open('dataset2_lda.txt',mode='a')
f.write("Mean : "+str(np.mean(result))+"\n")
f.write("Std : "+str(np.std(result))+"\n")
f.close()