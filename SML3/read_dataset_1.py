import os
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.io import imread
from skimage.transform import resize
from lda import LDA
from pca import PCA
from scipy.spatial.distance import euclidean,cosine


def createrocscore(X,Y):
    label=[]
    score=[]
    for i in range(0,len(X)):
        for j in range(i+1,len(X)):
            if(Y[i]==Y[j]):
                label.append(1)
            else:
                label.append(0)
            score.append((cosine(X[i],X[j])+1)/2)
    return label,score

image_list=[]

yourpath = os.getcwd()+"/Face_data"
for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        if ".pgm" in name:
            temp= os.path.join(root,name)
            # print(temp)
            image_list.append(temp)
    # for name in dirs:
        # print(os.path.join(root, name))
print (len(image_list))
y_data = []
data=[]
image_matrix=[]


for i in range(0,len(image_list)):
    p=os.path.dirname(image_list[i])
    p=int(p.split('\\')[-1])
    temp=imread(image_list[i],True)
    # print(temp.shape)
    temp = resize(temp,(25,25))
    temp=temp.flatten()
    y_data.append(p)
    # print(temp.shape)
    data.append(temp)
# print (y_data)
y_data=np.array(y_data)
data=np.array(data)


result=[]
mean=0


trainx,testx,trainy,testy=train_test_split(data,y_data,shuffle=True,test_size=0.3)
#
lda = LDA()
lda.calculate(trainx,trainy)
project = lda.project()
#
eigv,eigvect = np.linalg.eigh(project)
eigvect= eigvect[:,-10:]
#
trans_trainx = np.matmul(trainx, eigvect)
trans_testx = np.matmul(testx, eigvect)
#
# pca = PCA()
# pca.calc_vector_space(trainx)
# project_pca = pca.get_projection_vector(0.99)
# project_pca = np.array(project_pca)
# trans_trainx = np.matmul(project_pca, trainx.T).T
# trans_testx = np.matmul(project_pca, testx.T).T
# print(trans_testx.shape)
l,s,=createrocscore(trans_testx,testy)

fpr,tpr,_ = roc_curve(l,s)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.show()


#
#
# print(proj_trainx.shape)
# print(np.linalg.matrix_rank(project))

# eigv,eigvect = eigsh(project,k=10)
# print(np.argsort(eigv))
# print(eigv[-10:])
#
# c =0
# for i in eigv:
#     if(i>0):
#         c=c+1
# print(c)
#
from skimage.util import invert
#
#
#

# plot Images
# for i in range(0,10):
#     t = eigvect[:,i]
#     plt.imshow(img_as_float(t.reshape(50,50)))
#     plt.colorbar()
#     plt.savefig("plots/plt_"+str(i+1)+"_green.png")
#     plt.clf()
#     # plt.show()

#classifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
#
#
# # m = LogisticRegression()
# m = LinearSVC()
# # m.fit(trans_trainx,trainy)
# m.fit(trans_trainx,trainy)
# # m_t.fit(trans_trainx,trainy)
# # print(m.score(trans_testx,testy))
# score=m.score(trans_testx,testy)
# result.append(score)
# # print(m_t.score(trans_testx,testy))
# #
# f.write(str(fold+1)+" : "+str(score)+"\n")
# f.close()

#
# f=open('dataset1_pca.txt',mode='a')
# f.write("Mean : "+str(np.mean(result))+"\n")
# f.write("Std : "+str(np.std(result))+"\n")
# f.close()