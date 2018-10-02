import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.io import imread
from skimage.transform import resize
from pca import PCA
from sklearn.svm import LinearSVC
from sklearn.externals import joblib



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
    temp = resize(temp,(50,50))
    temp=temp.flatten()
    y_data.append(p)
    # print(temp.shape)
    data.append(temp)
# print (y_data)
y_data=np.array(y_data)
data=np.array(data)
# y_data = np.array(y_data).reshape((data.shape[0],1))
# print(data.shape,y_data.shape)
# print(len(y_data.shape))


pca_handle = PCA()
pca_handle.calc_vector_space(data)
# print(pca_handle.eigen_val)
# print(np.sum(pca_handle.eigen_val))
projection_matrix = pca_handle.get_projection_vector(0.90)
# print(len(projection_matrix))
# print(len(projection_matrix[1]))
projection_matrix = np.array(projection_matrix)


transformed_x = np.matmul(projection_matrix,data.T).T
joblib.dump(projection_matrix,"projection_matrix_99.sav")
print(transformed_x.shape)
# x_train,x_test,y_train,y_test=train_test_split(data,y_data,test_size=0.5,stratify=True)

x_train,x_test,y_train,y_test=train_test_split(data,y_data,test_size=0.3,stratify=y_data)

print(x_train.shape, y_train.shape, x_test.shape,y_test.shape)

x_train_projected = np.matmul(projection_matrix,x_train.T).T
x_test_projected = np.matmul(projection_matrix,x_test.T).T

x_train_projected,y_train,x_train = shuffle(x_train_projected,y_train,x_train)
x_test_projected,y_test,x_test = shuffle(x_test_projected,y_test,x_test)
model = LinearSVC(random_state=1)
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

