import numpy as np
from skimage import color
from skimage import transform
import os
from skimage.io import imread,imsave,imshow
from skimage.util import invert,img_as_float
from pca import PCA
import cv2
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.image as implt



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

data=[]
image_matrix=[]
for i in image_list:
    temp=imread(i,True)
    # print(temp.shape)
    temp = transform.resize(temp,(50,50))
    temp=temp.flatten()
    # print(temp.shape)
    data.append(temp)
#
# mean = np.mean(data, axis=0)
# X_centered = data - mean
#
# eigen_vect = np.linalg.svd(X_centered.T)[0]
# print(eigen_vect.shape)
#
pca_handle = PCA()
pca_handle.calc_vector_space(data)
# print(pca_handle.eigen_val)
# print(np.sum(pca_handle.eigen_val))
projection_matrix = pca_handle.get_projection_vector(0.99)
# print(len(projection_matrix))
# print(len(projection_matrix[1]))
projection_matrix = np.array(projection_matrix)
print (projection_matrix.shape)
for i in range(0,len(projection_matrix)):
    plt.imshow(img_as_float(projection_matrix[i].reshape(50,50)),cmap='jet')
    plt.colorbar()
    plt.savefig("output_eigen_vect/energy_99/eign_color_"+str(i)+".png")
    plt.close()

# data= np.array(data)

# implt.imread("output_eigen_vect/eign_00.png")
# plt.imshow(projection_matrix[1].reshape(50,50),cmap="gray")
# plt.savefig("outp.png")
# plt.show(cmap="gray")

# print(data.shape)
# new_data = np.matmul(projection_matrix,data.T)
# print(new_data.shape)
