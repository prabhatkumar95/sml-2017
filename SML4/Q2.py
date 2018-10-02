import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def plot_graph(x,y,filename,title,xlabel,ylabel):
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("plots/"+filename+".png")




#read dataset
import idx2numpy as idx2numpy

data_train = idx2numpy.convert_from_file('emnist/emnist-balanced-train-images-idx3-ubyte')
data_test = idx2numpy.convert_from_file('emnist/emnist-balanced-test-images-idx3-ubyte')
label_train = idx2numpy.convert_from_file('emnist/emnist-balanced-train-labels-idx1-ubyte')
label_test = idx2numpy.convert_from_file('emnist/emnist-balanced-test-labels-idx1-ubyte')

print(np.unique(label_test).shape)
print(data_train.shape,data_test.shape,label_train.shape,label_test.shape)

data_train_flatten=[]
data_test_flatten=[]


for i in range(0,data_train.shape[0]):
    data_train_flatten.append(data_train[i].flatten())

for i in range(0,data_test.shape[0]):
    data_test_flatten.append(data_test[i].flatten())

data_test=np.array(data_test_flatten)
data_train=np.array(data_train_flatten)



#
# accuracy vs learning rate
# accuracy_lr  =[]
# for l_rate in [0.2,0.1,0.001]:
#
#     model = MLPClassifier(hidden_layer_sizes=(512,256,64),activation="logistic",max_iter=100,verbose=True,solver='adam',learning_rate_init=l_rate)
#     model.fit(data_train,label_train)
#
#     accuracy_lr.append(model.score(data_test,label_test))
#
# print(accuracy_lr)
# plot_graph([0.2,0.1,0.001],accuracy_lr,"Q2_1_learning_rate_sigmoidv1", "Accuracy Vs Learning Rate (Sigmoid) (512,256,64)","Learning Rate","Accuracy")

#
# accuracy_lr  =[]
# for it in [20,60,100]:
#
#     model = MLPClassifier(hidden_layer_sizes=(512,256,128),activation="identity",max_iter=it,verbose=True,solver='adam',learning_rate_init=0.1,early_stopping=False)
#     model.fit(data_train,label_train)
#
#     accuracy_lr.append(model.score(data_test,label_test))
#
# print(accuracy_lr)
# plot_graph([20,60,100],accuracy_lr,"Q2_4_epochs_Sigmoid", "Accuracy Vs Epochs (Sigmoid)","Epochs","Accuracy")

model = MLPClassifier(hidden_layer_sizes=(256,128,64),activation="identity",max_iter=100,verbose=True,solver='sgd',learning_rate_init=0.1,alpha=0.01,early_stopping=True)
model.fit(data_train,label_train)

print(model.score(data_test,label_test))