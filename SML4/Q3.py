import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
import pickle


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

t= unpickle("CIFAR/bw_data_batch_1")
# joblib.dump(t,"CIFAR1/bw_data_batch_1")
# pickle.dump(t,open("CIFAR1/bw_data_batch_1",mode='w'),protocol=2)
trainx=t[b'data']

trainy = list(t[b'labels'])

for i in range(2,6):
    t= unpickle("CIFAR1/bw_data_batch_"+str(i))
    joblib.dump(t, "CIFAR1/bw_data_batch_1.sav")

    # pickle.dump(t,open("CIFAR1/bw_data_batch_"+str(i),mode='w'),protocol=2)
    temp= t[b'data']
    templ=t[b'labels']
    trainx = np.concatenate((trainx,temp),axis=0)
    trainy= trainy+list(templ)

t=unpickle("CIFAR1/bw_test_batch")
# pickle.dump(t,open("CIFAR1/bw_test_batch",mode=w),protocol=2)
testx = t[b'data']
testy = t[b'labels']



model = MLPClassifier(hidden_layer_sizes=(50),activation="relu",solver="adam")
bmodel = BaggingClassifier(base_estimator=model,n_estimators=5,verbose=True)
bmodel.fit(trainx,trainy)
bmodel.score(testx,testy)