import numpy as np


def reLu(x):
    x[x<0]=0
    return x

def dreLu(x):
    for j in range(len(x)):
        for i in range(0,len(x[j])):
            if(x[0][i]>0):
                x[j][i]=1
            else:
                x[j][i]=0
    return x



def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def dsoftmax(x):
    return

def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return x * (1.0 - x)



class NN:

    def __init__(self,_layer,learning_rate):
        self.layer = list(_layer)
        self.learningrate = learning_rate
        self.weight=[]
        self.basis=[]


    def fit(self,x,y,it):
        xshape = x.shape
        yshape=y.shape
        print (x.shape)
        activation=[]
        self.weight=[]
        self.basis=[]
        n_classes = len(np.unique(y))

        '''first layer weight and basis'''
        self.weight.append(np.random.uniform(low=-1.0,high=1.0,size=(xshape[1],self.layer[0])))
        #print self.weight[-1].shape
        self.basis.append(np.random.uniform(low=-1.0,high=1.0, size=(1,self.layer[0])))

        '''second layer onwards weight & basis initialisation'''
        for i in range(1,len(self.layer)):
            xtemp = np.random.uniform(low=-1.0, high=1.0, size=(self.layer[i-1], self.layer[i]))
            self.basis.append(np.random.uniform(low=-1.0,high=1.0, size=(1,self.layer[i])))
            self.weight.append(xtemp)
         #   print self.weight[-1].shape


        '''output layer weight and basis initialisation'''
        #last weight 50x1
        self.weight.append(np.random.uniform(low=-1.0, high=1.0, size=(self.layer[-1],n_classes)))
        self.basis.append(np.random.uniform(low=-1.0, high=1.0, size=(1,n_classes)))
        #print self.weight[-1].shape

        '''weight matrix list order 784x100, 100x50, 50x1'''


        for m in range(0,it):
            for r in range(0,x.shape[0]):
                x_input = x[r]
                activation=[]
                # print x[i].shape
                activation.append(x_input)
                '''calculate activation values for neurons and append to activation list'''
                for l in range(len(self.layer)):
                    # print activation[-1].shape
                    activation.append(reLu(np.dot(activation[-1],self.weight[l])+self.basis[l]))
                    # print self.weight[l].shape
                activation.append(softmax(np.dot(activation[-1],self.weight[-1])+self.basis[-1]))
                val = activation[-1]
                # print val.shape
                error_o = (val-y[r])

                errors=[]
                errors.append(error_o)


                # print len(activation)
                for i in range(len(activation)-2,0,-1):
                     # print i
                     # print(self.weight[i].T.shape)
                     errors.append(errors[-1].dot(self.weight[i].T)*dreLu(activation[i]))


                for i in range(0,len(self.weight)):
                    # print len(self.weight)
                    self.weight[i]=self.weight[i]-self.learningrate*np.dot(np.atleast_2d(activation[i]).T,errors[-(i+1)])
                    '''basis updation'''
                    self.basis[i]=self.basis[i]-np.sum(errors[-(i+1)])*self.learningrate
            # print('Error at iteration'+str(m)+ ':'+str((error_o)/2)+'\n')




    def predict (self,xtest):
        pred=[]
        for i in range(xtest.shape[0]):
            activation=[]
            activation.append(xtest[i]/float(255.0))
            for l in range(len(self.layer)):
                activation.append(sigmoid(np.dot(activation[-1], self.weight[l]) + self.basis[l]))

            activation.append(softmax(np.dot(activation[-1], self.weight[-1]) + self.basis[-1]))
            val = activation[-1]

            val=activation[-1]
            print (val[0])
            # print activation[-1].shape
            temp = np.zeros(10)
            temp.reshape(1,10)
            temp[np.argmax(val[0])]=1
            pred.append(temp)
        # print self.weight
        return pred

    def score(self,xtest,ytest):
        pred = []
        proba=[]
        sum = 0
        for i in range(xtest.shape[0]):
            activation = []
            activation.append(xtest[i])
            for l in range(len(self.layer)):
                activation.append(reLu(np.dot(activation[-1], self.weight[l]) + self.basis[l]))
            activation.append(softmax(np.dot(activation[-1], self.weight[-1]) + self.basis[-1]))

            val = activation[-1]

            pred.append(np.argmax(val[0]))
            proba.append(val[0][np.argmax(val[0])])

            if(np.argmax(val[0])==np.argmax(ytest[i])):
                sum=sum+1

        print(np.unique(pred,return_counts=True))
        return sum/float(ytest.shape[0]),proba,pred

