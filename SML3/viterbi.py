import numpy as np
import math

def viterbi(initial, observation,transition,obseq):
    table = np.zeros(initial.shape[0]*obseq.shape[0]).reshape(initial.shape[0],obseq.shape[0])
    max_index = np.zeros(initial.shape[0]*obseq.shape[0]).reshape(initial.shape[0],obseq.shape[0])
    # print(initial.shape,table[:,0].shape)
    table[:,0]=np.add(np.log(initial),np.log(observation[:,int(obseq[0])].reshape(26,1))).reshape(26,)

    for obs in range(1,obseq.shape[0]):
        for c in range(0,initial.shape[0]):
            temp = np.add(table[:,obs-1],np.log(transition[:,c]))
            # print(temp)
            index = np.argmax(temp)
            # print (index)
            max_index[c][obs]=int(index)
            table[c][obs]=temp[index]+np.log(observation[c][int(obseq[obs])])

    result=[]
    init = np.argmax(table[:,-1])
    result.append(init)
    for i in range(0,max_index.shape[1]-1):
        # print(init,i)
        index=max_index[int(init)][-i-1]
        result.append(index)
        init=index

    return result
