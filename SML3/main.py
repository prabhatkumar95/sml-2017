import numpy as np
from viterbi import viterbi

initial = np.loadtxt('HMM/initialStateDistribution.txt')

observation = np.loadtxt('HMM/observationProbMatrix_trans.txt')
# print(observation.shape)
trans = np.loadtxt('HMM/transitionProbMatrix_trans.txt')
t= np.loadtxt("HMM/observations_art_trans.txt")

initial=initial.reshape(initial.shape[0],1)
# print(initial.shape)
# print(observation.shape)
# print(trans.shape)

result = viterbi(initial,observation,trans,t)
result=result[::-1]
fresult=[int(result[0])]
for i in range(1,len(result)):
    if fresult[-1] != int(result[i]):
        fresult.append(int(result[i]))

print(fresult)
for i in fresult:
    print(chr(int(i)+65),end="")