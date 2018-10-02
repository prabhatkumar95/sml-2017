import matplotlib.pyplot as plt



plt.plot(['Linear','Relu','Sigmoid'],[0.2918085,0.0212765,0.0212765],label="(256,128,64,128)")
plt.plot(['Linear','Relu','Sigmoid'],[0.473297,0.0212765,0.0212765],label="(512,128,64)")
plt.plot(['Linear','Relu','Sigmoid'],[0.415644,0.0212765,0.0212765],label="(256,512,128)")
plt.legend()
plt.title("Accuracy Vs Activation")
plt.xlabel("Activation")
plt.ylabel("Accuracy")
plt.show()