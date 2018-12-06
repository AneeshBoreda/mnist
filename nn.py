import numpy as np; import math
from numpy import matlib
import random
import csv
import time

class Layer(object):
    def __init__(self, ilen,olen):
        self.ilen =ilen #layer size
        self.olen=olen
        self.cap=1
        self.weights=0.5*np.random.randn(ilen,olen)
        func=lambda x: max(min(x,self.cap),-1*self.cap)
        self.weights=np.vectorize(func)(self.weights)
        self.bias=0.5*np.random.randn(1,olen)
        self.bias=np.vectorize(func)(self.bias)


    #initialize bias, weights, etc. use numpy vectors/arrays

class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers #list of Layers
        self.layerlist=[]
        self.activations=[]
        for i in range(len(self.layers)-1):
            self.layerlist.append(Layer(self.layers[i],self.layers[i+1]))
            #initializations

    def forwardprop(self, input_vector):
        self.activations=[]
        self.z=[]
        sigmoid = lambda val: 1.0/(1.0 + np.exp(-val))
        cur=input_vector
        for L in self.layerlist:
            prod=cur.dot(L.weights)
            cur=sigmoid(prod+L.bias)
            self.activations.append(cur)
            self.z.append(prod)
            # print(input_vector)
            # print(L.weights)
            # print(L.bias)
            # print(cur)
         #forward propogate thru layers


    def backprop(self, y,learning_rate):
        L=len(self.layerlist)-1
        sigmoid = lambda val: 1.0/(1.0 + np.exp(-val))
        #sig_deriv = lambda val: (1-sigmoid(val))*sigmoid(val)
        sig_deriv = lambda val: (1-sigmoid(val))*sigmoid(val)
        #Calculate gradients
        #Update weights
        error=self.activations[L]-y
        delta=error*sig_deriv(self.z[L])
        # print(error)
        # print(self.activations[L])
        weight_delta=np.dot(delta.T,self.z[L-1].T).T
        # print(self.layerlist[L].weights.shape)
        self.layerlist[L].bias=self.layerlist[L].bias-learning_rate*delta
        self.layerlist[L].weights=self.layerlist[L].weights-learning_rate*weight_delta
        for i in range(2,L+1):
            self.layerlist[L+2-i].bias=self.layerlist[L+2-i].bias-learning_rate*delta
            delta=np.dot(self.layerlist[L+2-i].weights,delta.T)*sig_deriv(self.z[L+1-i])
            weight_delta=np.dot(delta,self.z[L+1-i].T)
            self.layerlist[L+1-i].weights=self.layerlist[L+1-i].weights-learning_rate*weight_delta

        # for i in range(2,L+1):
        #     sigprime=sig_deriv(self.activations[i])
        #     delta=np.dot(self.layerlist[L+1-i].weights,delta)*sigprime
        # delta=error*sig_deriv(self.activations[0])
        #
        # change=self.activations[1].dot(delta)
        # self.layers[L]=self.layers[L]-learning_rate*delta
        #
        # delta2=self.layers[L-1].weights.dot(delta)*sig_deriv(activations[1])
        # change2=delta2.dot(self.activations[2])




def train(neuralnet, iterations, training_data,labels,learning_rate):
    for j in range(iterations): #number of iterations, to be tuned
        for ind,i in enumerate(training_data):
            #print(ind, i)
            res=neuralnet.forwardprop(i)
            #print(res)
            neuralnet.backprop(one_hot(labels[ind],10),learning_rate)

            #train (forward, backprop)
def one_hot(val,size):
    arr=np.zeros(size)
    arr[int(val)]=1
    return arr
def test(neuralnet, testing_data):
    for data_point in testing_data:
        pass
        #get output for each data_point
        #compare output to ground truth
epochs=1
learning_rate=0.001

layers=[28*28,512,10]
#layers=[4,6,3,2]
n = NeuralNetwork(layers)
#train(n,epochs,np.array([[3,2,0,0],[0,0,0,0]]),np.array([1,0]),learning_rate)
start=time.time()
train_data=np.loadtxt('train.csv',delimiter=',')
loadtrain=time.time()
print (train_data.shape)
train(n,epochs,train_data[:,1:],train_data[:,0],0.001)
print('Time to load train:',loadtrain-start)
test_data=np.loadtxt('test.csv',delimiter=',',skiprows=1)
loadtest=time.time()
print('Time to load test:',loadtest-loadtrain)

#print(test)
#data i/o
#train network
#test network
