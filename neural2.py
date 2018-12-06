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
        self.w=0.5*np.random.randn(olen,ilen)
        func=lambda x: max(min(x,self.cap),-1*self.cap)
        self.w=np.vectorize(func)(self.w)
        self.b=0.5*np.random.randn(olen,1)
        self.b=np.vectorize(func)(self.b)


    #initialize bias, weights, etc. use numpy vectors/arrays

class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers #list of Layers
        self.ll=[]
        self.act=[]
        self.z=[]
        for i in range(len(self.layers)-1):
            self.ll.append(Layer(self.layers[i],self.layers[i+1]))
            #print(self.ll[i].w.shape)
            #initializations

    def forwardprop(self, input_vector):
        self.act=[]
        self.z=[]
        sigmoid = lambda val: 1.0/(1.0 + np.exp(-val))
        cur=input_vector
        self.act.append(cur)
        for L in self.ll:
            prod=np.dot(L.w,cur).reshape(-1,1)
            cur=sigmoid(prod+L.b)
            self.act.append(cur)
            self.z.append(prod)
            # print(input_vector)
            # print(L.weights)
            # print(L.bias)
            # print(cur)
         #forward propogate thru layers
        return cur
    def backprop(self, y,learning_rate):
        sigmoid = lambda val: 1.0/(1.0 + np.exp(-val))
        sig_deriv = lambda val: (1-sigmoid(val))*sigmoid(val)

        n=len(self.ll)-1
        dif=self.act[n+1]-y
        delta=dif*sig_deriv(self.z[n])
        dw=np.dot(delta,self.act[n].T)
        self.ll[n].w=self.ll[n].w-learning_rate*dw
        self.ll[n].b=self.ll[n].b-learning_rate*delta
        while n>0:
            n-=1
            delta=np.dot(self.ll[n+1].w.T,delta)*sig_deriv(self.z[n])
            dw=np.dot(delta,self.act[n].reshape(-1,1).T)
            self.ll[n].w=self.ll[n].w-learning_rate*dw
            self.ll[n].b=self.ll[n].b-learning_rate*delta


def train(neuralnet, iterations, training_data,labels,learning_rate):
    for j in range(iterations): #number of iterations, to be tuned
        for ind,i in enumerate(training_data):
            #print(ind, i)
            res=neuralnet.forwardprop(i)
            # print(res)
            neuralnet.backprop(one_hot(labels[ind],10),learning_rate)
            if ind%1000==0:
                print(ind)
        print('Epoch %d done'%(j+1))

            #train (forward, backprop)
def one_hot(val,size):
    arr=np.zeros(size)
    arr[int(val)]=1
    return arr.reshape(-1,1)

def test(neuralnet, testing_data,labels):
    correct=0
    f.write('id,solution')
    f=open('output.txt','w')
    for ind,data_point in enumerate(testing_data):
        res=np.argmax(neuralnet.forwardprop(data_point))
        f.write(str(int(labels[ind]))+', '+str(res)+'\n')
    f.close()



        #get output for each data_point
        #compare output to ground truth
epochs=5
learning_rate=0.005

layers=[28*28,512,10]
#layers=[3,4,5,2]
n = NeuralNetwork(layers)
#train(n,epochs,np.array([[3,2,0],[1,2,4]]),np.array([1,0]),learning_rate)
start=time.time()
train_data=np.loadtxt('train.csv',delimiter=',')
loadtrain=time.time()
#print (train_data.shape)
print('Time to load train:',loadtrain-start)
train(n,epochs,train_data[:,1:]/128-1,train_data[:,0],learning_rate)
train=time.time()
print('Time to train:',train-loadtrain)
test_data=np.loadtxt('test.csv',delimiter=',',skiprows=1)
loadtest=time.time()
print('Time to load test:',loadtest-train)
test(n,test_data[:,1:]/128-1,test_data[:,0])
#print('Accuracy = %d'%(acc))
#print(test)
#data i/o
#train network
#test network
