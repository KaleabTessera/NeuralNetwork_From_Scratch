#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""A Fully-connected Neural Network, build from scratch using only Numpy."""
 
# Imports
import numpy as np
# import sklearn
from sklearn import preprocessing
 
# NN class
class NN:
    def __init__(self,X,numHiddenLayers,numOfNodesPerHiddenLayer,y,showShapeOfNN=False):
        self.X = X
        self.numHiddenLayers = numHiddenLayers
        self.numOfNodesPerHiddenLayer = numOfNodesPerHiddenLayer
        self.y = y

        self.w1 =  np.random.rand(self.X.shape[1],numOfNodesPerHiddenLayer[0])
        self.w2 =  np.random.rand(numOfNodesPerHiddenLayer[0],numOfNodesPerHiddenLayer[1])
        self.w3 =  np.random.rand(numOfNodesPerHiddenLayer[1],self.y.shape[1])

        if(showShapeOfNN):
            print("X",self.X.shape)
            print("w1",self.w1.shape)
            print("w2",self.w2.shape)
            print("w3",self.w3.shape)
            print("y",self.y.shape)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        b1 = 1
        b2 = 1
        b3 = 1

        # Hidden layers
        self.h1 = self.sigmoid(np.dot(self.X,self.w1) + b1)
        self.h2 = self.sigmoid(np.dot(self.h1,self.w2)     + b2)
        
        #Output
        self.pred = self.sigmoid(np.dot(self.h2,self.w3)     + b3)
    
    # Mean Squared Error
    def loss(self):
        self.error =  np.mean((self.y - self.pred)**2)
        print("Loss: ",self.error)

    def loss_dir(self):
        return 2 *(self.y - self.pred)
      
    def sigmoid_der(self,a):
        return a * (1.0 - a) 

    def backprop(self,learning_rate=0.1):
        # dJQ_dW3 - affect of Weight Layer 3 on cost
        dS3_dW3 = self.h2
        dS3 = np.dot(self.h2,self.w3)
        dy_DS3 = self.sigmoid_der(dS3)
        dJQ_dy = self.loss_dir()
        dJQ_dW3 = np.multiply(dy_DS3,dJQ_dy).T.dot(dS3_dW3)
#         print("dJQ_dW3")
#         print(dJQ_dW3)

        # dJQ_dW2 - affect of Weight Layer 2 on cost
        dS2_dW2 = self.h1
        dS2 = np.dot(self.h1,self.w2)
        dh2_dS2 = self.sigmoid_der(dS2)#np.dot(self.sigmoid(dS2),(1-self.sigmoid(dS2)).T)
        
        dS3_dh2 = self.w3
        dJQ_dh2 = dS3_dh2.dot(np.multiply(dJQ_dy,dy_DS3).T)
        dJQ_dW2 = dJQ_dh2.dot(np.multiply(dh2_dS2,dS2_dW2)) #TODO Consider which should be multiplied and which dot product.
        
        #dJQ_dW1 - affect of Weight Layer 1 on cost
        dS1_dW1 = self.X
        dS1 = np.dot(self.X,self.w1)
        dh1_dS1 = self.sigmoid_der(dS1)
        
        dS2_dh1 = self.w2
        dh2_dS2 = self.sigmoid_der(dS2)
        dS3_dh2 = self.w3
        
        dJQ_dh1 = np.multiply(dJQ_dh2.T,dh2_dS2).dot(dS2_dh1)  #TODO check
        dJQ_dW1 = np.multiply(dJQ_dh1,dh1_dS1).T.dot(dS1_dW1) #TODO check
        
        self.w1 -= learning_rate * dJQ_dW1.T
        self.w2 -= learning_rate * dJQ_dW2.T
        self.w3 -= learning_rate * dJQ_dW3.T

       

        
    def run(self):
      for i in np.arange(10):
        self.forward()
        self.loss()
        self.backprop()


def main():
    data = np.loadtxt("data_banknote_authentication.txt",delimiter=',')
    X = data[:,0:4]
    
    scaler = preprocessing.StandardScaler() 
    X = scaler.fit_transform(X)
    
    y = data[:,4].reshape(99,1)
    numHiddenLayers = 2
    numNodesPerLayer = [10,10]
    neural_network = NN(X,numHiddenLayers,numNodesPerLayer,y,True)


    neural_network.run()
   

if __name__ == "__main__":
	main()