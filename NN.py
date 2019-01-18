#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""A Fully-connected Neural Network, build from scratch using only Numpy."""
 
# Imports
import numpy as np
 
# NN class
class NN:
    def __init__(self,X,numHiddenLayers,numOfNodesPerHiddenLayer,y,showShapeOfNN=False):
        self.X = X
        self.numHiddenLayers = numHiddenLayers
        self.numOfNodesPerHiddenLayer = numOfNodesPerHiddenLayer
        self.y = y

        self.w1 =  np.random.rand(self.X.shape[1],numOfNodesPerHiddenLayer[0])
        self.w2 =  np.random.rand(numOfNodesPerHiddenLayer[0],numOfNodesPerHiddenLayer[1])
        self.w3 =  np.random.rand(numOfNodesPerHiddenLayer[1],y.shape[1])

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
 

    def loss(self):
        self.error= (-self.y * np.log(self.pred) - (1 - self.y) * np.log(1 - self.pred)).mean()
        print("Loss: ",self.error)

    def sigmoid_der(self,a):
        return np.dot(self.sigmoid(a),(1 - self.sigmoid(a)).T)

    def backprop(self):
        # dJQ_dW3 - affect of Weight Layer 3 on cost
        dS1_dW3 = self.h2
        dS1 = np.dot(self.h2,self.w3)
        dy_DS1 = np.dot(self.sigmoid(dS1), (1 - self.sigmoid(dS1)).T)
        dJQ_dy = (self.pred - self.y )/(self.pred)*(1-self.pred)
        dJQ_dW3 = dJQ_dy.T.dot(dy_DS1).dot(dS1_dW3)
        print("dJQ_dW3")
        print(dJQ_dW3)

        # dJQ_dW2 - affect of Weight Layer 2 on cost
        dS2_dW2 = self.h1
        dS2 = np.dot(self.h1,self.w2)
        dh2_dS2 = np.dot(self.sigmoid(dS2),(1-self.sigmoid(dS2)).T)
        dJQ_dh2 = self.sigmoid_der(dS1).dot(dJQ_dy).dot(self.w3.T)
        dJQ_dW2 = dJQ_dh2.T.dot(dh2_dS2).dot(dS2_dW2)
        print("dJQ_dW2")
        print(dJQ_dW2)


def main():
    X = np.ones((100,4), dtype=np.int16 )
    y = np.random.rand(100,1)
    
    numHiddenLayers = 2
    numNodesPerLayer = [10,10]
    neural_network = NN(X,numHiddenLayers,numNodesPerLayer,y)

    neural_network.forward()
    neural_network.loss()
    neural_network.backprop()

if __name__ == "__main__":
	main()