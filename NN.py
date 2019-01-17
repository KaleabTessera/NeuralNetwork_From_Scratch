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
        h1 = self.sigmoid(np.dot(self.X,self.w1) + b1)
        h2 = self.sigmoid(np.dot(h1,self.w2)     + b2)
        
        #Output
        self.pred =  np.dot(h2,self.w3)     + b3
 

    def loss(self):
        # L2 Norm
        n = self.X.shape[0]
        self.error = 1/n * np.sum((self.y - self.pred) ** 2)
        print("Loss: ",self.error)


def main():
    X = np.ones((100,4), dtype=np.int16 )
    y = np.random.rand(100,2)
    
    numHiddenLayers = 2
    numNodesPerLayer = [10,10]
    neural_network = NN(X,numHiddenLayers,numNodesPerLayer,y)

    neural_network.forward()
    neural_network.loss()

if __name__ == "__main__":
	main()