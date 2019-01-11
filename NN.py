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
        self.w3 =  np.random.rand(numOfNodesPerHiddenLayer[1],y.shape[0])

        if(showShapeOfNN)
            print(self.w1.shape)
            print(self.w2.shape)
            print(self.w3.shape)
 
 
def main():
    X = np.ones((100,4), dtype=np.int16 )
    y = np.random.rand(2)
    
    numHiddenLayers = 2
    numNodesPerLayer = [10,10]
    neural_network = NN(X,numHiddenLayers,numNodesPerLayer,y)

if __name__ == "__main__":
	main()