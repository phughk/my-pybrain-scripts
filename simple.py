#!/usr/bin/python

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

def logicTest():
	inLayer = LinearLayer(2)
	hiddenLayer = SigmoidLayer(6)
	outLayer = LinearLayer(4) # OR, AND, NOT, XOR
	
	n=FeedForwardNetwork()
	n.addInputModule(inLayer)
	n.addModule(hiddenLayer)
	n.addOutputModule(outLayer)

	inToHidden = FullConnection(inLayer, hiddenLayer)
	hiddenToOut = FullConnection(hiddenLayer, outLayer)

	n.addConnection(inToHidden)
	n.addConnection(hiddenToOut)

	n.sortModules()
	
	print n.activate([0, 1])

if __name__ == '__main__':
	logicTest()
