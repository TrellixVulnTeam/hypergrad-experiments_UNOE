import lasagne.layers as ll
import theano.tensor as T
from lasagne.objectives import categorical_crossentropy
from lasagne import nonlinearities
from layers import DenseLayerWithReg, Conv2DLayerWithReg
from theano.ifelse import ifelse
import theano
import numpy as np
import time


# Michael: Batch normalization? http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html


class MLP(object):
    def __init__(self, x, y, args):
        self.params_theta = []
        self.params_lambda = []
        self.params_weight = []
        if args.dataset == 'mnist':
            input_size = (None, 28*28)
        elif args.dataset == 'cifar10':
            input_size = (None, 3, 32*32)
        else:
            raise AssertionError
        layers = [ll.InputLayer(input_size)]
        penalty = theano.shared(np.array(0.))
        for (k, num) in enumerate(args.MLPlayer):
            # the last layer should use softmax
            if k == len(args.MLPlayer) - 1:
                # layers.append(ll.DenseLayer(layers[-1], num, nonlinearity=nonlinearities.softmax))
                layers.append(DenseLayerWithReg(args, layers[-1], num_units=num,
                                                nonlinearity=nonlinearities.softmax))
            else:
                # layers.append(ll.DenseLayer(layers[-1], num))
                layers.append(DenseLayerWithReg(args, layers[-1], num_units=num))
            if layers[-1].W is not None:
                self.params_theta += [layers[-1].W, layers[-1].b]
                self.params_weight += [layers[-1].W]

                # define new regularization term for a layer
                if args.regL2 is True:
                    tempL2 = layers[-1].L2 * T.sqr(layers[-1].W) #Michael: use 10**regularization constants
                    penalty += T.sum(tempL2)
                    self.params_lambda += [layers[-1].L2]
                if args.regL1 is True:
                    tempL1 = layers[-1].L1 * T.abs(layers[-1].W) #Michael: use 10**regularization constants
                    penalty += T.sum(tempL1)
                    self.params_lambda += [layers[-1].L1]

        self.layers = layers
        self.y = ll.get_output(layers[-1], x, deterministic=False)
        self.prediction = T.argmax(self.y, axis=1)
        self.penalty = penalty
        # self.penalty = penalty if penalty != 0. else T.constant(0.)
        print(self.params_lambda)
        # time.sleep(20)
        # cost function
        self.loss = T.mean(categorical_crossentropy(self.y, y))
        self.lossWithPenalty = T.add(self.loss, self.penalty)
        print("loss and losswithpenalty", type(self.loss), type(self.lossWithPenalty))
        # self.classError = T.mean(T.cast(T.neq(self.prediction, y), 'float32'))


class ConvNet(object):
    def add_params_to_self(self, args, layer):
        if layer.W is not None:
            self.params_theta += [layer.W, layer.b] # Michael: weights and biases?
            self.params_weight += [layer.W] # Michael: weights but not biases?

            # define new regularization term for a layer
            if args.regL2 is True:
                #tempL2 = layer.L2 * T.sqr(layer.W)
                tempL2 = (10.**layer.L2) * T.sqr(layer.W) #Michael: use 10.**regularization constants
                self.penalty += T.sum(tempL2)
                self.params_lambda += [layer.L2]
            if args.regL1 is True:
                #tempL1 = layer.L1 * layer.W
                tempL1 = (10.**layer.L1) * T.abs(layer.W) #Michael: use 10.**regularization constants
                self.penalty += T.sum(tempL1)
                self.params_lambda += [layer.L1]

    def __init__(self, x, y, args):
        self.params_theta = []
        self.params_lambda = []
        self.params_weight = []
        if args.dataset == 'mnist':
            input_size = (None, 1, 28, 28)
        elif args.dataset == 'cifar10':
            input_size = (None, 3, 32, 32)
        else:
            raise AssertionError
        layers = [ll.InputLayer(input_size)]
        self.penalty = theano.shared(np.array(0.))

        #conv1
        layers.append(Conv2DLayerWithReg(args, layers[-1], 20, 5))
        self.add_params_to_self(args, layers[-1])
        layers.append(ll.MaxPool2DLayer(layers[-1], pool_size=2, stride=2))
        #conv1
        layers.append(Conv2DLayerWithReg(args, layers[-1], 50, 5))
        self.add_params_to_self(args, layers[-1])
        layers.append(ll.MaxPool2DLayer(layers[-1], pool_size=2, stride=2))
        
        # Michael: add dropout
        layers.append(ll.DropoutLayer(layers[-1]))    # Michael
        #fc1
        layers.append(DenseLayerWithReg(args, layers[-1], num_units=500))
        self.add_params_to_self(args, layers[-1])
        layers.append(ll.DropoutLayer(layers[-1]))    # Michael
        #softmax
        layers.append(DenseLayerWithReg(args, layers[-1], num_units=10, nonlinearity=nonlinearities.softmax))
        self.add_params_to_self(args, layers[-1])
        # no dropout on output

        self.layers = layers
        self.y = ll.get_output(layers[-1], x, deterministic=False)
        self.prediction = T.argmax(self.y, axis=1)
        # self.penalty = penalty if penalty != 0. else T.constant(0.)
        print(self.params_lambda)
        # time.sleep(20)
        # cost function
        self.loss = T.mean(categorical_crossentropy(self.y, y))
        self.lossWithPenalty = T.add(self.loss, self.penalty)
        print("loss and losswithpenalty", type(self.loss), type(self.lossWithPenalty))

# Michael: wide resnet: https://gist.github.com/FlorianMuellerklein/3d9ba175038a3f2e7de3794fa303f1ee
# https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne/blob/master/models.py