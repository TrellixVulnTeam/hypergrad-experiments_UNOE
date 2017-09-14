#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Creates a DenseNet model in Lasagne, following the paper
"Densely Connected Convolutional Networks"
by Gao Huang, Zhuang Liu, Kilian Q. Weinberger, 2016.
https://arxiv.org/abs/1608.06993

This defines the model in a different way than existing implementations,
concatenating the normalized feature maps rather than normalizing the same
features again and again. This can cut training time by 20-25%. Note that
the model still has separately learned scales and shifts in front of each
rectifier, otherwise it would reduce to post-activation (conv-bn-relu).

Author: Jan Schlüter

Rewritten as a class for consistency with DrMAD
"""


from __future__ import print_function #use parentheses for print
from __future__ import division #/ is floating point division, // rounds to an integer


import lasagne
from lasagne.layers import (InputLayer, ConcatLayer,
                            DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                            NonlinearityLayer)

from lasagne.layers import Layer
from lasagne import nonlinearities, init

from layers import DenseLayerWithReg as DenseLayer, Conv2DLayerWithReg as Conv2DLayer, ScaleLayer, BiasLayer, GaussianNoiseLayer
from lasagne.nonlinearities import rectify, softmax
#try:
#    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
#except ImportError:
from lasagne.layers import BatchNormLayer



import lasagne.layers as ll
import theano.tensor as T
from lasagne.objectives import categorical_crossentropy
import theano
import numpy as np
import time




class DenseNet(object):
    def add_params_to_self(self, args, layer):
        # W for weights with corresponding hyperparamters, b for weights without hyperparameters
        # Want to params_lambda and params_weight to be ordered the same (corresponding [(weights, hyperparameters),...])
        # otherwise Hessian-vector products will not be properly aligned
        if layer.W is not None: #regularized weights
            self.params_weight += [layer.W]
            if layer.b is not None:
                self.params_theta += [layer.W, layer.b]
            else:
                self.params_theta +=[layer.W]
            # define new regularization term for a layer
            if layer.L2 is not None:
                #tempL2 = layer.L2 * T.sqr(layer.W)
                tempL2 = (10.**layer.L2) * T.sqr(layer.W) #regularization on logarithmic scale
                #print("regularization", layer.name, tempL2, tempL2.type)
                self.penalty += T.sum(tempL2)
                self.params_lambda += [layer.L2]
            if layer.L1 is not None:
                tempL1 = (10.**layer.L1) * T.abs(layer.W) #Michael: use 10.**regularization constants
                self.penalty += T.sum(tempL1)
                self.params_lambda += [layer.L1]
            #if layer.initScale is not None:
                #self.params_lambda += [layer.initScale]
        elif layer.b is not None: #all unregularized weights
            self.params_theta += [layer.b]

    #def add_noise_hypers_to_self(self, layer): 
    #    self.params_noise += [layer.sigma]


    def affine_relu_conv(self, args, layers, channels, filter_size, dropout, name_prefix):
        #TODO: treat initialization as hyperparameter, but don't regularize?
        layers.append(ScaleLayer(args, layers[-1], name=name_prefix + '_scale'))
        self.add_params_to_self(args, layers[-1])
        #no regularization on biases
        layers.append(BiasLayer(args, layers[-1], name=name_prefix + '_shift'))
        self.add_params_to_self(args, layers[-1])
        layers.append(NonlinearityLayer(layers[-1], nonlinearity=rectify,
                                    name=name_prefix + '_relu'))
        #TODO: regularization here is disconnected
        layers.append(Conv2DLayer(args, layers[-1], channels, filter_size, pad='same',
                              W=lasagne.init.HeNormal(gain='relu'),
                              b=None, nonlinearity=None,
                              name=name_prefix + '_conv'))
        self.add_params_to_self(args, layers[-1])
        if dropout:
            layers.append(DropoutLayer(layers[-1], dropout))
        #TODO: add Gaussian noise? Better to put after BN?
        return layers[-1]

    def dense_block(self, args, layers, num_layers, growth_rate, dropout, name_prefix):
        # concatenated 3x3 convolutions
        for n in range(num_layers):
            network = layers[-1]
            conv = self.affine_relu_conv(args, layers, channels=growth_rate,
                                    filter_size=3, dropout=dropout,
                                    name_prefix=name_prefix + '_l%02d' % (n + 1))
            #TODO: treat initialization as hyperparameter, but don't regularize parameters?
            conv = BatchNormLayer(conv, name=name_prefix + '_l%02dbn' % (n + 1),
                                  beta=None, gamma=None)
            #TODO: add Gaussian noise?
            layers.append(conv) #redundant?
            if args.addActivationNoise:
                conv = GaussianNoiseLayer(layers[-1], name=name_prefix + '_l%02dGn' % (n + 1), 
                                                 sigma=init.Constant(args.invSigmoidActivationNoiseMagnitude), shared_axes='auto')         
                self.params_noise.append(conv.sigma)
                layers.append(conv)
            #self.add_params_to_self(args, conv) #no parameters, beta=gamma=None
            layers.append(ConcatLayer([network, conv], axis=1,
                                  name=name_prefix + '_l%02d_join' % (n + 1)))
        return layers[-1]
    
    
    def transition(self, args, layers, dropout, name_prefix):
        # a transition 1x1 convolution followed by avg-pooling
        self.affine_relu_conv(args, layers, channels=layers[-1].output_shape[1],
                                   filter_size=1, dropout=dropout,
                                   name_prefix=name_prefix)
        layers.append(Pool2DLayer(layers[-1], 2, mode='average_inc_pad',
                              name=name_prefix + '_pool'))
        #TODO: treat initialization as hyperparameter, but don't regularize parameters?
        layers.append(BatchNormLayer(layers[-1], name=name_prefix + '_bn',
                                 beta=None, gamma=None))
        #TODO: add Gaussian noise
        if args.addActivationNoise:
                layers.append(GaussianNoiseLayer(layers[-1], name=name_prefix + '_Gn', 
                                                sigma=init.Constant(args.invSigmoidActivationNoiseMagnitude), shared_axes='auto'))
                self.params_noise.append(layers[-1].sigma)
        #self.add_params_to_self(args, layers[-1]) #no parameters, beta=gamma=None
        return layers[-1]

    def __init__(self, x, y, args):
        self.params_theta = []
        self.params_lambda = []
        self.params_weight = []
        self.params_noise = []
        if args.dataset == 'mnist':
            input_size = (None, 1, 28, 28)
        elif args.dataset == 'cifar10':
            input_size = (None, 3, 32, 32)
        else:
            raise AssertionError
        
        if (args.depth - 1) % args.num_blocks != 0:
            raise ValueError("depth must be num_blocks * n + 1 for some n")
        
        
        # input and initial convolution
        layers = [InputLayer(input_size)]
        self.penalty = theano.shared(np.array(0.))
        #TODO: add Gaussian noise
        if args.addInputNoise:
            layers.append(GaussianNoiseLayer(layers[-1], name='input_Gn', 
                                             sigma=init.Constant(args.invSigmoidInputNoiseMagnitude), shared_axes='all'))
            self.params_noise.append(layers[-1].sigma)
        
        layers.append(Conv2DLayer(args, layers[-1], args.first_output, 3, pad='same',
                              W=lasagne.init.HeNormal(gain='relu'),
                              b=None, nonlinearity=None, name='pre_conv'))
        self.add_params_to_self(args, layers[-1])
        
        layers.append(BatchNormLayer(layers[-1], name='pre_bn', beta=None, gamma=None))
        #TODO: add Gaussian noise, or not, because of following note
        #self.add_params_to_self(args, layers[-1])
        # note: The authors' implementation does *not* have a dropout after the
        #       initial convolution. This was missing in the paper, but important.
        # if dropout:
        #     layers.append(DropoutLayer(network, dropout))
        # dense blocks with transitions in between
        

        n = (args.depth - 1) // args.num_blocks
        for b in range(args.num_blocks):
            self.dense_block(args, layers, n - 1, args.growth_rate, args.dropout,
                                  name_prefix='block%d' % (b + 1))
            if b < args.num_blocks - 1:
                self.transition(args, layers, args.dropout,
                                     name_prefix='block%d_trs' % (b + 1))
                  
        # post processing until prediction
        #TODO: treat initialization as hyperparameter, but don't regularize weights
        layers.append(ScaleLayer(args, layers[-1], name='post_scale'))
        self.add_params_to_self(args, layers[-1])
        layers.append(BiasLayer(args, layers[-1], name='post_shift'))
        self.add_params_to_self(args, layers[-1])
        layers.append(NonlinearityLayer(layers[-1], nonlinearity=rectify,
                                    name='post_relu'))
        #TODO: noise here?
        layers.append(GlobalPoolLayer(layers[-1], name='post_pool'))
        #TODO: regularize
        layers.append(DenseLayer(args, layers[-1], args.classes, nonlinearity=softmax,
                             W=lasagne.init.HeNormal(gain=1), name='output'))
        self.add_params_to_self(args, layers[-1])
        self.layers = layers

        print(self.params_theta)
        print(self.params_weight)
        print(self.params_lambda)
        print(self.params_noise)

        #training time: deterministic=False
        self.y = ll.get_output(layers[-1], x, deterministic=False)
        self.prediction = T.argmax(self.y, axis=1)
        # cost function
        self.loss = T.mean(categorical_crossentropy(self.y, y))
        self.lossWithPenalty = T.add(self.loss, self.penalty)
        
        #validation time: deterministic=True
        self.y_det = ll.get_output(layers[-1], x, deterministic=True)
        self.prediction_det = T.argmax(self.y, axis=1)
        # cost function
        self.loss_det = T.mean(categorical_crossentropy(self.y_det, y))
        self.lossWithPenalty_det = T.add(self.loss_det, self.penalty)
        print("loss and losswithpenalty", type(self.loss), type(self.lossWithPenalty))



