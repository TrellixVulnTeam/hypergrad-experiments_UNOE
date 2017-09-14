import sys, os
sys.path.append(os.path.abspath('../../../'))



"""Runs for paper"""
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
import numpy as np
import pickle
from collections import defaultdict
from funkyyak import grad, kylist, getval

import hypergrad.mnist as mnist
from hypergrad.mnist import random_partition
from hypergrad.nn_utils import make_nn_funs, VectorParser
#from hypergrad.optimizers import sgd_meta_only_mad as sgd
from hypergrad.optimizers2 import sgd_meta_only_mad, sgd_meta_only, sgd_meta_only_mad2
from hypergrad.util import RandomState, dictslice, dictmap
from hypergrad.odyssey import omap

layer_sizes = [784, 50, 50, 50, 10]
N_layers = len(layer_sizes) - 1
batch_size = 50
#N_iters = 2000    # 5000
N_iters = 2000

N_train = 20000 # Actually will use N_train - N_valid?
N_valid = 5000
N_tests = 5000

all_N_meta_iter = [0, 0, 20] #universal, layers, units
alpha = 0.05  #0.1
meta_alpha = 0.07
beta = 0.95   # 0.1
seed = 0
N_thin = 500
N_meta_thin = 1
log_L2 = -4.0 #TODO: on top of other per-unit log_L2? 
log_init_scale = -3.0


def run():
    RS = RandomState((seed, "top_rs"))
    all_data = mnist.load_data_as_dict()
    train_data, tests_data = random_partition(all_data, RS, [N_train, N_tests])
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size

    def transform_weights(z_vect, transform): #TODO: isn't this a scale transformation?
        return z_vect * np.exp(transform)

    def regularization(z_vect):
        return np.dot(z_vect, z_vect) * np.exp(log_L2)

    def constrain_reg(t_vect, name):
        all_t = w_parser.new_vect(t_vect)
        for i in range(N_layers): #Don't regularize biases
            all_t[('biases', i)] = 0.0
        if name == 'universal': #One regularization hyperparameter for all weights
            #TODO: does computing means of means make sense? Not the same as just the mean of all.
            t_mean = np.mean([np.mean(all_t[('weights', i)])
                              for i in range(N_layers)])
            for i in range(N_layers):
                all_t[('weights', i)] = t_mean
        elif name == 'layers': #One regularization hyperparameter for each layer
            #TODO: changes the exact hypergradient norm, but not the DrMAD norm. Why??? DrMAD is already constrained?
            #print t_vect.shape
            for i in range(N_layers):
                #print "diff after contraining" + str(np.linalg.norm(all_t[('weights', i)] - np.mean(all_t[('weights', i)])))
                all_t[('weights', i)] = np.mean(all_t[('weights', i)])
        elif name == 'units':
            print t_vect.shape #44860; this is correct
            #for i in range(N_layers):
                #print "weights "+ str(i) + ": " + str(np.linalg.norm(np.mean(all_t[('weights', i)], axis=1, keepdims=True) - np.mean(all_t[('weights', i)], axis=1, keepdims=True)))
            #for i in range(N_layers):
                #TODO: This was the same as layer-wise
                #all_t[('weights', i)] = np.mean(all_t[('weights', i)], axis=1, keepdims=True)
        else:
            raise Exception
        return all_t.vect

    def process_transform(t_vect):
        # Remove the redundancy due to sharing transformations within units
        all_t = w_parser.new_vect(t_vect)
        new_t = np.zeros((0,))
        for i in range(N_layers):
            layer = all_t[('weights', i)]
            assert np.all(layer[:, 0] == layer[:, 1])
            cur_t = log_L2 - 2 * layer[:, 0] #TODO: equivalent regularization weights
            new_t = np.concatenate((new_t, cur_t))
        return new_t
        
    def train_z(data, z_vect_0, transform):
        N_data = data['X'].shape[0]
        def primal_loss(z_vect, transform, i_primal, record_results=False):
            RS = RandomState((seed, i_primal, "primal"))
            idxs = RS.randint(N_data, size=batch_size)
            minibatch = dictslice(data, idxs)
            w_vect = transform_weights(z_vect, transform) #TODO: this is a scale transformation, not regularization!
            loss = loss_fun(w_vect, **minibatch) #use new scale for prediction
            reg = regularization(z_vect) #regularize original scale
            #TODO: should be equivalent: w = z*e^transform, so 
            # f(z*e^transform) + e^\lambda||z||^2 = f(w) + e^\lambda||z||^2 = f(w) + e^(\lambda)||e^-2transform w||^2
            # see process_transform
            
            #if record_results and i_primal % N_thin == 0:
                #print "Iter {0}: train: {1}".format(i_primal, getval(loss))
            return loss + reg
        return sgd_meta_only_mad(grad(primal_loss), transform, z_vect_0, alpha, beta, N_iters)
    
    def train_z_exact(data, z_vect_0, transform, meta_iteration=0):
        N_data = data['X'].shape[0]
        def primal_loss(z_vect, transform, i_primal, record_results=False):
            RS = RandomState((seed, i_primal, "primal"))
            idxs = RS.randint(N_data, size=batch_size)
            minibatch = dictslice(data, idxs)
            w_vect = transform_weights(z_vect, transform)
            loss = loss_fun(w_vect, **minibatch)
            reg = regularization(z_vect)
            #if record_results and i_primal % N_thin == 0:
            #    print "Iter {0}: train: {1}".format(i_primal, getval(loss))
            return loss + reg
        return sgd_meta_only(grad(primal_loss), transform, z_vect_0, alpha, beta, N_iters, meta_iteration=meta_iteration)

    def train_z2(data, z_vect_0, transform):
        N_data = data['X'].shape[0]
        def primal_loss(z_vect, transform, i_primal, record_results=False):
            RS = RandomState((seed, i_primal, "primal"))
            idxs = RS.randint(N_data, size=batch_size)
            minibatch = dictslice(data, idxs)
            w_vect = transform_weights(z_vect, transform)
            loss = loss_fun(w_vect, **minibatch)
            reg = regularization(z_vect)
            return loss + reg
        return sgd_meta_only_mad2(grad(primal_loss), transform, z_vect_0, alpha, beta, N_iters)
    

    all_transforms, all_train_loss, all_valid_loss, all_tests_loss, all_train_rates, all_valid_rates, all_tests_rates, all_avg_regs, hypergrad_angles, hypergrad_angles2, hypergrad_signs_angles, hypergrad_signs_angles2, hypergrad_norms, hypergrad_norms2, exact_hypergrad_norms = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    def train_reg(reg_0, constraint, N_meta_iter, i_top):
        def hyperloss(transform, i_hyper, cur_train_data, cur_valid_data, cur_tests_data):
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            z_vect_0 = RS.randn(N_weights) * np.exp(log_init_scale)
            z_vect_final = train_z(cur_train_data, z_vect_0, transform)
            w_vect_final = transform_weights(z_vect_final, transform) #TODO: initial scale AND regularization
            
         
            train_loss = getval(loss_fun(w_vect_final, **cur_train_data))
            print "Training loss (unregularized) = " +str(train_loss)
            all_train_loss.append(train_loss)
            valid_loss = getval(loss_fun(w_vect_final, **cur_valid_data))
            print "Validation loss = " +str(valid_loss)
            all_valid_loss.append(valid_loss)
            tests_loss = getval(loss_fun(w_vect_final, **cur_tests_data))
            print "Test loss = " +str(tests_loss)
            all_tests_loss.append(tests_loss)
            
            """plt.plot(all_train_loss, label="training loss (unregularized)")
            plt.plot(all_valid_loss, label="validation loss")
            plt.plot(all_tests_loss, label="test loss")
            plt.title("loss vs meta iteration")
            plt.xlabel("meta iteration")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig("loss"+str(N_iters)+"_corrected.png")
            plt.clf()"""
            
            
            train_rate = getval(frac_err(w_vect_final, **cur_train_data))
            print "Training error rate = " +str(train_rate)
            all_train_rates.append(train_rate)
            valid_rate = getval(frac_err(w_vect_final, **cur_valid_data))
            print "Validation error rate = " +str(valid_rate)
            all_valid_rates.append(valid_rate)
            tests_rate = getval(frac_err(w_vect_final, **cur_tests_data))
            print "Test error rate = " +str(tests_rate)
            all_tests_rates.append(tests_rate)
            
            """plt.plot(all_train_rates, label="training error rate")
            plt.plot(all_valid_rates, label="validation error rate")
            plt.plot(all_tests_rates, label="test error rate")
            plt.title("error rate vs meta iteration")
            plt.xlabel("meta iteration")
            plt.ylabel("error rate")
            plt.legend()
            plt.savefig("error"+str(N_iters)+"_corrected.png")
            plt.clf()"""

            
            return loss_fun(w_vect_final, **cur_valid_data)
        hypergrad = grad(hyperloss) #No chain rule here

        def hyperloss_exact(transform, i_hyper, cur_train_data, cur_valid_data, cur_tests_data, meta_it=0):
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            z_vect_0 = RS.randn(N_weights) * np.exp(log_init_scale)
            z_vect_final = train_z_exact(cur_train_data, z_vect_0, transform, meta_iteration=meta_it)
            w_vect_final = transform_weights(z_vect_final, transform)
            return loss_fun(w_vect_final, **cur_valid_data)
        hypergrad_exact = grad(hyperloss_exact) #No chain rule here
        
        def hyperloss2(transform, i_hyper, cur_train_data, cur_valid_data, cur_tests_data):
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            z_vect_0 = RS.randn(N_weights) * np.exp(log_init_scale)
            z_vect_final = train_z2(cur_train_data, z_vect_0, transform)
            w_vect_final = transform_weights(z_vect_final, transform)
            return loss_fun(w_vect_final, **cur_valid_data)
        hypergrad2 = grad(hyperloss2)
            
        '''def error_rate(transform, i_hyper, cur_train_data, cur_valid_data):
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            z_vect_0 = RS.randn(N_weights) * np.exp(log_init_scale)
            z_vect_final = train_z(cur_train_data, z_vect_0, transform) #TODO: recomputing path?
            w_vect_final = transform_weights(z_vect_final, transform)
            return frac_err(w_vect_final, **cur_valid_data)'''

        cur_reg = reg_0 #initial regularization, besides regularization() function
        for i_hyper in range(N_meta_iter):
            print "Hyper iter "+ str(i_hyper)
            """if i_hyper % N_meta_thin == 0:
                test_rate = error_rate(cur_reg, i_hyper, train_data, tests_data)
                all_tests_rates.append(test_rate)
                all_transforms.append(cur_reg.copy())
                all_avg_regs.append(np.mean(cur_reg))
                print "Hyper iter {0}, error rate {1}".format(i_hyper, all_tests_rates[-1])
                print "Cur_transform", np.mean(cur_reg)"""
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            #cur_split = random_partition(train_data, RS, [N_train - N_valid, N_valid]) #cur_train_data, cur_valid_data
            #raw_grad = hypergrad(cur_reg, i_hyper, *cur_split)
            cur_train_data, cur_valid_data = random_partition(train_data, RS, [N_train - N_valid, N_valid])
            raw_grad = hypergrad(cur_reg, i_hyper, cur_train_data, cur_valid_data, tests_data)
            raw_grad2 = hypergrad2(cur_reg, i_hyper, cur_train_data, cur_valid_data, tests_data)
            raw_grad_exact = hypergrad_exact(cur_reg, i_hyper, cur_train_data, cur_valid_data, tests_data, meta_it=i_hyper)
            #print "before constraining grad"
            constrained_grad = constrain_reg(raw_grad, constraint)
            constrained_grad2 = constrain_reg(raw_grad2, constraint)
            constrained_grad_exact = constrain_reg(raw_grad_exact, constraint)
            print(np.linalg.norm(raw_grad))
            print(np.linalg.norm(raw_grad2))
            #TODO: #Exploding DrMAD gradient; ~10^10x larger than exact gradient with N_safe_sampling = N_iters
            print(np.linalg.norm(raw_grad_exact))
            # TODO: sometimes negative???
            
            
            hypergrad_angle = np.dot(constrained_grad, constrained_grad_exact)/(np.linalg.norm(constrained_grad)*np.linalg.norm(constrained_grad_exact))
            hypergrad_angles.append(hypergrad_angle)
            hypergrad_angle2 = np.dot(constrained_grad2, constrained_grad_exact)/(np.linalg.norm(constrained_grad2)*np.linalg.norm(constrained_grad_exact))
            hypergrad_angles2.append(hypergrad_angle2)
            print("cosine of angle between DrMAD and exact = " +str(hypergrad_angle))
            print("cosine of angle between DrMAD2 and exact = " +str(hypergrad_angle2))
            
            hypergrad_signs_angle = np.dot(np.sign(constrained_grad), np.sign(constrained_grad_exact))/len(constrained_grad)
            hypergrad_signs_angles.append(hypergrad_signs_angle)
            print("cosine of angle between signs of DrMAD and exact = " +str(hypergrad_signs_angle))
            hypergrad_signs_angle2 = np.dot(np.sign(constrained_grad2), np.sign(constrained_grad_exact))/len(constrained_grad2)
            hypergrad_signs_angles2.append(hypergrad_signs_angle2)
            print("cosine of angle between signs of DrMAD2 and exact = " +str(hypergrad_signs_angle2))
            
            
            """plt.plot(hypergrad_angles, label="exact vs DrMAD")
            plt.plot(hypergrad_signs_angles, label="signs exact vs signs DrMAD")
            plt.plot(hypergrad_angles2, label="exact vs DrMAD2")
            plt.plot(hypergrad_signs_angles2, label="signs exact vs signs DrMAD2")
            plt.title("Cosine of angle between hypergradients vs meta iteration")
            plt.xlabel("meta iteration")
            plt.ylabel("cosine of angle")
            plt.legend()
            plt.savefig("angle"+str(N_iters)+"_corrected2.png")
            plt.clf()"""
            
            hypergrad_norm = np.linalg.norm(constrained_grad)
            hypergrad_norms.append(hypergrad_norm)
            print("DrMAD norm = "+ str(hypergrad_norm))
            hypergrad_norm2 = np.linalg.norm(constrained_grad2)
            hypergrad_norms2.append(hypergrad_norm2)
            print("DrMAD2 norm = "+ str(hypergrad_norm2))
            exact_hypergrad_norm = np.linalg.norm(constrained_grad_exact)
            exact_hypergrad_norms.append(exact_hypergrad_norm)
            print("Exact norm = "+ str(exact_hypergrad_norm))
            
            """plt.plot(hypergrad_norms, label="DrMAD hypergradient")
            plt.plot(hypergrad_norms2, label="DrMAD2 hypergradient")
            plt.plot(exact_hypergrad_norms, label="Exact hypergradient")
            plt.title("Norms of hypergradients vs meta iteration")
            plt.xlabel("meta iteration")
            plt.ylabel("norm")
            plt.legend()
            plt.savefig("norms"+str(N_iters)+"_corrected2.png")
            plt.clf()"""
            
            cur_reg -= np.sign(constrained_grad) * meta_alpha #TODO: signs of gradient...
            #TODO: momentum
        return cur_reg

    reg = np.zeros(N_weights)+0.2 #TODO: initial -log regularization; not in log scale?
    constraints = ['universal', 'layers', 'units']
    # TODO: uses multiple kinds of hyperparameter sharing, but in order
    for i_top, (N_meta_iter, constraint) in enumerate(zip(all_N_meta_iter, constraints)):
        print "Top level iter {0}".format(i_top), constraint
        reg = train_reg(reg, constraint, N_meta_iter, i_top)

    all_L2_regs = np.array(zip(*map(process_transform, all_transforms)))
    #return all_L2_regs, all_tests_rates, all_avg_regs
    all_L2_regs, all_train_loss, all_valid_loss, all_tests_loss, all_train_rates, all_valid_rates, all_tests_rates, all_avg_regs, hypergrad_angles, hypergrad_signs_angles, hypergrad_norms, exact_hypergrad_norms

def plot():
    import matplotlib.pyplot as plt
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['image.interpolation'] = 'none'
    with open('results.pkl') as f:
        #all_L2_regs, all_tests_rates, all_avg_regs = pickle.load(f)
        all_L2_regs, all_train_loss, all_valid_loss, all_tests_loss, all_train_rates, all_valid_rates, all_tests_rates, all_avg_regs, hypergrad_angles, hypergrad_signs_angles, hypergrad_norms, exact_hypergrad_norms = pickle.load(f)

    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(211)
    color_cycle = ['RoyalBlue', 'DarkOliveGreen', 'DarkOrange', 'MidnightBlue']
    colors = []
    for i, size in enumerate(layer_sizes[:-1]):
        colors += [color_cycle[i]] * size
    for c, L2_reg_curve in zip(colors, all_L2_regs):
        ax.plot(L2_reg_curve, color=c)
    ax.set_ylabel('Log L2 regularization')
    # ax.set_ylim([-3.0, -2.5])

    ax = fig.add_subplot(212)
    ax.plot(all_avg_regs)
    ax.set_ylabel('Average log L2 regularization')

    ax = fig.add_subplot(213)
    ax.plot(all_tests_rates)
    ax.set_ylabel('Test error (%)')
    ax.set_xlabel('Meta iterations')
    plt.savefig("reg_learning_curve.eps", format='eps', dpi=1000)

    initial_filter = np.array(all_L2_regs)[:layer_sizes[0], -1].reshape((28, 28))
    fig.clf()
    fig.set_size_inches((5, 5))
    ax = fig.add_subplot(111)
    ax.matshow(initial_filter, cmap = mpl.cm.binary)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('bottom_layer_filter.eps', format='eps', dpi=1000)

if __name__ == '__main__':
    import time
    t0 = time.time()
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    t1 = time.time()

    total = t1 - t0
    print(total)
    plot()