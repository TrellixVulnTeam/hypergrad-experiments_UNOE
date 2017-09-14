"""
    This training script is just a practice on CIFAR10 using Theano.

"""

#Michael: Python 3 compatibility
from __future__ import print_function #use parentheses for print
from __future__ import division #/ is floating point division, // rounds to an integer


from time import time
import theano
import theano.tensor as T
import numpy as np
#from preprocess.read_preprocess import read_preprocess #Michael: for GPU
#from preprocess.read_preprocess_old import read_preprocess #Michael: use original preprocess version
from cifar10 import load_dataset
from args import setup
#from models import MLP, ConvNet
from densenet_fast import DenseNet
from updates import update, updates_hyper
from random import sample as random_sample, seed as random_seed
import time as TIME
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict

theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano.config.optimizer='fast_compile'







"""
Set up the network
"""


args = setup()
print('all argument: ', args)
temp_lambda = None
loss_change = []
tmp_weights = None
random_seed(args.seed)
    

if args.model == 'convnet':
    x = T.ftensor4('x')
elif args.model == 'mlp':
    x = T.matrix('x')
else:
    raise AttributeError
y = T.matrix('y')
lr_ele = T.fscalar('lr_ele')

lr_ele_true = np.array(args.lrEle, theano.config.floatX)
mom = args.momEle #momentum
lr_hyper = T.fscalar('lr_hyper')
grad_valid_weight = T.tensor4('grad_valid_weight')


model = DenseNet(x=x, y=y, args=args)
#model = ConvNet(x=x, y=y, args=args)

velocities = [theano.shared(np.asarray(param.get_value(borrow=True)*0., dtype=theano.config.floatX), broadcastable=param.broadcastable, name=param.name+'_vel') for param in model.params_theta]
lambda_velocities = [theano.shared(np.asarray(lamb.get_value(borrow=True)*0., dtype=theano.config.floatX), broadcastable=lamb.broadcastable, name=lamb.name+'_vel') for lamb in model.params_lambda]
momHyper = args.momHyper
momLlr = args.momLlr




X_elementary, Y_elementary, X_test, Y_test = load_dataset(args) #normalized
#Use a large validation set (as in CPU experiments) to avoid overfitting the hyperparameters
X_hyper = X_elementary[0:5000]
Y_hyper = Y_elementary[0:5000]
X_elementary = X_elementary[5000:]
Y_elementary = Y_elementary[5000:]


# Phase 1
# elementary SGD variable update list (constant momenta and learning rates)

#this uses Nesterov momentum and a common learning rate for all elementary parameters
update_ele, update_valid, output_valid_list, share_var_dloss_dweight = update(model.params_theta, model.params_lambda, model.params_weight,
                                  velocities, model.loss, model.penalty, model.lossWithPenalty,
                                  lr_ele, mom, args.eleAlg)
                                  
#update_ele, update_valid, output_valid_list, share_var_dloss_dweight = update2(model.params_theta, model.params_lambda, model.params_weight,
#                                  velocities, model.loss, model.penalty, model.lossWithPenalty,
#                                  log_learning_rates, lr_hyper, mom)
#output_valid_list is used as dv_t in DrMAD


func_elementary = theano.function(
    inputs=[x, y, lr_ele], 
    #inputs=[x, y],
    outputs=[model.lossWithPenalty, model.loss, model.prediction],
    updates=update_ele, #Michael: update_ele will apply the SGD step when func_elementary is called
    on_unused_input='ignore',
    allow_input_downcast=True)

func_eval = theano.function(
    inputs=[x, y],
    outputs=[model.loss_det, model.prediction_det], #use deterministic=True for these
    on_unused_input='ignore',
    allow_input_downcast=True)
    
func_eval_train = theano.function(
    inputs=[x, y],
    outputs=[model.lossWithPenalty_det, model.loss_det, model.prediction_det], #use deterministic=True for these
    on_unused_input='ignore',
    allow_input_downcast=True)

# Phase 2
func_hyper_valid = theano.function(
    inputs=[x, y],
    outputs=[model.loss_det, model.prediction_det],# + output_valid_list,
    updates=update_valid, #this updates output_valid_list to dloss_dweight = T.grad(model.loss, model.params_weight)
    on_unused_input='ignore',
    allow_input_downcast=True)
    #output_valid_list in the outputs returns an array corresponding to the previous value
    # not what output_valid_list should be updated to by update_valid
    # so just use .get_value() on output_valid_list's elements, instead



# Phase 3
#output_valid_list here should be dv_t
#phase_3_input is supposed to be the same variable as output_valid_list
#model.params_weight instead of model.params_theta because penalty does not depend on biases, and
#and we want initial biases to *always* be 0
#update_hyper, output_hyper_list, phase_3_input = updates_hyper(model.params_lambda, model.params_weight,
#                                                model.lossWithPenalty, grad_l_theta, output_valid_list)
# These are the Theano shared, without values!
# updates for phase 3
#removed update_hyper, 
update_hyper, HVP_weight, HVP_lambda, phase_3_input = updates_hyper(model.params_lambda, model.params_weight,
                                                model.lossWithPenalty, output_valid_list)
print('output_valid_list',output_valid_list is phase_3_input)

#Checks
#print("weight", len(HVP_weight_temp), len(model.params_weight))
#print("lambda", len(HVP_lambda_temp), len(model.params_lambda))
#for i in range(len(update_hyper)):
#    if not update_hyper[i][0].type == update_hyper[i][1].type:
#        print(i,update_hyper[i], update_hyper[i][0].type, update_hyper[i][1].type)

# Phase 3
# dloss_dpenalty = T.grad(model.loss, model.params_lambda)

func_hyper_t0 = time()

#TODO: this is what's slow; can take several hours
func_hyper = theano.function(
    inputs=[x, y],
    #outputs=output_hyper_list + output_valid_list, #HVP_value
    #outputs = HVP_weight_temp+HVP_lambda_temp+output_valid_list,
    updates=update_hyper, #TODO
    on_unused_input='ignore',
    allow_input_downcast=True)
# will just access and update the stored variables in HVP_weight and HVP_lambda

print("func_hyper defined, taking "+str(time()-func_hyper_t0)+"s")
#On my laptop's CPU, "func_hyper defined, taking 354.448412895s" ~ 6 minutes, or 178.747463942s with FAST_COMPILE
#On school GPU with FAST_COMPILE "func_hyper defined, taking 495.1831388473511s" ~ 8min

#below will be lists of lists. 
# each list in training_errors, corresponding to a meta-iteration, will contain the training error at the end of each epoch, but also before training
# 0 for beginning of training, k+1 for end of epoch k
#TODO: use random batches for errors/losses, not the same batches predictably (i.e. first n_eval)

train_errors = []
valid_errors = []
#hyper_errors = []
test_errors = []

unreg_train_losses = []
train_losses = []
valid_losses = []
#hyper_losses = []
test_losses = []

norms = []
theta_names = [param.name for param in model.params_theta]


#update_lambda, fix_weight = temp_lambda, tmp_weights
#update_llrs = temp_llrs



#make a directory with a timestamp name, and save results to it
import os
print("---------------------------------------------------------------------------------------------------")
print('Reproduce_{:%Y-%b-%d_%Hh%Mm%Ss}'.format(datetime.datetime.now()))
print("---------------------------------------------------------------------------------------------------")
outdir = 'Reproduce_{:%Y-%b-%d_%Hh%Mm%Ss}'.format(datetime.datetime.now())
os.mkdir(outdir)

#copy files to directory in case of changes
import shutil
shutil.copy2(os.path.basename(__file__), outdir) #simple_mlp2_gpu.py
shutil.copy2('args.py', outdir)
shutil.copy2('updates.py', outdir)
shutil.copy2('layers.py', outdir)
shutil.copy2('densenet_fast.py', outdir)




def run_exp(args, update_lambda, fix_weight, epochs):
#def run_exp(args, update_lambda, update_llrs, fix_weight, epochs):
    #shuffle training and validation sets? But keep test data the same and separate
    #Want to break symmetry in a consistent way and hyperparameters match elementary parameters, so don't shuffle?
    #global X_elementary, Y_elementary, X_hyper, Y_hyper, X_valid, Y_valid, X_test, Y_test
    global X_elementary, Y_elementary, X_hyper, Y_hyper, X_test, Y_test
    global training_errors, unregularized_training_errors, valid_errors, hyper_errors, test_errors, training_losses, valid_losses, hyper_losses, test_losses, norms, theta_names

    #update the hyperparameters
    if update_lambda:
        for up, origin in zip(update_lambda, model.params_lambda):
            origin.set_value(np.array(up))
            # boo = origin.get_value()
            # print 'update', type(up), type(boo), boo[1]

    #reinitialize the elementary parameters (exactly as before; break symmetry the same way again)
    if fix_weight:
        #for fix, origin in zip(fix_weight, model.params_weight):
         #   origin.set_value(np.array(fix))
        for fix, origin, velocity in zip(fix_weight, model.params_theta, velocities):
            origin.set_value(np.array(fix))
            velocity.set_value(velocity.get_value()*0.) #reset velocities to 0
    else: #store the elementary parameters for the first time, as arrays
    #TODO: this could be put outside of run_exp()
        fix_weight = []
        for origin in model.params_theta:
            fix_weight.append(origin.get_value(borrow=False)) #copies


    #Phase 1






    """
         Phase 1: meta-forward

    """

    #n_ele, n_valid, n_test = X_elementary.shape[0], X_valid.shape[0], X_test.shape[0]
    n_ele, n_hyper, n_test = X_elementary.shape[0], X_hyper.shape[0], X_test.shape[0]
    
    print("# of epochs: " + str(epochs))
    print("# of ele, valid, test: ", n_ele, n_hyper, n_test)
    #number of batches, or number of iterations per epoch
    n_batch_ele = n_ele // args.batchSizeEle #integer division, so some data may be lost
    #n_batch_ele = 100 #TODO: 
    print("# of ele batches ="+ str(n_batch_ele))
    test_perm, ele_perm, hyper_perm = range(0, n_test), range(0, n_ele), range(0, n_hyper)
    last_iter = epochs * n_batch_ele - 1
    iter_index_cache = []
    # save the model parameters into theta_initial
    theta_initial = fix_weight
    #TODO: checkpoints
    checkpoints = OrderedDict()
    checkpoints[0] = theta_initial
    dist_from_checkpoints = OrderedDict()
    dist_from_checkpoints[0] = []
    dist_from_checkpoints[1] = []
    dist_from_checkpoints[epochs-1] = []
    
    n_eval = 1000 #sample size from training/elementary, hyper/valid and test sets to evaluate on
    
    
    #use func_eval on initial parameters for elementary, valid/hyper and test sets, starting new lists in each
    temp_idx = ele_perm[:n_eval]
    batch_x, batch_y = X_elementary[temp_idx], Y_elementary[temp_idx]
    tmp_y = np.zeros((n_eval, 10))
    for idx, element in enumerate(batch_y):
        tmp_y[idx][element] = 1
    batch_y = tmp_y
    train_loss, unreg_train_loss, train_pred = func_eval_train(batch_x, batch_y)

    wrong = 0
    for e1, e2 in zip(train_pred, Y_elementary[temp_idx]):
        if e1 != e2:
            wrong += 1
    train_error = 100. * wrong / n_eval
    print("Eval on Train Set: unreg_loss = %.4f, loss = %.4f, error = %.4f" %
          (unreg_train_loss, train_loss, train_error))
    unreg_train_losses.append([float(unreg_train_loss)])
    train_losses.append([float(train_loss)])
    train_errors.append([train_error])

    temp_idx = hyper_perm[:n_eval]
    batch_x, batch_y = X_hyper[temp_idx], Y_hyper[temp_idx]
    tmp_y = np.zeros((n_eval, 10))
    for idx, element in enumerate(batch_y):
        tmp_y[idx][element] = 1
    batch_y = tmp_y
    valid_loss, valid_pred = func_eval(batch_x, batch_y)

    wrong = 0
    for e1, e2 in zip(valid_pred, Y_hyper[temp_idx]):
        if e1 != e2:
            wrong += 1
    valid_error = 100. * wrong / n_eval
    print("Eval on Valid Set: loss = %.4f, error = %.4f" %
          (valid_loss, valid_error))
    valid_losses.append([float(valid_loss)])
    valid_errors.append([valid_error])


    temp_idx = test_perm[:n_eval]
    batch_x, batch_y = X_test[temp_idx], Y_test[temp_idx]
    tmp_y = np.zeros((n_eval, 10))
    for idx, element in enumerate(batch_y):
        tmp_y[idx][element] = 1
    batch_y = tmp_y
    test_loss, test_pred = func_eval(batch_x, batch_y)

    wrong = 0
    for e1, e2 in zip(test_pred, Y_test[temp_idx]):
        if e1 != e2:
            wrong += 1
    # eval_error = 1. * sum(int(Y_test[temp_idx] != batch_y)) / n_eval
    test_error = 100. * wrong / n_eval
    print("Eval on Test Set: loss = %.4f, error = %.4f" %
          (test_loss, test_error))
    test_losses.append([float(test_loss)])
    test_errors.append([test_error])
    
    for param in model.params_theta:
        norms.append([np.linalg.norm(param.get_value(borrow=True))])
    
    #Save values and plots
    save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses,
        train_errors, valid_errors, test_errors, norms, theta_names)
    
    
    
    lr_ele_true = args.lrEle
    
    #dist_from_theta0 = []
    
    t_start = time()
    
    #TODO: checkpoints for first and last epochs? Should they correspond to learning rate schedule?
    

    def learning_rate(current_epoch, lrEleInit=args.lrEle): #TODO:
        if current_epoch >= 60:
            return lrEleInit/100.
        elif current_epoch >= 30:
            return lrEleInit/10.
        else:
            return lrEleInit
    
    

    
    
    T = epochs * n_batch_ele
    #T = n_batch_ele//3  #TODO: change back
    #T = 20
    #curr_epoch = 0
    for i in range(0, T): #SGD steps
        #prev_epoch = curr_epoch
        curr_epoch = i // n_batch_ele
        curr_batch = i % n_batch_ele
        
        

        """
            Learning rate and momentum schedules.

        """
        
        #From DenseNet
        
        if curr_epoch >= 30:
            lr_ele_true = args.lrEle/10.
        elif curr_epoch >= 60:
            lr_ele_true = args.lrEle/100.
        
        

        """
            Update

        """
        
        sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)] #Michael: batch indices
        iter_index_cache.append(sample_idx_ele)
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele] #batch data
        tmp_y = np.zeros((args.batchSizeEle, 10)) #10 for 10 classes; put a 1 in row=idx and column=class=element of idx 
        for idx, element in enumerate(batch_y): #idx = index, element = element at that index
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        
        #update elementary parameters
        train_loss, unreg_train_loss, train_pred = func_elementary(batch_x, batch_y, lr_ele_true)

        #Checkpoints
        #This code can be made more general
        if i > 3 and i < min(20,T): #TODO: first epoch
            dist_from_checkpoints[0].append(sum([np.linalg.norm(param.get_value(borrow=True)-param0) for param, param0 in zip(model.params_theta, checkpoints[0])])/(i+1) )
            #dist_from_theta0.append(
            #sum([np.linalg.norm(param.get_value(borrow=True)-param0) for param, param0 in zip(model.params_theta, theta_initial)])/(i+1) )
        if curr_epoch == 1 and curr_batch==0:
            checkpoints[1] = [model.params_theta.get_value(borrow=False)]
        if curr_epoch == 1 and curr_batch>0 and curr_batch<100: #use only the first 100 of the epoch
            dist_from_checkpoints[1].append(sum([np.linalg.norm(param.get_value(borrow=True)-param0) for param, param0 in zip(model.params_theta, checkpoints[curr_epoch])])/curr_batch )
        if curr_epoch == epochs-1 and curr_batch==n_batch_ele - 100:
            checkpoints[curr_epoch] = [model.params_theta.get_value(borrow=False)]
        if curr_epoch == epochs-1 and curr_batch > n_batch_ele - 100: #use last 100
            dist_from_checkpoints[epochs-1].append(sum([np.linalg.norm(param.get_value(borrow=True)-param0) for param, param0 in zip(model.params_theta, checkpoints[curr_epoch])])/(curr_batch - n_batch_ele + 100))
        

        """
            Evaluate

        """
        
        if i%20==0:
            wrong = 0
            for e1, e2 in zip(train_pred, Y_elementary[sample_idx_ele]): #
                if e1 != e2:
                    wrong += 1
            train_error = 100. * wrong / len(train_pred)
            print("Train Set: Epoch %d, batch %d, time = %ds, loss = %.4f, unreg loss = %.4f, error = %.4f" %
                  (curr_epoch, curr_batch, time() - t_start, train_loss, unreg_train_loss, train_error))
        
        
        #if args.verbose or (curr_batch == n_batch_ele - 1): #verbose for each iteration, batch for epoch
        # use sample of 1000 to evaluate training, validation and tests losses/errors
        if curr_batch == n_batch_ele - 1: #last batch
            #use deterministic option for BN layers
            if curr_epoch == epochs-1: #if it's also the last epoch, do a bigger test
                n_eval = 5000
            else:
                n_eval = 1000
        
            #temp_idx = ele_perm[:n_eval] #same 1000 each time?
            n_ele_eval_batches = n_ele // n_eval
            temp_idx = ele_perm[(curr_epoch % n_ele_eval_batches)*n_eval: ((curr_epoch % n_ele_eval_batches)+1)*n_eval]
            batch_x, batch_y = X_elementary[temp_idx], Y_elementary[temp_idx]
            tmp_y = np.zeros((n_eval, 10))
            for idx, element in enumerate(batch_y):
                tmp_y[idx][element] = 1
            batch_y = tmp_y
            train_loss, unreg_train_loss, train_pred = func_eval_train(batch_x, batch_y)

            wrong = 0
            for e1, e2 in zip(train_pred, Y_elementary[temp_idx]):
                if e1 != e2:
                    wrong += 1
            train_error = 100. * wrong / n_eval
            print("Eval on Train Set: Epoch %d, batch %d, time = %ds, loss = %.4f, unreg loss = %.4f, error = %.4f" %
                  (curr_epoch, curr_batch, time() - t_start, train_loss, unreg_train_loss, train_error))
            unreg_train_losses[-1].append(float(unreg_train_loss))
            train_losses[-1].append(float(train_loss))
            train_errors[-1].append(train_error)


            #temp_idx = hyper_perm[:n_eval] #same 1000 each time?
            n_hyper_eval_batches = n_hyper // n_eval
            temp_idx = hyper_perm[(curr_epoch % n_hyper_eval_batches)*n_eval: ((curr_epoch % n_hyper_eval_batches)+1)*n_eval]
            batch_x, batch_y = X_hyper[temp_idx], Y_hyper[temp_idx]
            tmp_y = np.zeros((n_eval, 10))
            for idx, element in enumerate(batch_y):
                tmp_y[idx][element] = 1
            batch_y = tmp_y
            valid_loss, valid_pred = func_eval(batch_x, batch_y)

            wrong = 0
            for e1, e2 in zip(valid_pred, Y_hyper[temp_idx]):
                if e1 != e2:
                    wrong += 1
            valid_error = 100. * wrong / n_eval
            print("Eval on Valid Set: Epoch %d, batch %d, time = %ds, loss = %.4f, error = %.4f" %
                  (curr_epoch, curr_batch, time() - t_start, valid_loss, valid_error))
            valid_losses[-1].append(float(valid_loss))
            valid_errors[-1].append(valid_error)



            #temp_idx = test_perm[:n_eval] #same 1000 each time?
            n_test_eval_batches = n_test // n_eval
            temp_idx = test_perm[(curr_epoch % n_test_eval_batches)*n_eval: ((curr_epoch % n_test_eval_batches)+1)*n_eval]
            batch_x, batch_y = X_test[temp_idx], Y_test[temp_idx]
            tmp_y = np.zeros((n_eval, 10))
            for idx, element in enumerate(batch_y):
                tmp_y[idx][element] = 1
            batch_y = tmp_y
            test_loss, test_pred = func_eval(batch_x, batch_y)

            wrong = 0
            for e1, e2 in zip(test_pred, Y_test[temp_idx]):
                if e1 != e2:
                    wrong += 1
            # eval_error = 1. * sum(int(Y_test[temp_idx] != batch_y)) / n_eval
            test_error = 100. * wrong / n_eval
            print("Eval on Test Set: Epoch %d, batch %d, time = %ds, loss = %.4f, error = %.4f" %
                  (curr_epoch, curr_batch, time() - t_start, test_loss, test_error))
            test_losses[-1].append(float(test_loss))
            test_errors[-1].append(test_error)
            
            for norm, param in zip(norms, model.params_theta):
                norm.append(np.linalg.norm(param.get_value(borrow=True)))
            #TODO: separate by meta-iter
            #            for k, param in enumerate(model.params_theta):
            #    norms[k+len(model.params_theta)*meta_iter].append(np.linalg.norm(param.get_value(borrow=True)))
            
            
            #Save and save plots
            save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses,
                train_errors, valid_errors, test_errors, norms, theta_names)


    # save the model parameters after T1 into theta_final
    theta_final = []
    for w in model.params_theta:
        theta_final.append(w.get_value())




    t_phase2 = time()
    """
        Phase 2: Validation on Hyper set and computation of dw_T = grad_w f(w_T), for f the validation loss

    """
    n_hyper = X_hyper.shape[0]
    n_batch_hyper = n_hyper // args.batchSizeHyper #5000/64 ~ 78
    hyper_perm = range(0, n_hyper)
    n_batch_samples_hyper = n_batch_hyper // 5
    # np.random.shuffle(hyper_perm)

    err_valid = 0.
    cost_valid = 0.
    t_start = time()
    grad_l_theta = [] #= dw_T = gradient of validation loss wrt final weights
    #for i in range(0, n_batch_hyper):
    # random sample without replacement
    # Or could just cycle based on meta-iter, 
    # but because we're only doing a few meta-iters, better to get closer to SGD and avoid periodic effects
    for j, i in enumerate(random_sample(range(n_batch_hyper), k=n_batch_samples_hyper)): #~15 batches covering about 5000/5 = 1000
        sample_idx = hyper_perm[(i * args.batchSizeHyper):((i + 1) * args.batchSizeHyper)]
        batch_x, batch_y = X_hyper[sample_idx], Y_hyper[sample_idx]
        # TODO: refactor, too slow
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        #res = func_hyper_valid(batch_x, batch_y)
        #valid_cost, pred_hyper, grad_temp = res[0], res[1], res[2:]
        valid_cost, pred_hyper = func_hyper_valid(batch_x, batch_y)  #updates output_valid_list to gradient
        err_tmp = 1. * sum(np.argmax(batch_y, axis=1) != pred_hyper) / args.batchSizeHyper
        err_valid += err_tmp
        # print "err_temp", err_tmp
        cost_valid += valid_cost

        # accumulate gradient and then take the average
        if j == 0: #instead of the randomly chosen i
            #for grad, meh in zip(grad_temp, output_valid_list):
            #    print(np.linalg.norm(grad), np.linalg.norm(meh.get_value()))
            for grad in output_valid_list:
                grad_l_theta.append(grad.get_value(borrow=False)) 
        else:
            for k, grad in enumerate(output_valid_list):
                grad_l_theta[k] += grad.get_value(borrow=False)
    err_valid /= (n_batch_samples_hyper)
    cost_valid /= (n_batch_samples_hyper)

    # get average grad of all iterations on validation set
    for i, grad in enumerate(grad_l_theta):
        #print("grad_l_theta_norm", np.linalg.norm(grad_l_theta[i]))
        grad_l_theta[i] = grad / (np.array((n_batch_samples_hyper) * 1., dtype=theano.config.floatX))
        #print("grad_l_theta_norm", np.linalg.norm(grad_l_theta[i]))


    print("Valid on Hyper Set: time = %ds, valid_err = %.2f, valid_loss = %.4f" %
          (time() - t_start, err_valid * 100, cost_valid))

    print("Phase 2 duration="+str(time()-t_phase2)+"s")
    
    t_phase3 = time()

    """
        Phase 3: meta-backward

    """

    print('Phase 3: Reversing path and computing hypergradient')
    

    """
    #output_valid_list here should be dv_t
    #phase_3_input is supposed to be the same variable as output_valid_list?
    #model.params_weight instead of model.params_theta because penalty does not depend on biases, and
    #and we want initial biases to *always* be 0
    #update_hyper, output_hyper_list, phase_3_input = updates_hyper(model.params_lambda, model.params_weight,
    #                                                model.lossWithPenalty, grad_l_theta, output_valid_list)
    # These are the Theano shared, without values!
    # updates for phase 3
    update_hyper, HVP_weight_temp, HVP_lambda_temp, phase_3_input = updates_hyper(model.params_lambda, model.params_weight,
                                                    model.lossWithPenalty, output_valid_list)
    print('output_valid_list',output_valid_list is phase_3_input)
    
    #Checks
    #print("weight", len(HVP_weight_temp), len(model.params_weight))
    #print("lambda", len(HVP_lambda_temp), len(model.params_lambda))
    #for i in range(len(update_hyper)):
    #    if not update_hyper[i][0].type == update_hyper[i][1].type:
    #        print(i,update_hyper[i], update_hyper[i][0].type, update_hyper[i][1].type)

    # Phase 3
    # dloss_dpenalty = T.grad(model.loss, model.params_lambda)
    
    #this is what's slow
    func_hyper = theano.function(
        inputs=[x, y],
        #outputs=output_hyper_list + output_valid_list, #HVP_value
        outputs = HVP_weight_temp+HVP_lambda_temp+output_valid_list,
        updates=update_hyper,
        on_unused_input='ignore',
        allow_input_downcast=True)
    """

    # initialization
    up_lambda, up_v = [], []
    for param in model.params_lambda:
        temp_param = np.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX) #np.zeros_like(param.get_value() * 0., dtype=theano.config.floatX)
        up_lambda += [temp_param]

    for param in model.params_weight:
        temp_v = np.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX) #np.zeros_like(param.get_value() * 0., dtype=theano.config.floatX)
        up_v += [temp_v]

    up_theta = grad_l_theta #dw_0
    #actually only weights from model.params_weight in update

    #print("Phase 3 setup duration="+str(time()-t_phase3))
    
    #calculate number of reverse iterations
    
    #t_phase3_pass = time()
    n_backwards = OrderedDict()
    """#2nd gets 2^-2, 3rd gets 2^-3, 4th gets 2^-4 so on; first gets what's left
    weights = [0.]+[2.**-(i+1) for i in range(1,len(dist_from_theta0))]
    weights[0] = 1.-sum(weights[1:]) #there might be a closed form here
    beta = 0.
    for i in range(0,len(weights)):
        beta = beta + weights[i]*dist_from_theta0[i]
    beta = beta/sum([np.linalg.norm(paramT-param0) for paramT, param0 in zip(theta_final, theta_initial)])
    n_backward = int(np.ceil(args.nBackwardMult/beta)) #max(N,int(np.ceil(1./beta)))
    #print("weights", weights)
    #print("dist_from_theta0", dist_from_theta0)
    
    print("n_backward", n_backward)
    n_backward = min(n_backward, 600, T)"""
    
    #to first checkpoint
    #2nd gets 2^-2, 3rd gets 2^-3, 4th gets 2^-4 so on; first gets what's left
    weights = [0.]+[2.**-(i+1) for i in range(1,len(dist_from_checkpoints[0]))]
    weights[0] = 1.-sum(weights[1:]) #there might be a closed form here
    beta = 0.
    for i in range(0,len(weights)):
        beta = beta + weights[i]*dist_from_checkpoints[0][i]
    beta = beta/sum([np.linalg.norm(paramT-param0) for paramT, param0 in zip(checkpoints[1], checkpoints[0])])
    n_backwards[0] = int(np.ceil(args.nBackwardMult/beta)) #max(N,int(np.ceil(1./beta)))
    print("n_backwards[0]", n_backwards[0])
    n_backwards[0] = min(n_backwards[0], n_batch_ele)
    
    #to 2nd checkpoint
    beta = np.mean(dist_from_checkpoints[1])/sum([np.linalg.norm(paramT-param0) for paramT, param0 in zip(checkpoints[epochs-1], checkpoints[1])])
    n_backwards[1] = int(np.ceil(args.nBackwardMult/beta)) #max(N,int(np.ceil(1./beta)))
    print("n_backwards[1]", n_backwards[1])
    n_backwards[1] = min(n_backwards[1], n_batch_ele)
    
    #to end checkpoint
    beta = np.mean(dist_from_checkpoints[epochs-1])/sum([np.linalg.norm(paramT-param0) for paramT, param0 in zip(theta_final, checkpoints[epochs-1])])
    n_backwards[epochs-1] = int(np.ceil(args.nBackwardMult/beta)) #max(N,int(np.ceil(1./beta)))
    print("n_backwards["+str(epochs-1)+"]", n_backwards[epochs-1])
    n_backwards[epochs-1] = min(n_backwards[epochs-1], n_batch_ele)
        
    
    
    # the backwards approximating path
    # init for pseudo params
    pseudo_params = []
    for i, v in enumerate(model.params_theta):
        pseudo_params.append(v.get_value())

    """
    def replace_pseudo_params(ratio):
        for i, param in enumerate(model.params_theta):
            pseudo_params[i] = (1 - ratio) * theta_initial[i] + ratio * theta_final[i]
            param.set_value(pseudo_params[i])
    """
    

    def replace_pseudo_params(ratio, checkpoint0, checkpoint1):
        for i, param in enumerate(model.params_theta):
            pseudo_params[i] = (1 - ratio) * checkpoint0[i] + ratio * checkpoint1[i]
            param.set_value(pseudo_params[i])
            
    #iter_index_cache = iter_index_cache[:n_backward]

    #rho = np.linspace(0.001, 0.999, n_backward)
    #rho = np.linspace(0.,1.,n_backward+1)[0:n_backward]
    rhos = OrderedDict()
    rhos[0] = np.linspace(0.,1.,n_backwards[0]+1)[0:n_backwards[0]]
    rhos[1] = np.linspace(0.,1.,n_backwards[1]+1)[0:n_backwards[1]]
    rhos[epochs-1] = np.linspace(0.,1.,n_backwards[epochs-1]+1)[0:n_backwards[epochs-1]]
    lrs = OrderedDict()
    lrs[0], lrs[1], lrs[epochs-1] = args.lrEle, args.lrEle/10., args.lrEle/10. #last one NOT /100.
    
    for ep 
        for iteration in range(n_backward)[::-1]:
        #backwards approximating path
        replace_pseudo_params(rho[iteration])         # line 4
        curr_batch = iteration % n_batch_ele
        if iteration % 40 == 0:
            print("Phase 3, ep{} iter{}, total{}".format(curr_epoch, curr_batch, iteration))
        #sample_idx_ele = iter_index_cache[iteration] # TODO: should we use the same batches? The number of iterations doesn't correspond anymore
        sample_idx_ele = random_sample(n_batch_ele, 1)
        # sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)]
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        for p3, p1, input_p in zip(up_v, up_theta, phase_3_input):
            p3 *= mom #corrected
            #p3 += lr_ele_true * p1 #dv += a_t dw
            p3 += args.lrEle * p1 #TODO: how should this change with schedule?
            input_p.set_value(p3)
        #for i in range(len(HVP_value)):
        #    print("HVP_value", i, HVP_value[i].shape, np.linalg.norm(HVP_value[i]))
        func_hyper(batch_x, batch_y)
        cnt = 0
        # update separately in case the lists have different lenths (not all parametrized layers are regularized)
        for p1, hvp1 in zip(up_theta, HVP_weight):
            #if np.linalg.norm(hvp1.get_value(borrow=True)) == 0.:
            #    print(cnt, "hvp1", hvp1.shape, np.linalg.norm(hvp1.get_value(borrow=True)))
            p1 -= (1. - mom) * np.array(hvp1.get_value())
            cnt += 1
        cnt = 0
        for p2, hvp2 in zip(up_lambda, HVP_lambda):
            #if np.linalg.norm(hvp2.get_value(borrow=True)) == 0.: #cnt = 1, 2 and 3 (not 0)
            #    print(cnt, "hvp2", hvp2.shape, np.linalg.norm(hvp2.get_value(borrow=True)))
            p2 -= (1. - mom) * np.array(hvp2.get_value())
            cnt += 1
    
    """
    for iteration in range(n_backward)[::-1]:
        #backwards approximating path
        replace_pseudo_params(rho[iteration])         # line 4
        curr_epoch = iteration // n_batch_ele
        curr_batch = iteration % n_batch_ele
        if iteration % 40 == 0:
            print("Phase 3, ep{} iter{}, total{}".format(curr_epoch, curr_batch, iteration))
        #sample_idx_ele = iter_index_cache[iteration] # TODO: should we use the same batches? The number of iterations doesn't correspond anymore
        sample_idx_ele = random_sample(n_batch_ele, 1)
        # sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)]
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        for p3, p1, input_p in zip(up_v, up_theta, phase_3_input):
            p3 *= mom #corrected
            #p3 += lr_ele_true * p1 #dv += a_t dw
            p3 += args.lrEle * p1 #TODO: how should this change with schedule?
            input_p.set_value(p3)
        # Hessian-vector product; these only include regularized parameters (model.params_weight)
        #HVP_value = func_hyper(batch_x, batch_y)
        # should just keep separate lists in updates_hyper?
        #HVP_weight_value = HVP_value[:len(HVP_weight_temp)] #[:len(model.params_weight)] #HVP_value[:len(model.params_weight)]
        #HVP_lambda_value = HVP_value[len(HVP_weight_temp):len(HVP_weight_temp)+len(HVP_lambda_temp)] #[len(model.params_weight):len(model.params_weight)+len(model.params_lambda)]
        #debug_orz = HVP_value[len(HVP_weight_temp)+len(HVP_lambda_temp):] #[len(model.params_weight)+len(model.params_lambda):]
        #HVP_weight_value, HVP_lambda_value, valid_list = func_hyper(batch_x, batch_y)
        #for i in range(len(HVP_value)):
        #    print("HVP_value", i, HVP_value[i].shape, np.linalg.norm(HVP_value[i]))
        func_hyper(batch_x, batch_y)
        cnt = 0
        # update separately in case the lists have different lenths (not all parametrized layers are regularized)
        for p1, hvp1 in zip(up_theta, HVP_weight):
            #if np.linalg.norm(hvp1.get_value(borrow=True)) == 0.:
            #    print(cnt, "hvp1", hvp1.shape, np.linalg.norm(hvp1.get_value(borrow=True)))
            p1 -= (1. - mom) * np.array(hvp1.get_value())
            cnt += 1
        cnt = 0
        for p2, hvp2 in zip(up_lambda, HVP_lambda):
            #if np.linalg.norm(hvp2.get_value(borrow=True)) == 0.: #cnt = 1, 2 and 3 (not 0)
            #    print(cnt, "hvp2", hvp2.shape, np.linalg.norm(hvp2.get_value(borrow=True)))
            p2 -= (1. - mom) * np.array(hvp2.get_value())
            cnt += 1
        #for p3 in up_v:
            #p3 *= mom #corrected above
        
            #for i in range(0, len(up_theta)):
            #    print("up_theta["+str(i)+"].shape "+ str(up_theta[i].shape))
            #for i in range(0, len(HVP_weight_value)):
            #    print("HVP_weight_value["+str(i)+"].shape "+ str(HVP_weight_value[i].shape))
            #for i in range(0, len(up_lambda)):
            #    print("up_lambda["+str(i)+"].shape "+ str(up_lambda[i].shape))
            #for i in range(0, len(HVP_lambda_value)):
            #    print("HVP_lambda_value["+str(i)+"].shape "+ str(HVP_lambda_value[i].shape))
    """
    print("Phase 3 backward pass duration="+str(time()-t_phase3)+"s")
    
    return up_lambda, fix_weight
    #print('up_log_learning_rates', up_log_learning_rates)
    #return up_lambda, up_log_learning_rates, fix_weight, test_loss, test_error


def update_lambda_every_meta(ori, up, hyper_lr, updatemode='output_unit'): #returns the new hyperparameter values
    tmp = []
    for x, y in zip(ori, up):
        if updatemode == 'unit' or (updatemode == 'output_unit' and x.name == 'output.L2'):
            new_y = np.mean(y, axis=1, keepdims=True) #TODO: average over axis 1 (keeping the axis 1 dimension), which corresponds to the unit for DenseLayer W
            #Same as ScaleLayer and BiasLayer default sharing
            #But for conv layer, axis=1 is the input channels, so axis=0 for the filters would make more sense
            #This also means many values are stored multiple times separately and redundantly...
            tmp.append(x.get_value() - np.sign(new_y) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX)) #what's this np.array for?
            #print("metaupdate", x.get_value()[0][1], tmp[-1][0][1])
        else: #per parameter
            tmp.append(x.get_value() - np.sign(y) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX))
            #print("metaupdate", x.get_value()[0][1], tmp[-1][0][1])
    return tmp

def update_lambda_every_meta_mom(ori, up, lam_vels, hyper_lr, hyper_mom, updatemode= 'output_unit'): #returns the new hyperparameter values
    tmp = []
    for x, y, v in zip(ori, up, lam_vels):
        if updatemode == 'unit' or (updatemode == 'output_unit' and x.name == 'output.L2'):
            new_y = np.mean(y, axis=1, keepdims=True)
            hyper_mom = np.array(float(hyper_mom) * 1., dtype=theano.config.floatX)
            v.set_value(hyper_mom*v.get_value(borrow=True) - (1.-hyper_mom)*np.sign(new_y))
            tmp.append(x.get_value(borrow=True) + v.get_value(borrow=True) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX))
            #print("metaupdate", x.get_value()[0][1], tmp[-1][0][1])
        else: #per parameter
            hyper_mom = np.array(float(hyper_mom) * 1., dtype=theano.config.floatX)
            v.set_value(hyper_mom*v.get_value(borrow=True) - (1.-hyper_mom)*np.sign(y))
            tmp.append(x.get_value(borrow=True) + v.get_value(borrow=True) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX))
            #print("metaupdate", x.get_value()[0][1], tmp[-1][0][1])
    return tmp

#Nesterov? meh

        
def save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses, train_errors, valid_errors, test_errors, norms, theta_names):
    # all losses in one plot
    for i in range(0, len(test_losses)): #for each meta-iteration
        plt.plot(unreg_train_losses[i], label="Unreg tra "+str(i))
        plt.plot(train_losses[i], label="Train "+str(i))
        plt.plot(valid_losses[i], label="Valid "+str(i))
        plt.plot(test_losses[i], label="Test "+str(i))
    plt.title("Loss vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    # all errors in one plot
    for i in range(0, len(test_errors)): #for each meta-iteration
        plt.plot(train_errors[i], label="Train "+str(i))
        plt.plot(valid_errors[i], label="Valid "+str(i))
        plt.plot(test_errors[i], label="Test "+str(i))
    plt.title("Error vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/errors.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/errors.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    #each loss type separately, but for all meta-iters
    for i in range(0, len(test_losses)):
        plt.plot(unreg_train_losses[i], label="Meta "+str(i))
    plt.title("Unreg train loss vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/unreg_train_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/unreg_train_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_losses)):
        plt.plot(train_losses[i], label="Meta "+str(i))
    plt.title("Train loss vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/train_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/train_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_losses)):
        plt.plot(valid_losses[i], label="Meta "+str(i))
    plt.title("Valid loss vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/valid_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/valid_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_losses)):
        plt.plot(test_losses[i], label="Meta "+str(i))
    plt.title("Test loss vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/test_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/test_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    #each error type separately, but for all meta-iters
    for i in range(0, len(test_errors)):
        plt.plot(train_errors[i], label="Meta "+str(i))
    plt.title("Train error vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/train_errors.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/train_errors.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_errors)):
        plt.plot(test_errors[i], label="Meta "+str(i))
    plt.title("Valid error vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/valid_errors.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/valid_errors.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_errors)):
        plt.plot(test_errors[i], label="Meta "+str(i))
    plt.title("Test error vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/test_errors.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/test_errors.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    
    #all final losses in one plot
    plt.plot([unreg_train_losses[i][-1] for i in range(len(unreg_train_losses))], label="Unreg tra")
    plt.plot([train_losses[i][-1] for i in range(len(train_losses))], label="Train")
    plt.plot([valid_losses[i][-1] for i in range(len(valid_losses))], label="Valid")
    plt.plot([test_losses[i][-1] for i in range(len(test_losses))], label="Test")
    plt.title("Final loss vs meta-iteration")
    plt.xlabel("Meta-iteration")
    plt.ylabel("Final loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/losses_per_meta.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/losses_per_meta.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    # all final errors in one plot
    plt.plot([train_errors[i][-1] for i in range(len(train_errors))], label="Train")
    plt.plot([valid_errors[i][-1] for i in range(len(valid_errors))], label="Valid")
    plt.plot([test_errors[i][-1] for i in range(len(test_errors))], label="Test")
    plt.title("Final error vs meta-iteration")
    plt.xlabel("Meta-iteration")
    plt.ylabel("Final error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/errors_per_meta.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/errors_per_meta.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    
            
    f=open(outdir+'/losses.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump([unreg_train_losses, train_losses, valid_losses, test_losses], f)
    f.close()
    f=open(outdir+'/errors.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump([train_errors, valid_errors, test_errors], f)
    f.close()
    f=open(outdir+'/norms.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump(norms, f)
    f.close()


def load_errs_losses(outdir):
    f=open(outdir+'/losses.pckl', 'rb')
    unreg_train_losses, train_losses, valid_losses, test_losses = pickle.load(f, encoding='latin1')
    f.close()
    f=open(outdir+'/errors.pckl', 'rb')  # Python 2: open(..., 'w')
    train_errors, valid_errors, test_errors = pickle.load(f, encoding='latin1')
    f.close()
    return unreg_train_losses, train_losses, valid_losses, test_losses, train_errors, valid_errors, test_errors

def load_log_regs(outdir):
    f=open(outdir+'/hypers.pckl', 'rb')  # Python 2: open(..., 'w')
    per_layer_log_reg = pickle.load(f, encoding='latin1')
    f.close()
    return per_layer_log_reg



per_layer_log_reg = [[np.mean(reg.get_value())] for reg in model.params_lambda]
    
for i in range(args.metaEpoch):
    print("Meta iter "+str(i))
    
    #update_lambda, fix_weight = temp_lambda, tmp_weights
    
    epochs = min(int(args.maxEpochInit*(args.maxEpochMult**float(i))), args.maxMaxEpoch)
    #First get the gradients
    temp_lambda, tmp_weights = run_exp(args, temp_lambda, tmp_weights, epochs)
    #temp_lambda, temp_llrs, tmp_weights, test_loss, test_error = run_exp(args, temp_lambda, temp_llrs, tmp_weights)    
    #temp_lambda, tmp_weights, test_loss, test_error = model.params_lambda, up_lambda, fix_weight, test_loss, test_error
    
    #Then get the steps
    #temp_lambda = update_lambda_every_meta(model.params_lambda, temp_lambda, args.lrHyper, 'unit')
    temp_lambda = update_lambda_every_meta_mom(model.params_lambda, temp_lambda, lambda_velocities, args.lrHyper, args.momHyper)
    #Note that the last hyperparameters aren't tested
    
    
    #save model parameters, hyperparameters, hyperparameter velocities
    f=open(outdir+'/lambda.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump(temp_lambda, f)
    f.close()
    f=open(outdir+'/theta.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump(model.params_theta, f)
    f.close()
    f=open(outdir+'/fixed_initial_thetas.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump(tmp_weights, f)
    f.close()
    f=open(outdir+'/lambda_vels.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump(lambda_velocities, f)
    f.close()
    
    
    #store per-layer average log regularization
    for j, reg, reg2 in zip(range(0,len(temp_lambda)),temp_lambda, model.params_lambda):
        per_layer_log_reg[j].append(np.mean(reg))
        plt.plot(per_layer_log_reg[j], label=reg2.name)
    plt.title("Per-layer average log reg hyper vs meta-iter")
    plt.xlabel("Meta iteration")
    plt.ylabel("Average log reg hyper")
    lgd = plt.legend(bbox_to_anchor=(2, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/reg.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/reg.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    f=open(outdir+'/layer_log_regs.pckl', 'wb')  # Python 2: open(..., 'w')
    pickle.dump(per_layer_log_reg, f)
    f.close()

    
    
    print("---------------------------------------------------------------------------------------------------")
    print("Training Result: ")
    print("Test loss: "+str(test_losses[-1][-1]))
    print("Test error: "+str(test_errors[-1][-1]))
    print("---------------------------------------------------------------------------------------------------")
