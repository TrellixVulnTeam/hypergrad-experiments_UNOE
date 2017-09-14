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

X_elementary, Y_elementary, X_test, Y_test = load_dataset(args) #normalized
#Use a large validation set (as in CPU experiments) to avoid overfitting the hyperparameters
X_hyper = X_elementary[0:5000]
Y_hyper = Y_elementary[0:5000]
X_elementary = X_elementary[5000:]
Y_elementary = Y_elementary[5000:]


#TODO: seeds for dropout, reinitialize BN layers
#import lasagne.random
#np.random = np.random.RandomState(args.seed)
#rand = np.random.RandomState(args.seed)
#lasagne.random.set_rng(rand)
model = DenseNet(x=x, y=y, args=args)
#model = ConvNet(x=x, y=y, args=args)

velocities = [theano.shared(np.asarray(param.get_value(borrow=True)*0., dtype=theano.config.floatX), broadcastable=param.broadcastable, name=param.name+'_vel') for param in model.params_theta]
momLlr = args.momLlr









#make a directory with a timestamp name, and save results to it
import os
print("---------------------------------------------------------------------------------------------------")
print('Forward_only_{:%Y-%b-%d_%Hh%Mm%Ss}'.format(datetime.datetime.now()))
print("---------------------------------------------------------------------------------------------------")
outdir = 'Forward_only_{:%Y-%b-%d_%Hh%Mm%Ss}'.format(datetime.datetime.now())
os.mkdir(outdir)

#copy files to directory in case of changes
import shutil
shutil.copy2(os.path.basename(__file__), outdir) #simple_mlp2_gpu.py
shutil.copy2('args.py', outdir)
shutil.copy2('updates.py', outdir)
shutil.copy2('layers.py', outdir)
shutil.copy2('densenet_fast.py', outdir)





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
distances_from_final = []
#grad_norms = []
velocity_norms = []
theta_names = [param.name for param in model.params_theta]


#update_lambda, fix_weight = temp_lambda, tmp_weights
#update_llrs = temp_llrs


dlossWithPenalty_dtheta = theano.grad(model.lossWithPenalty, model.params_theta)
update_ele, update_valid, output_valid_list, share_var_dloss_dweight = update(model.params_theta, model.params_lambda, model.params_weight,
                                  velocities, model.loss, model.penalty, dlossWithPenalty_dtheta,
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






theta_final = None


def run_exp(args, fix_weight, theta_final, epochs):
    global X_elementary, Y_elementary, X_hyper, Y_hyper, X_test, Y_test
    global training_errors, unregularized_training_errors, valid_errors, hyper_errors, test_errors, training_losses, valid_losses, hyper_losses, test_losses, norms, theta_names

    #reinitialize the elementary parameters (exactly as before; break symmetry the same way again)
    if fix_weight:
        #for fix, origin in zip(fix_weight, model.params_weight):
         #   origin.set_value(np.array(fix))
        for fix, origin, velocity in zip(fix_weight, model.params_theta, velocities):
            #origin.set_value(np.array(fix))
            velocity.set_value(velocity.get_value()*0.) #reset velocities to 0
    else: #store the elementary parameters for the first time, as arrays
    #TODO: this could be put outside of run_exp()
        fix_weight = []
        for origin in model.params_theta:
            fix_weight.append(origin.get_value()) #copies (borrow=False by default)


    # Phase 1
    # elementary SGD variable update list (constant momenta and learning rates)
    
    #TODO:
    """
    dlossWithPenalty_dtheta = theano.grad(model.lossWithPenalty, model.params_theta)
    update_ele, update_valid, output_valid_list, share_var_dloss_dweight = update(model.params_theta, model.params_lambda, model.params_weight,
                                      velocities, model.loss, model.penalty, dlossWithPenalty_dtheta,
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
    """


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
    if theta_final:
        for param, fin in zip(model.params_theta, theta_final):
            distances_from_final.append([np.linalg.norm(param.get_value(borrow=True)-fin)])
    else:
        for v in velocities:
            velocity_norms.append([np.linalg.norm(v.get_value(borrow=True))])
    
    lr_ele_true = args.lrEle
    
    t_start = time()
    
    T = epochs * n_batch_ele
    #T = n_batch_ele//3  #TODO: change back
    #T = 40
    for i in range(0, T): #SGD steps
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
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele] #batch data
        tmp_y = np.zeros((args.batchSizeEle, 10)) #10 for 10 classes; put a 1 in row=idx and column=class=element of idx 
        for idx, element in enumerate(batch_y): #idx = index, element = element at that index
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        
        #update elementary parameters
        train_loss, unreg_train_loss, train_pred = func_elementary(batch_x, batch_y, lr_ele_true)
        
        for norm, param in zip(norms, model.params_theta):
            norm.append(np.linalg.norm(param.get_value(borrow=True)))
        if theta_final:
            for j, param, fin in zip(range(len(model.params_theta)), model.params_theta, theta_final):
                distances_from_final[j].append(np.linalg.norm(param.get_value(borrow=True)-fin))
        else:
            for j, v in enumerate(velocities):
                velocity_norms[j].append(np.linalg.norm(v.get_value(borrow=True)))
        
        
        
        
        
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
        if curr_batch == n_batch_ele - 1: #last batch #TODO:
        #if i==T-1:
            #use deterministic option for BN layers
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
            
            #Save and save plots
            save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses,
                train_errors, valid_errors, test_errors, norms, distances_from_final, velocity_norms, theta_names)


    # save the model parameters after T1 into theta_final
    if not theta_final:
        theta_final = [param.get_value() for param in model.params_theta]
        f=open(outdir+'/theta_final.pckl', 'wb')  # Python 2: open(..., 'w')
        pickle.dump(theta_final, f)
        f.close()


    return fix_weight, theta_final


def save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses, train_errors, valid_errors, test_errors, norms, distances_from_final, velocity_norms, theta_names):
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
    
    if theta_final:
        for i, param in enumerate(model.params_theta):
            plt.plot(distances_from_final[i], label=param.name)
        plt.title("Distance from theta_final vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Distance from theta_final")
        lgd = plt.legend(bbox_to_anchor=(2, 1.025), loc='upper right', ncol=2)
        plt.savefig(outdir+"/distances.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(outdir+"/distances.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()
        f=open(outdir+'/distances_from_final.pckl', 'wb')  # Python 2: open(..., 'w')
        pickle.dump(distances_from_final, f)
        f.close()
    else:
        for i, param in enumerate(model.params_theta):
            plt.plot(norms[i], label=param.name)
        plt.title("Norm vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Norm")
        lgd = plt.legend(bbox_to_anchor=(2, 1.025), loc='upper right', ncol=2)
        plt.savefig(outdir+"/norms.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(outdir+"/norms.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()
        f=open(outdir+'/norms.pckl', 'wb')  # Python 2: open(..., 'w')
        pickle.dump(norms, f)
        f.close()
        
        for i, param in enumerate(model.params_theta):
            plt.plot(velocity_norms[i], label=param.name)
        plt.title("Velocity norm vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Velocity norm")
        lgd = plt.legend(bbox_to_anchor=(2, 1.025), loc='upper right', ncol=2)
        plt.savefig(outdir+"/v_norms.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(outdir+"/v_norms.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()
        f=open(outdir+'/v_norms.pckl', 'wb')  # Python 2: open(..., 'w')
        pickle.dump(velocity_norms, f)
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



f=open(outdir+'/theta_initial.pckl', 'wb')  # Python 2: open(..., 'w')
pickle.dump([param.get_value(borrow=True) for param in model.params_theta], f)
f.close()

    
tmp_weights, theta_final = run_exp(args, tmp_weights, theta_final, args.maxMaxEpoch)

#rand = np.random.RandomState(args.seed)
#lasagne.random.set_rng(rand)
#TODO:
#model = DenseNet(x=x, y=y, args=args) #reinitialize batch norm layers
for p1, p2 in zip(model.params_theta, tmp_weights):
    p1.set_value(p2)

run_exp(args, tmp_weights, theta_final, args.maxMaxEpoch)