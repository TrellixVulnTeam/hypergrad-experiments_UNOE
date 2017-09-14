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
from models import MLP, ConvNet
from densenet_fast import DenseNet
from updates import update, updates_hyper, updates_hyper2, update2, updates_noise_hyper
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
temp_noise = None
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

mom = args.momEle #momentum
lr_hyper = T.fscalar('lr_hyper')
grad_valid_weight = T.tensor4('grad_valid_weight')


model = DenseNet(x=x, y=y, args=args)


velocities = [theano.shared(np.asarray(param.get_value(borrow=True)*0., dtype=theano.config.floatX), broadcastable=param.broadcastable, name=param.name+'_vel') for param in model.params_theta]

#extra lr parameters
log_learning_rates = [theano.shared(np.full_like(param.get_value(borrow=True), np.log(args.lrEle), dtype=theano.config.floatX), broadcastable=param.broadcastable, name=param.name+'_llr') for param in model.params_theta]
temp_llrs = None
temp_noise = None

if not args.momLlr == 0.: #TODO: if not using momentum (both llr and lambda or neither)
    llr_velocities = [theano.shared(np.asarray(llr.get_value(borrow=True)*0., dtype=theano.config.floatX), broadcastable=llr.broadcastable, name=llr.name+'_vel') for llr in log_learning_rates]
    lambda_velocities = [theano.shared(np.asarray(lamb.get_value(borrow=True)*0., dtype=theano.config.floatX), broadcastable=lamb.broadcastable, name=lamb.name+'_vel') for lamb in model.params_lambda]
    momHyper = args.momHyper
    momLlr = args.momLlr
else:
    llr_velocities = []
    lambda_velocities = []
    




X_elementary, Y_elementary, X_test, Y_test = load_dataset(args) #normalized
#Use a large validation set (as in CPU experiments) to avoid overfitting the hyperparameters
#TODO: try even larger validation set; seems to be overfitting at 10000
#What about keeping training and validation sets together?
X_hyper = X_elementary[0:22000]
Y_hyper = Y_elementary[0:22000]
X_elementary = X_elementary[22000:]
Y_elementary = Y_elementary[22000:]


n_ele, n_hyper, n_test = X_elementary.shape[0], X_hyper.shape[0], X_test.shape[0]
n_batch_ele = n_ele // args.batchSizeEle #integer division, so some data may be lost
test_perm, ele_perm, hyper_perm = range(0, n_test), range(0, n_ele), range(0, n_hyper)
n_eval = 700 #sample size from training/elementary, hyper/valid and test sets to evaluate on
# will be overwritten below    




#make a directory with a timestamp name, and save results to it
import os
print('Online_{:%Y-%b-%d_%Hh%Mm%Ss}'.format(datetime.datetime.now()))
outdir = 'Online_{:%Y-%b-%d_%Hh%Mm%Ss}'.format(datetime.datetime.now())
os.mkdir(outdir)

#copy files to directory in case of changes
import shutil
shutil.copy2('simple_mlp2_gpu_online.py', outdir)
#shutil.copy2(os.path.basename(__file__), outdir) #simple_mlp2_gpu_online.py
shutil.copy2('args.py', outdir)
shutil.copy2('updates.py', outdir)
shutil.copy2('layers.py', outdir)
shutil.copy2('densenet_fast.py', outdir)



# Phase 1
# elementary SGD variable update list (constant momenta and learning rates)

dlossWithPenalty_dtheta = theano.grad(model.lossWithPenalty, model.params_theta)

update_ele, update_valid, output_valid_list, share_var_dloss_dweight = update2(model.params_theta, model.params_lambda, model.params_weight,
                                  velocities, model.loss, model.penalty, dlossWithPenalty_dtheta,
                                  log_learning_rates, mom)
#output_valid_list is used as dv_t in DrMAD


func_elementary = theano.function(
    inputs=[x, y],
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
    outputs=[model.loss_det, model.prediction_det], # + output_valid_list,
    updates=update_valid, #this updates output_valid_list to dloss_dweight = T.grad(model.loss, model.params_weight)
    on_unused_input='ignore',
    allow_input_downcast=True)

grad_l_theta = [np.asarray(param.get_value()*0., dtype=theano.config.floatX) for param in model.params_theta] 
#= dw_T = gradient of validation loss wrt final weights


# Phase 3
#output_valid_list here should be dv_t
#phase_3_input is supposed to be the same variable as output_valid_list?
#model.params_weight instead of model.params_theta because penalty does not depend on biases, and
#and we want initial biases to *always* be 0
#update_hyper, output_hyper_list, phase_3_input = updates_hyper(model.params_lambda, model.params_weight,
#                                                model.lossWithPenalty, grad_l_theta, output_valid_list)
# These are the Theano shared, without values!
# updates for phase 3
update_hyper, HVP_theta, HVP_lambda, phase_3_input = updates_hyper2(model.params_lambda, model.params_theta, 
                                                                               model.params_weight, dlossWithPenalty_dtheta, 
                                                                               output_valid_list)
if args.addActivationNoise or args.addInputNoise:
    update_noise_hyper, HVP_noise = updates_noise_hyper(model.params_noise, model.params_theta, dlossWithPenalty_dtheta, output_valid_list)                                                                               
                                                                               
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

#TODO: this is what's slow
if not args.addActivationNoise or args.addInputNoise:
    func_hyper = theano.function(
        inputs=[x, y],
        #outputs=output_hyper_list + output_valid_list, #HVP_value
        #outputs = HVP_theta_temp+HVP_lambda_temp+output_valid_list,
        updates=update_hyper,
        on_unused_input='ignore',
        allow_input_downcast=True)
else:
    func_hyper = theano.function(
        inputs=[x, y],
        updates=update_hyper+update_noise_hyper,
        on_unused_input='ignore',
        allow_input_downcast=True)

print("func_hyper defined, taking "+str(time()-func_hyper_t0)+"s")
#On my laptop's CPU, "func_hyper defined, taking 354.448412895s" ~ 6 minutes, or 178.747463942s with FAST_COMPILE
#On school GPU with FAST_COMPILE "func_hyper defined, taking 495.1831388473511s" ~ 8min


#reverse path functions
updates_reverse_params = []
for velocity, param, llr in zip(velocities, model.params_theta, log_learning_rates):
    updates_reverse_params.append((param, param - T.exp(llr) * velocity))
func_reverse_params = theano.function(inputs=[], updates=updates_reverse_params)
updates_reverse_velocities = []
for velocity, grad in zip(velocities, dlossWithPenalty_dtheta):
    updates_reverse_velocities.append((velocity, (velocity + (1-mom)*grad)/mom))
func_reverse_velocities = theano.function(inputs=[x, y], updates=updates_reverse_velocities, allow_input_downcast=True)


#TODO: noise for parameters
#if args.addParameterNoise:
    #noise = 




train_errors = []
valid_errors = []
test_errors = []

unreg_train_losses = []
train_losses = []
valid_losses = []
test_losses = []

norms = []
theta_names = [param.name for param in model.params_theta]


#update_lambda, fix_weight = temp_lambda, tmp_weights
#update_llrs = temp_llrs










theta_initial_initial = [param.get_value(borrow=False) for param in model.params_theta]

def run_exp(args, update_lambda, update_llrs, update_noise, first_iter):
    global X_elementary, Y_elementary, X_hyper, Y_hyper, X_test, Y_test
    global training_errors, unregularized_training_errors, valid_errors, hyper_errors, test_errors, training_losses, valid_losses, hyper_losses, test_losses, norms, theta_names

    #update the hyperparameters
    if update_lambda:
        for up, origin in zip(update_lambda, model.params_lambda):
            origin.set_value(np.array(up))
    #Log learning rates
    #print("update_llrs", update_llrs)
    if update_llrs:
        for up, origin in zip(update_llrs, log_learning_rates):
            origin.set_value(np.array(up))
    #noise hypers
    if (args.addActivationNoise or args.addInputNoise) and update_noise:
        for up, origin in zip(update_noise, model.params_noise):
            origin.set_value(np.array(up))


    #Phase 1


    """
         Phase 1: meta-forward

    """


    
    iter_index_cache = []
    if args.onlineItersPerUpdate == args.nReversedIters:
        theta_initial = [param.get_value(borrow=False) for param in model.params_theta] #TODO:
    T = args.onlineItersPerUpdate
    for i in range(first_iter, first_iter+T): # SGD steps
        curr_epoch = i // n_batch_ele #TODO: always 0???
        curr_batch = i % n_batch_ele


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

        train_loss, unreg_train_loss, train_pred = func_elementary(batch_x, batch_y)

    curr_epoch = (first_iter+T-1) // n_batch_ele 
    #TODO: less frequently?
    wrong = 0
    for e1, e2 in zip(train_pred, Y_elementary[sample_idx_ele]): #
        if e1 != e2:
            wrong += 1
    train_error = 100. * wrong / len(train_pred)
    print("Train Set: Epoch %d, batch %d, loss = %.4f, unreg loss = %.4f, error = %.4f" %
          (curr_epoch, curr_batch, train_loss, unreg_train_loss, train_error))

    #TODO:
    # save the model parameters after T1 into theta_final
    """
    theta_final = []
    for w in model.params_theta:
        theta_final.append(w.get_value())
    velocities_final = []
    for v in velocities:
        velocities_final.append(v.get_value())
    """


    """
        Phase 2: Validation on Hyper set and computation of dw_T = grad_w f(w_T), for f the validation loss

    """
    n_hyper = X_hyper.shape[0]
    n_batch_hyper = n_hyper // args.batchSizeHyper #10000/64 ~ 156
    hyper_perm = range(0, n_hyper)
    n_batch_samples_hyper = 3*args.nReversedIters # T, nReversedIters #TODO: 
    # np.random.shuffle(hyper_perm)

    err_valid = 0.
    cost_valid = 0.
    #t_start = time()
    for j, i in enumerate(random_sample(range(n_batch_hyper), n_batch_samples_hyper)): #use T=args.onlineItersPerUpdate batches
        sample_idx = hyper_perm[(i * args.batchSizeHyper):((i + 1) * args.batchSizeHyper)]
        batch_x, batch_y = X_hyper[sample_idx], Y_hyper[sample_idx]
        # TODO: refactor, too slow
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        valid_cost, pred_hyper = func_hyper_valid(batch_x, batch_y) #updates output_valid_list to gradient
        err_tmp = 1. * sum(np.argmax(batch_y, axis=1) != pred_hyper) / args.batchSizeHyper
        err_valid += err_tmp
        # print "err_temp", err_tmp
        cost_valid += valid_cost

        # accumulate gradient and then take the average
        if j == 0: #instead of the randomly chosen i
            #for grad, meh in zip(grad_temp, output_valid_list):
            #    print(np.linalg.norm(grad), np.linalg.norm(meh.get_value()))
            for k, grad in enumerate(output_valid_list):
                grad_l_theta[k] = grad.get_value(borrow=False)
        else:
            for k, grad in enumerate(output_valid_list):
                grad_l_theta[k] += grad.get_value(borrow=False)
    err_valid /= n_batch_samples_hyper
    cost_valid /= n_batch_samples_hyper

    # get average grad of all iterations on validation set
    for i, grad in enumerate(grad_l_theta):
        #print("grad_l_theta norm 1", np.linalg.norm(grad_l_theta[i]))
        grad_l_theta[i] = grad / (np.array(T * 1., dtype=theano.config.floatX))
        #print("grad_l_theta norm 2", np.linalg.norm(grad_l_theta[i]))

    print("Valid on Hyper Set: valid_err = %.2f, valid_loss = %.4f" %
          (err_valid * 100, cost_valid))
          

    """
        Phase 3: meta-backward

    """

    # initialization
    up_lambda, up_v, up_noise = [], [], []
    for param in model.params_lambda:
        temp_param = np.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX) #np.zeros_like(param.get_value() * 0., dtype=theano.config.floatX)
        up_lambda += [temp_param]
        
    for param in model.params_noise:
        temp_param = np.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)
        up_noise += [temp_param]

    for param in model.params_theta:
        temp_v = np.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX) #np.zeros_like(param.get_value() * 0., dtype=theano.config.floatX)
        up_v += [temp_v]

    up_theta = grad_l_theta #dw_0, w=theta=all weights

    # --------------------------------------------------------------------------------
    # Exactly reverse SGD+momentum steps (without exact arithmetic)

    up_log_learning_rates = [np.zeros_like(llr.get_value(borrow=True), dtype=theano.config.floatX) for llr in log_learning_rates]
    
    T_rev = args.nReversedIters #TODO: T_rev = T; reversing fewer iterations than all T=args.onlineItersPerUpdate instead to speed things up
    for iteration in range(T-T_rev, T)[::-1]:
        #dlearning_rate = dw^T v, but log-scale, per-parameter, shared for each iteration
        for p1, velocity, llr, up_llr in zip(up_theta, velocities, log_learning_rates, up_log_learning_rates):
           up_llr += p1*velocity.get_value(borrow=True)*np.exp(llr.get_value(borrow=True))
        sample_idx_ele = iter_index_cache[iteration]
        # sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)]
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        #reverse SGD+momentum step
        func_reverse_params()
        func_reverse_velocities(batch_x, batch_y)
        for p3, p1, input_p, llr in zip(up_v, up_theta, phase_3_input, log_learning_rates):
            if(np.linalg.norm(p1)) == 0.:
                print('p1')
            p3 *= mom #corrected
            p3 += np.exp(llr.get_value(borrow=True)) * p1 #dv += a_t dw
            input_p.set_value(p3)
            if np.linalg.norm(p3) == 0.:
                print('p3')
        #print('output_valid_list',output_valid_list is phase_3_input)
        func_hyper(batch_x, batch_y)
        cnt = 0
        # update separately in case the lists have different lenths (not all parametrized layers are regularized)
        for p1, hvp1 in zip(up_theta, HVP_theta):
            if np.linalg.norm(hvp1.get_value(borrow=True)) == 0.:
                print(cnt, "hvp1", hvp1.shape, np.linalg.norm(hvp1.get_value(borrow=True)))
            p1 -= (1. - mom) * np.array(hvp1.get_value())
            cnt += 1
        cnt = 0
        for p2, hvp2 in zip(up_lambda, HVP_lambda):
            if np.linalg.norm(hvp2.get_value(borrow=True)) == 0.: #cnt = 1, 2 and 3 (not 0)
                print(cnt, "hvp2", hvp2.shape, np.linalg.norm(hvp2.get_value(borrow=True)))
            p2 -= (1. - mom) * np.array(hvp2.get_value())
            cnt += 1
        #TODO: Gaussian noise; don't bother reusing the previous noise exactly
        if args.addActivationNoise or args.addInputNoise:
            cnt = 0
            for pn, hvpn in zip(up_noise, HVP_noise):
                if np.linalg.norm(hvp2.get_value(borrow=True)) == 0.: #cnt = 1, 2 and 3 (not 0)
                    print(cnt, "hvpn", hvpn.shape, np.linalg.norm(hvpn.get_value(borrow=True)))
                pn -= (1. - mom) * np.array(hvpn.get_value())
                cnt += 1
    
    # TODO: Check if reversing worked
    if args.onlineItersPerUpdate == args.nReversedIters:
        dist = 0.
        prop_error = 0.
        n_params = 0
        for param, param2 in zip(model.params_theta, theta_initial):
            dist += np.sum(np.abs(param.get_value(borrow=True)-param2))
            prop_error += 2.*np.sum(np.abs(param.get_value(borrow=True)-param2)/(np.abs(param2)+np.abs(param.get_value(borrow=True))+10.**-10.))
            n_params += np.sum(np.full_like(param.get_value(borrow=True), 1))
        print('L1 distance between theta_initial and params_theta after reverse pass='+ str(dist))
        print('Number of parameters=' + str(int(n_params)))
        print('Average L1 dist=' + str((dist/np.float(n_params))))
        print('Average proportional error=' + str(prop_error/np.float(n_params)))
    

    # --------------------------------------------------------------------------------
    

    #TODO:    
    #return to parameters at the end of the SGD steps
    """    
    for p1, p2 in zip(theta_final, model.params_theta):
        p2.set_value(p1)
    for v1, v2 in zip(velocities_final, velocities):
        v2.set_value(v1)
    """
    for sig, un in zip(model.params_noise, up_noise):
        print(sig.name, np.linalg.norm(un))

    #print('up_log_learning_rates', up_log_learning_rates)
    return up_lambda, up_log_learning_rates, up_noise


# clip llrs between -104 and 0 (np.float32(np.e**-104.) == 0.)

def update_lambda_lr_every_meta(ori, up, llr, up_llr, noise, up_noise, hyper_lr, lr_Llr, updatemode='output_unit'):
    tmp = []
    for x, y in zip(ori, up):
        if updatemode == 'unit' or (updatemode == 'output_unit' and x.name == 'output.L2'):
            new_y = np.mean(y, axis=1, keepdims=True)
            tmp.append(np.clip(x.get_value() - np.sign(new_y) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX), a_min=-7, a_max=-2))
            #print("metaupdate", x.get_value()[0][1], tmp[-1][0][1])
            #print('shared lambda')
        else: #per parameter
            tmp.append(np.clip(x.get_value() - np.sign(y) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX), a_min=-7, a_max=-2))
    tmp2 = []
    for x, y in zip(llr, up_llr):
        if updatemode == 'unit' or (updatemode == 'output_unit' and x.name == 'output.W_llr'):
            new_y = np.mean(y, axis=1, keepdims=True)
            tmp2.append(np.clip(x.get_value() - np.sign(new_y) * np.array(float(lr_Llr) * 1., dtype=theano.config.floatX), a_min=np.log(0.00001), a_max=0))
            #print("metaupdate", x.get_value()[0][1], tmp[-1][0][1])
            #print('shared llr')
        else: #per parameter
            tmp2.append(np.clip(x.get_value() - np.sign(y) * np.array(float(lr_Llr) * 1., dtype=theano.config.floatX), a_min=np.log(0.00001), a_max=0))
    tmp3 = []
    for x, y in zip(noise, up_noise):
        tmp3.append(np.clip(x.get_value() - np.sign(y) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX), a_min=-10, a_max=0))
        
    return tmp, tmp2, tmp3


def update_lambda_lr_every_meta_mom(ori, up, lam_vels, llr, up_llr, llr_vels, hyper_lr, lr_Llr, hyper_mom, llr_mom, updatemode='output_unit'):
    tmp = []
    hyper_mom = np.array(float(hyper_mom) * 1., dtype=theano.config.floatX)
    for x, y, v in zip(ori, up, lam_vels):
        if updatemode == 'unit' or (updatemode == 'output_unit' and x.name == 'output.L2'):
            new_y = np.mean(y, axis=1, keepdims=True)
            v.set_value(hyper_mom*v.get_value(borrow=True) - (1.-hyper_mom)*np.sign(new_y))
            tmp.append(np.clip(x.get_value(borrow=True) + v.get_value(borrow=True) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX), a_min=-7., a_max=-2.))
        else: #per parameter
            v.set_value(hyper_mom*v.get_value(borrow=True) - (1.-hyper_mom)*np.sign(y))
            tmp.append(np.clip(x.get_value(borrow=True) + v.get_value(borrow=True) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX), a_min=-7., a_max=-2.))
    tmp2 = []
    llr_mom = np.array(float(llr_mom) * 1., dtype=theano.config.floatX)
    for x, y, v in zip(llr, up_llr, llr_vels):
        if updatemode == 'unit' or (updatemode == 'output_unit' and x.name == 'output.W_llr'):
            new_y = np.mean(y, axis=1, keepdims=True)
            v.set_value(llr_mom*v.get_value(borrow=True) - (1.-llr_mom)*np.sign(new_y))
            tmp2.append(np.clip(x.get_value(borrow=True) + v.get_value(borrow=True) * np.array(float(lr_Llr) * 1., dtype=theano.config.floatX), a_min=-16., a_max=0.))
        else: #per parameter
            v.set_value(llr_mom*v.get_value(borrow=True) - (1.-llr_mom)*np.sign(y))
            tmp2.append(np.clip(x.get_value(borrow=True) + v.get_value(borrow=True) * np.array(float(lr_Llr) * 1., dtype=theano.config.floatX), a_min=-16., a_max=0.))
    return tmp, tmp2


def rt(interval, mult):
    return range(0,len(interval)*mult, mult)
      
def save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses, train_errors, valid_errors, test_errors, norms, theta_names, label='Epoch', ItersPerUp = -1):
    if label=='Epoch':
        x = 1
    elif label == 'Iteration':
        x = ItersPerUp
    
    # all losses in one plot
    for i in range(0, len(test_losses)): #for each meta-iteration
        plt.plot(rt(unreg_train_losses[i], x), unreg_train_losses[i], label="Unreg tra "+str(i))
        plt.plot(rt(train_losses[i], x), train_losses[i], label="Train "+str(i))
        plt.plot(rt(valid_losses[i], x), valid_losses[i], label="Valid "+str(i))
        plt.plot(rt(test_losses[i], x), test_losses[i], label="Test "+str(i))
    plt.title("Loss vs "+label)
    plt.xlabel(label)
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    # all errors in one plot
    for i in range(0, len(test_errors)): #for each meta-iteration
        plt.plot(rt(train_errors[i], x), train_errors[i], label="Train "+str(i))
        plt.plot(rt(valid_errors[i], x), valid_errors[i], label="Valid "+str(i))
        plt.plot(rt(test_errors[i], x), test_errors[i], label="Test "+str(i))
    plt.title("Error vs "+label)
    plt.xlabel(label)
    plt.ylabel("Error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/errors.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/errors.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    #each loss type separately, but for all meta-iters
    for i in range(0, len(test_losses)):
        plt.plot(rt(unreg_train_losses[i], x), unreg_train_losses[i], label="Meta "+str(i))
    plt.title("Unreg train loss vs "+label)
    plt.xlabel(label)
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/unreg_train_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/unreg_train_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_losses)):
        plt.plot(rt(train_losses[i], x), train_losses[i], label="Meta "+str(i))
    plt.title("Train loss vs "+label)
    plt.xlabel(label)
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/train_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/train_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_losses)):
        plt.plot(rt(valid_losses[i], x), valid_losses[i], label="Meta "+str(i))
    plt.title("Valid loss vs "+label)
    plt.xlabel(label)
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/valid_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/valid_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_losses)):
        plt.plot(rt(test_losses[i], x), test_losses[i], label="Meta "+str(i))
    plt.title("Test loss vs "+label)
    plt.xlabel(label)
    plt.ylabel("Loss")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/test_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/test_losses.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    #each error type separately, but for all meta-iters
    for i in range(0, len(test_errors)):
        plt.plot(rt(train_errors[i], x), train_errors[i], label="Meta "+str(i))
    plt.title("Train error vs "+label)
    plt.xlabel(label)
    plt.ylabel("Error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/train_errors.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/train_errors.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_errors)):
        plt.plot(rt(test_errors[i], x), test_errors[i], label="Meta "+str(i))
    plt.title("Valid error vs "+label)
    plt.xlabel(label)
    plt.ylabel("Error")
    lgd = plt.legend(bbox_to_anchor=(1.9, 1.025), loc='upper right', ncol=2)
    plt.savefig(outdir+"/valid_errors.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outdir+"/valid_errors.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    for i in range(0, len(test_errors)):
        plt.plot(rt(test_errors[i], x), test_errors[i], label="Meta "+str(i))
    plt.title("Test error vs "+label)
    plt.xlabel(label)
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
per_layer_llr = [[np.mean(llr.get_value())/np.log(10.)] for llr in log_learning_rates] #log_10 scale
per_layer_log_noise = [[np.mean(logsigma.get_value())] for logsigma in model.params_noise]


#TODO: save initial parameters/llrs (and reg?), and reinitialize these, and velocities

for metameta in range(args.metaEpoch):
    print("Meta-meta iter "+str(metameta))
    
    epochs = min(int(args.maxEpochInit*(args.maxEpochMult**float(metameta))), args.maxMaxEpoch)
    
    
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
    if args.onlineItersPerUpdate >= 20:
        label = 'Iteration'
        mult = args.onlineItersPerUpdate
    else:
        label = 'Epoch'
        mult = 1
    save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses,
        train_errors, valid_errors, test_errors, norms, theta_names, label, args.onlineItersPerUpdate)
    
    
    t_start = time()
    ele_iter = 0
    curr_epoch = 0
    print('elementary iterations before restarting', epochs*n_batch_ele)
    while ele_iter < epochs*n_batch_ele:
        temp_lambda, temp_llrs, temp_noise = run_exp(args, temp_lambda, temp_llrs, temp_noise, ele_iter)
        ele_iter = ele_iter+args.onlineItersPerUpdate
        if args.momLlr == 0.:
            temp_lambda, temp_llrs, temp_noise = update_lambda_lr_every_meta(model.params_lambda, temp_lambda, log_learning_rates, temp_llrs, model.params_noise, temp_noise, args.lrHyper, args.lrLlr)
        #else: #TODO: noise
        #    temp_lambda, temp_llrs = update_lambda_lr_every_meta_mom(model.params_lambda, temp_lambda, lambda_velocities, log_learning_rates, temp_llrs, llr_velocities, args.lrHyper, args.lrLlr, args.momHyper, args.momLlr)
        
        prev_epoch = curr_epoch
        curr_epoch = ele_iter // n_batch_ele
        #Put all the testing/saving here
        if label=='Iteration' or (label=='Epoch' and curr_epoch > prev_epoch):
            #TODO: need to change plot labels now
            #if ele_iter + args.onlineItersPerUpdate >= epochs*n_batch_ele: #if last iteration of the meta-meta-iter
            #    n_eval = 5000 #too big
            #else:
            n_eval = 700

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
            print("Eval on Train Set: Epoch %d, loss = %.4f, unreg loss = %.4f, error = %.4f" %
                  (curr_epoch, train_loss, unreg_train_loss, train_error))
            unreg_train_losses[-1].append(float(unreg_train_loss))
            train_losses[-1].append(float(train_loss))
            train_errors[-1].append(train_error)

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
            print("Eval on Valid Set: Epoch %d, time = %ds, loss = %.4f, error = %.4f" %
                  (curr_epoch, time() - t_start, valid_loss, valid_error))
            valid_losses[-1].append(float(valid_loss))
            valid_errors[-1].append(valid_error)

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
            print("Eval on Test Set: Epoch %d, time = %ds, loss = %.4f, error = %.4f" %
                  (curr_epoch, time() - t_start, test_loss, test_error))
            test_losses[-1].append(float(test_loss))
            test_errors[-1].append(test_error)
            
            for k, param in enumerate(model.params_theta):
                norms[k+len(model.params_theta)*metameta].append(np.linalg.norm(param.get_value(borrow=True)))
            
            #Save and save plots
            save_errors_losses_norms(outdir, unreg_train_losses, train_losses, valid_losses, test_losses,
                train_errors, valid_errors, test_errors, norms, theta_names, label, args.onlineItersPerUpdate)
            
            if ele_iter + args.onlineItersPerUpdate >= epochs*n_batch_ele:
                #TODO: save these less frequently
                f=open(outdir+'/lambda.pckl', 'wb')  # Python 2: open(..., 'w')
                pickle.dump(temp_lambda, f)
                f.close()
                f=open(outdir+'/log_learning_rates.pckl', 'wb')  # Python 2: open(..., 'w')
                pickle.dump(temp_llrs, f)
                f.close()
                f=open(outdir+'/lambda_vels.pckl', 'wb')  # Python 2: open(..., 'w')
                pickle.dump(lambda_velocities, f)
                f.close()
                f=open(outdir+'/llr_vels.pckl', 'wb')  # Python 2: open(..., 'w')
                pickle.dump(llr_velocities, f)
                f.close()
                f=open(outdir+'/theta.pckl', 'wb')  # Python 2: open(..., 'w')
                pickle.dump(model.params_theta, f)
                f.close()
                if args.addActivationNoise or args.addInputNoise:
                    f=open(outdir+'/log_noise.pckl', 'wb')  # Python 2: open(..., 'w')
                    pickle.dump(temp_noise, f)
                    f.close()
            
            
            #per-layer average log regularization
            for j, reg, reg2 in zip(range(0,len(temp_lambda)),temp_lambda, model.params_lambda):
                per_layer_log_reg[j].append(np.mean(reg))
                plt.plot(rt(per_layer_log_reg[j], mult), per_layer_log_reg[j], label=reg2.name)
            plt.title("Per-layer average log reg hyper vs "+label)
            plt.xlabel(label)
            plt.ylabel("Average log reg hyper")
            lgd = plt.legend(bbox_to_anchor=(2, 1.025), loc='upper right', ncol=2)
            plt.savefig(outdir+"/reg.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.savefig(outdir+"/reg.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()
            f=open(outdir+'/layer_log_regs.pckl', 'wb')  # Python 2: open(..., 'w')
            pickle.dump(per_layer_log_reg, f)
            f.close()
            
            if args.addActivationNoise or args.addInputNoise:
                #print(per_layer_log_noise)
                #per-layer average log noise sigma
                for j, reg, reg2 in zip(range(0,len(temp_noise)), temp_noise, model.params_noise):
                    per_layer_log_noise[j].append(np.mean(reg))
                    plt.plot(rt(per_layer_log_noise[j], mult), per_layer_log_noise[j], label=reg2.name)
                plt.title("Per-layer inverse sigmoid noise sigma vs "+label)
                plt.xlabel(label)
                plt.ylabel("Average inverse sigmoid noise sigma")
                lgd = plt.legend(bbox_to_anchor=(2, 1.025), loc='upper right', ncol=2)
                plt.savefig(outdir+"/noise.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.savefig(outdir+"/noise.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.clf()
                f=open(outdir+'/layer_log_noise.pckl', 'wb')  # Python 2: open(..., 'w')
                pickle.dump(per_layer_log_noise, f)
                f.close()
            
            #per-layer average log learning rate
            for j, llr, llr2 in zip(range(0,len(temp_llrs)), temp_llrs, log_learning_rates):
                per_layer_llr[j].append(np.mean(llr)/np.log(10.))
                plt.plot(rt(per_layer_llr[j], mult), per_layer_llr[j], label=llr2.name)
            plt.title("Per-layer average log learning rate vs "+label)
            plt.xlabel(label)
            plt.ylabel("Average log learning rate")
            lgd = plt.legend(bbox_to_anchor=(2.3, 1.025), loc='upper right', ncol=2)
            plt.savefig(outdir+"/llrs.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.savefig(outdir+"/llrs.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()
            f=open(outdir+'/layer_llrs.pckl', 'wb')  # Python 2: open(..., 'w')
            pickle.dump(per_layer_llr, f)
            f.close()
            
        #curr_epoch = ele_iter // n_batch_ele
        
        
    if metameta < args.metaEpoch - 1: #for all but the last meta-meta-iteration
        #TODO: reinitilialize the elementary parameters, velocities and learning rates, but NOT regularization
        for param, ori in zip(model.params_theta, theta_initial_initial):
            param.set_value(ori)
        for v in velocities:
            v.set_value(np.asarray(v.get_value(borrow=True)*0., dtype=theano.config.floatX))
        for llr in log_learning_rates:
            llr.set_value(np.full_like(llr.get_value(borrow=True), np.log(args.lrEle)))
        for lamb in model.params_lambda: #TODO: reinitialize the L2 hypers with the layerwise median to avoid having to deal with symmetry breaking
        #median rather than average for robustness, and simply because it minimizes the total distance, rather than squared distance, so it should be fastest to return to values before taking median?
            lamb.set_value(np.full_like(lamb.get_value(borrow=True), np.median(lamb.get_value(borrow=True))))
        for llr_v in llr_velocities:
            llr_v.set_value(np.asarray(llr_v.get_value(borrow=True)*0., dtype=theano.config.floatX))
        for lamb_v in lambda_velocities:
            lamb_v.set_value(np.asarray(lamb_v.get_value(borrow=True)*0., dtype=theano.config.floatX))
        for j, reg in enumerate(model.params_lambda):
            per_layer_log_reg[j].append(np.mean(reg.get_value(borrow=True)))
        for j, llr in enumerate(log_learning_rates):
            per_layer_llr[j].append(np.log(args.lrEle)/np.log(10.)) #log10 scale
        if args.addActivationNoise or args.addInputNoise:
            for ls in model.params_noise:
                ls.set_value(np.full_like(ls.get_value(borrow=True), np.log10(0.1))) #TODO: hardcoded initial noise
            for j, ls in enumerate(model.params_noise):
                per_layer_log_noise[j].append(args.invSigmoidActivationNoiseMagnitude) #TODO: separate for input
    print("---------------------------------------------------------------------------------------------------")
    print("Training Result: ")
    print("Test loss: "+str(test_losses[-1][-1]))
    print("Test error: "+str(test_errors[-1][-1]))
    print("---------------------------------------------------------------------------------------------------")
