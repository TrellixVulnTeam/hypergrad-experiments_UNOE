"""
    This module is to collect and calculate updates for different theano function.

"""
import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd, momentum
from hypergrad import hypergrad
from pprint import pprint
from collections import OrderedDict

#TODO: update Jul30 6pm added clipping for values (not gradients)


# Mihael: compute gradient; is there any "stochastic" part in here? 
def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for
    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.
    Raises
    ------
    ValueError
        If `loss_or_grads` is a list of a different length than `params`, or if
        any element of `params` is not a shared variable (while we could still
        compute its gradient, we can never update it and want to fail early).
    """
    for p in params:
        if not isinstance(p, theano.compile.SharedVariable):
            print(p)
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)



"""
xs = []
for i in range(0,2):
    x = theano.shared(i)
    xs.append(x)
"""

# Michael: this is one SGD update step
# How does this work? Each call creates new updates list, and new velocity variables initialized to 0
def custom_mom_old(loss_or_grads, params, learning_rate, momentum=0.9, clip=3.):

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True) #borrow=True the original data, not a copy, is returned
        # TODO: the velocity is shared outside, but reinitialized each time?
        # but this is a different velocity for each (param, grad)? 
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable, name=param.name+'_velocity') #same name?
        updates[velocity] = momentum * velocity - (1. - momentum) * grad #velocity is 0?
        updates[param] = T.clip(param + learning_rate * velocity, -10.**clip, 10.**clip)
        #TODO: Theano updates mean the velocity used to update the param is the original one, not the one calculated here
    return updates #becomes update_ele in 'update' below


def custom_mom(loss_or_grads, params, velocities, learning_rate, momentum=0.9, clip=3.):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    for param, velocity, grad in zip(params, velocities, grads):
        new_velocity = momentum * velocity - (1. - momentum) * grad
        updates[velocity] = new_velocity
        updates[param] = T.clip(param + learning_rate * new_velocity, -10.**clip, 10.**clip) #use new velocity here
    return updates #becomes update_ele in 'update' below

#TODO: can't use clipping for exact reverse path
def custom_mom2(loss_or_grads, params, velocities, llrs, momentum=0.9):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    for param, velocity, grad, llr in zip(params, velocities, grads, llrs):
        new_velocity = momentum * velocity - (1. - momentum) * grad
        updates[velocity] = new_velocity
        updates[param] = param + T.exp(llr) * new_velocity #use new velocity here
    return updates #becomes update_ele in 'update' below

def custom_mom_Nesterov(loss_or_grads, params, velocities, learning_rate, momentum=0.9, clip=3.):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    #v = mu * v - learning_rate * dx
    #x += -mu * v_prev + (1 + mu) * v
    #https://arxiv.org/pdf/1212.0901v2.pdf
    #http://cs231n.github.io/neural-networks-3/#sgd
    for param, velocity, grad in zip(params, velocities, grads):
        new_velocity = momentum * velocity - learning_rate * grad
        updates[param] = T.clip(param - learning_rate * velocity + (1. + momentum) * new_velocity, -10.**clip, 10.**clip)
        updates[velocity] = new_velocity
    return updates #becomes update_ele in 'update' below


"""def custom_mom_reverse(loss, params, velocities, learning_rate, momentum=0.9):
    updates = OrderedDict()
    new_params =[]
    for param, velocity in zip(params, velocities): #reverse parameter updates
        new_param = param - learning_rate * velocity
        updates[param] = new_param
        new_params.append(new_param)
    grads = theano.grad(loss, new_params)
    for velocity, grad in zip(velocities, grads): #reverse velocity update
        updates[velocity] = (velocity + (1-momentum)*grad)/momentum
    return updates
"""
#Above gives disconnected error, since the loss doesn't actually depend on new_params



#TODO: grad_dloss_dweight is used as dv_t. Is this correct? It gets updated to dloss_dweight
def update(params_theta, params_lambda, params_weight, velocities, loss, penalty, lossWithPenalty,
           lr_ele, mom):
    # (1) phase 1
    update_ele = custom_mom_Nesterov(lossWithPenalty, params_theta, velocities, lr_ele, momentum=mom)

    #The rest is used for grad_l_theta = dw_T = gradient of validation loss wrt final weights
    dloss_dweight = T.grad(loss, params_weight)
    
    #print("type 1,", type(dloss_dweight[0]))
    #try_1 = T.grad(penalty, params_weight, disconnected_inputs='ignore') #some layers aren't regularized, so are "Disconnected"
    #print("type 2", type(try_1[0]))
    #try_2 = T.grad(lossWithPenalty, params_weight)
    #print("type 3", type(try_2[0]))
    #try_3 = [-grad for grad in dloss_dweight]
    #print("type 4", type(try_3[0]))

    # (2) calc updates for Phase 2
    # in this phase, no params is updated. Only dl_dweight will be saved.

    update_valid = []
    grad_dloss_dweight = []
    for param, grad in zip(params_weight, dloss_dweight):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  broadcastable=param.broadcastable,
                                  name='dl_dweight_%s' % param.name)
        update_valid += [(save_grad, grad)]
        grad_dloss_dweight += [save_grad]

    return update_ele, update_valid, grad_dloss_dweight, dloss_dweight
    # output_valid_list = grad_dloss_dweight = [save_grad's], which should get updated to [grad's] = dloss_dweight 
    # after update_valid is used by calling func_hyper_valid


def update2(params_theta, params_lambda, params_weight, velocities, loss, penalty, lossWithPenalty,
           llr_ele, mom):
    # (1) phase 1
    # it's a simple MLP, we can use lasagne's routine to calc the updates
    update_ele = custom_mom2(lossWithPenalty, params_theta, velocities, llr_ele, momentum=mom)

    #The rest is used for grad_l_theta = dw_T = gradient of validation loss wrt final weights
    dloss_dtheta = T.grad(loss, params_theta)
 
    # (2) calc updates for Phase 2
    # in this phase, no params is updated. Only dl_dtheta will be saved.

    update_valid = []
    grad_dloss_dtheta = []
    for param, grad in zip(params_theta, dloss_dtheta):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  broadcastable=param.broadcastable,
                                  name='dl_dtheta_%s' % param.name)
        update_valid += [(save_grad, grad)]
        grad_dloss_dtheta += [save_grad]

    return update_ele, update_valid, grad_dloss_dtheta, dloss_dtheta







#def updates_hyper(params_lambda, params_weight, lossWithPenalty, grad_l_theta, share_var_dloss_dweight):
def updates_hyper(params_lambda, params_weight, lossWithPenalty, share_var_dloss_dweight):
    #share_var_dloss_dweight should be dv_t ??
    # (3) meta_backward
    update_hyper = []
    HVP_weight = []
    HVP_lambda = []
    #grad_valid_weight = [] #TODO: doesn't seem to be used after initialization?
    dLoss_dweight = T.grad(lossWithPenalty, params_weight) # the TensorVariable, no value yet
    #print("where's the tensorvariable comes from?", type(dLoss_dweight[0]), type(params_weight))
    """for grad in grad_l_theta: 
        #print(type(grad))
        #TODO: Do updates to grad_valid_weight mean updates to grad_l_theta? Probably not, or else grad_valid_weight wolud be redundant
        save_grad = theano.shared(np.asarray(grad, dtype=theano.config.floatX),
                                  name='grad_L_theta_'+str(grad.shape))
        grad_valid_weight += [save_grad]"""


    HVP_weight_temp, HVP_lambda_temp = hypergrad(params_lambda, params_weight,
                                          dLoss_dweight, share_var_dloss_dweight) #the TensorVariables, no values yet
    #print"weights", (len(params_weight), len(HVP_weight_temp))
    
    #TODO: what are these updates for? storing HVPs?     
    #store HVP_weight_temp and HVP_lambda_temp in 
    #print('Inside updates_hyper')
    for param, grad in zip(params_weight, HVP_weight_temp):
        #tensor of zeros of same shape and broadcastable as param
        #without broadcastable, this changed the type from TensorType(float32, (False, False, True, True)) to TensorType(float32, 4D)
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  name='HVP_weight_'+param.name, broadcastable=param.broadcastable)
        #save_grad = theano.shared(T.zeros_like(param))
        #save_grad.name = param.name+'_HVP_weight'
        update_hyper += [(save_grad, grad)]
        HVP_weight += [save_grad]
        #print(param.name, np.linalg.norm(grad.get_value())) #AttributeError: 'TensorVariable' object has no attribute 'get_value'

    #print("lambda", len(params_lambda), len(HVP_lambda_temp))
    for param, grad in zip(params_lambda, HVP_lambda_temp):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  name='HVP_lambda_'+param.name, broadcastable=param.broadcastable)
        #save_grad = theano.shared(T.zeros_like(param))
        #save_grad.name = param.name+'HVP_lambda'
        update_hyper += [(save_grad, grad)]
        HVP_lambda += [save_grad]
        #print(param.name, np.linalg.norm(grad.get_value()))
    
    #output_hyper_list = HVP_weight_temp + HVP_lambda_temp #concatenation
    #return update_hyper, output_hyper_list, share_var_dloss_dweight
    return update_hyper, HVP_weight, HVP_lambda, share_var_dloss_dweight





def updates_hyper_no(params_lambda, params_weight, lossWithPenalty, share_var_dloss_dweight):
    #share_var_dloss_dweight should be dv_t
    # (3) meta_backward
    dLoss_dweight = T.grad(lossWithPenalty, params_weight)
    HVP_weight_temp, HVP_lambda_temp = hypergrad(params_lambda, params_weight,
                                          dLoss_dweight, share_var_dloss_dweight)
    return HVP_weight_temp, HVP_lambda_temp, share_var_dloss_dweight















from hypergrad import Lop2

#Same as updates_hyper, but with HVP_weight replaced by HVP_theta, using all elementary parameters, not just the regularized weights
#HVP_lambda is the same as before, and only uses params_weight, rather than params_theta
def updates_hyper2(params_lambda, params_theta, params_weight, lossWithPenalty, share_var_dloss_dtheta):
    # (3) meta_backward
    update_hyper = []
    HVP_theta = []
    HVP_lambda = []
    dLoss_dtheta = T.grad(lossWithPenalty, params_theta)
    
    #isolate weights separately
    dLoss_dweight = []
    share_var_dloss_dweight = []
    j=0
    for w in params_weight:
        while params_theta[j] is not w:
            j=j+1
        dLoss_dweight.append(dLoss_dtheta[j])
        share_var_dloss_dweight.append(share_var_dloss_dtheta[j])
    print(len(dLoss_dweight), len(share_var_dloss_dweight), len(params_weight))
    print(len(dLoss_dtheta), len(share_var_dloss_dtheta), len(params_theta))
    #Use Lop2 here directly
    HVP_theta_temp = Lop2(dLoss_dtheta, params_theta, share_var_dloss_dtheta)
    HVP_lambda_temp = Lop2(dLoss_dweight, params_lambda, share_var_dloss_dweight)
    for param, grad in zip(params_theta, HVP_theta_temp):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  name='HVP_theta_'+param.name, broadcastable=param.broadcastable)
        update_hyper += [(save_grad, grad)]
        HVP_theta += [save_grad]

    for param, grad in zip(params_lambda, HVP_lambda_temp):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  name='HVP_lambda_'+param.name, broadcastable=param.broadcastable)
        update_hyper += [(save_grad, grad)]
        HVP_lambda += [save_grad]
    return update_hyper, HVP_theta, HVP_lambda, share_var_dloss_dtheta