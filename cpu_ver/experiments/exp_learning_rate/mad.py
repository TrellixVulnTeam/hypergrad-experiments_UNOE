import sys, os
sys.path.append(os.path.abspath('../../'))


# TODO: it looks like the training and test losses are the same?


"""Gradient descent to optimize everything,
subject to the constraint that weight initialization can't be set to zero,
and using a adam for the meta-optimization."""
import numpy as np
import numpy.random as npr
import pickle
from collections import defaultdict

from funkyyak import grad, kylist

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit
#from hypergrad.optimizers import sgd4_mad as sgd4, rms_prop, adam
from hypergrad.optimizers2 import sgd4_mad_with_exact as sgd4, adam_compare as adam

# ----- Fixed params -----
layer_sizes = [784, 10]
batch_size = 200
#N_iters = 50 #Michael: Far too small
N_iters = 500
#TODO: N_iters = 1 doesn't work, why? Parser wants to treat the rates as single items rather than singleton vectors? 
N_classes = 10
#Michael: MNIST has 70000
N_train = 10000 #Maybe increase this?
# N_valid = 10**3 #too small; overfitting to validation set?
N_valid = 5000 #new default
N_tests = 10**3
N_batches = 10 #N_train / batch_size
thin = np.ceil(N_iters/N_batches) #number of epochs ?
#N_iters = N_epochs * N_batches

# TODO: more regularization constants, no regularization on the bias, no learning rate/momentum/initial scale optimization
# Check other experiments

# N_iter, thin, N_batches, batch_size aren't consistent? 
# N_batches is not equal to N_train / batch_size; 
# it's just how many times we want to record elementary parameter statistics

# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = -4.0
init_log_alphas = -1.0
init_invlogit_betas = inv_logit(0.5)
init_log_param_scale = -2.0
# ----- Superparameters -----
meta_alpha = 0.04
N_meta_iter = 50

global_seed = npr.RandomState(3).randint(1000)


# TODO: check this
def fill_parser(parser, items):
    #for i, name in enumerate(parser.names):
     #   print(name, parser[name].size)
    partial_vects = [np.full(parser[name].size, items[i])
                     for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

def run():
    train_data, valid_data, tests_data = load_data_dicts(N_train, N_valid, N_tests)
    parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes) #only uses two different regularization hyperparameters, one for each layer?
    N_weight_types = len(parser.names) # = 2
    print(parser.names)
    hyperparams = VectorParser()
    hyperparams['log_L2_reg']      = np.full(N_weight_types, init_log_L2_reg)
    hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
    hyperparams['log_alphas']      = np.full(N_iters, init_log_alphas)
    hyperparams['invlogit_betas']  = np.full(N_iters, init_invlogit_betas)
    fixed_hyperparams = VectorParser()
    fixed_hyperparams['log_param_scale']  = np.full(N_iters, init_log_param_scale) #don't update scale
    #TODO: remove scale from gradient, then?
    
    exact_metagrad = VectorParser()
    exact_metagrad['log_L2_reg']      = fill_parser(parser, hyperparams['log_L2_reg']) #np.zeros(N_weight_types)
    exact_metagrad['log_param_scale'] = fill_parser(parser, fixed_hyperparams['log_param_scale']) #np.zeros(N_weight_types)
    exact_metagrad['log_alphas']      = np.zeros(N_iters)
    exact_metagrad['invlogit_betas']  = np.zeros(N_iters)
    
    exact_metagrad2 = VectorParser()
    exact_metagrad2['log_L2_reg']      = np.zeros(N_weight_types)
    exact_metagrad2['log_param_scale'] = np.zeros(N_weight_types)
    exact_metagrad2['log_alphas']      = np.zeros(N_iters)
    exact_metagrad2['invlogit_betas']  = np.zeros(N_iters)
    
    #exact_metagrad = exact_metagradV.vect
    #print(hyperparams.vect)
    #exact_metagrad = [np.zeros(N_weight_types), np.zeros(N_weight_types), np.zeros(N_iters), np.zeros(N_iters)] #initialize

    # TODO: memoize
    def primal_optimizer(hyperparam_vect, i_hyper):
        def indexed_loss_fun(w, L2_vect, i_iter):
            rs = npr.RandomState(npr.RandomState(global_seed + i_hyper + i_iter * 10000).randint(1000))
            seed = i_hyper * 10**6 + i_iter   # Deterministic seed needed for backwards pass.
            idxs = rs.randint(N_train, size=batch_size)
            return loss_fun(w, train_data['X'][idxs], train_data['T'][idxs], L2_vect)

        learning_curve_dict = defaultdict(list)
        def callback(x, v, g, i_iter):
            if i_iter % thin == 0: # N_batches=10 times
                learning_curve_dict['learning_curve'].append(loss_fun(x, **train_data))
                learning_curve_dict['grad_norm'].append(np.linalg.norm(g))
                learning_curve_dict['weight_norm'].append(np.linalg.norm(x))
                learning_curve_dict['velocity_norm'].append(np.linalg.norm(v))

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        # TODO: why doesn't the following line work with N_iter=1?
        W0 = fill_parser(parser, np.exp(fixed_hyperparams['log_param_scale'])) #don't update scale
        W0 *= npr.RandomState(global_seed + i_hyper).randn(W0.size)
        # TODO: Put on proper scale; no SGD on log/invlogit scale
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas  = logit(cur_hyperparams['invlogit_betas'])
        
        # TODO: check this
        L2_reg = fill_parser(parser, np.exp(cur_hyperparams['log_L2_reg']))
        
        W_opt = sgd4(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg), exact_metagrad, callback)
        #W_opt = sgd4(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg), callback)
        #callback(W_opt, N_iters)
        return W_opt, learning_curve_dict

    def hyperloss(hyperparam_vect, i_hyper):
        W_opt, _ = primal_optimizer(hyperparam_vect, i_hyper)
        return loss_fun(W_opt, **valid_data)
    hyperloss_grad = grad(hyperloss)
    # TODO: This is where the chain rule happens, dhyperloss/dW_opt x dW_opt/dhyperparam_vect; first term is SGD

    meta_results = defaultdict(list)
    old_metagrad = [np.ones(hyperparams.vect.size)]
    #def meta_callback(hyperparam_vect, i_hyper, metagrad):
    def meta_callback(hyperparam_vect, i_hyper, metagrad, exact_metagrad=exact_metagrad):
        x, learning_curve_dict = primal_optimizer(hyperparam_vect, i_hyper)
        cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
        for field in cur_hyperparams.names:
            meta_results[field].append(cur_hyperparams[field])
        # these are the unregularized losses below; default sets L2_reg=0.0
        meta_results['train_loss'].append(loss_fun(x, **train_data))
        meta_results['valid_loss'].append(loss_fun(x, **valid_data))
        meta_results['tests_loss'].append(loss_fun(x, **tests_data))
        meta_results['train_err'].append(frac_err(x, **train_data))
        meta_results['valid_err'].append(frac_err(x, **valid_data))
        meta_results['test_err'].append(frac_err(x, **tests_data))
        meta_results['learning_curves'].append(learning_curve_dict)
        print("metagrad", len(metagrad))
        meta_results['meta_grad_magnitude'].append(np.linalg.norm(metagrad))
        meta_results['meta_grad_angle'].append(np.dot(old_metagrad[0], metagrad) \
                                               / (np.linalg.norm(metagrad)*
                                                  np.linalg.norm(old_metagrad[0])))
        #Michael: added comparisons with exact metagrad here
        #(2) Angle condition:  More strongly, is the cosine of the angle between the two strictly bounded away from 0?
        #(3) Length: Since hypergradient optimization procedures do not necessarily use a proper line search, it may also be important for the approximate hypergradient to have a length comparable to the true hypergradient.
        
        
        exact_metagrad2['log_L2_reg']      = [sum(exact_metagrad['log_L2_reg'][0:7840]), sum(exact_metagrad['log_L2_reg'][7840:7850])]
        exact_metagrad2['log_param_scale'] = [sum(exact_metagrad['log_param_scale'][0:7840]), sum(exact_metagrad['log_param_scale'][7840:7850])]
        exact_metagrad2['log_alphas']      = exact_metagrad['log_alphas']
        exact_metagrad2['invlogit_betas']  = exact_metagrad['invlogit_betas']
    
        meta_results['exact_meta_grad_magnitude'].append(np.linalg.norm(exact_metagrad2.vect))
        meta_results['DrMAD_exact_angle'].append(np.dot(exact_metagrad2.vect, metagrad) \
                                               / (np.linalg.norm(metagrad)*
                                                  np.linalg.norm(exact_metagrad2.vect)))
    
        #TODO: do the above for parameters separately? E.g. check log_alphas separately
                                                  
        old_metagrad[0] = metagrad
        print "Meta Epoch {0} Train loss {1:2.4f} Valid Loss {2:2.4f}" \
              " Test Loss {3:2.4f} Test Err {4:2.4f}".format(
            i_hyper, meta_results['train_loss'][-1], meta_results['valid_loss'][-1],
            meta_results['tests_loss'][-1], meta_results['test_err'][-1])  #Michael: train->tests
#    final_result = adam(hyperloss_grad, hyperparams.vect,
#                            meta_callback, N_meta_iter, meta_alpha)
    final_result = adam(hyperloss_grad, hyperparams.vect, exact_metagrad,
                            meta_callback, N_meta_iter, meta_alpha)
    #write modified adam to ignore exact hypergrad in sgd4_mad_with_exact
    #meta_callback(final_result, N_meta_iter)
    parser.vect = None # No need to pickle zeros
    return meta_results, parser


def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser = pickle.load(f)

    # ----- Nice versions of Alpha and beta schedules for paper -----
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(211)
    #ax.set_title('Alpha learning curves')
    ax.plot(np.exp(results['log_alphas'][-1]), 'o-', label="Step size")
    #ax.set_xlabel('Learning Iteration', fontproperties='serif')
    low, high = ax.get_ylim()
    ax.set_ylim([0, high])
    ax.set_ylabel('Step size', fontproperties='serif')
    ax.set_xticklabels([])

    ax = fig.add_subplot(212)
    #ax.set_title('Alpha learning curves')
    ax.plot(logit(results['invlogit_betas'][-1]), 'go-', label="Momentum")
    low, high = ax.get_ylim()
    ax.set_ylim([low, 1])
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum', fontproperties='serif')

    #ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
    #          prop={'family':'serif', 'size':'12'})
    fig.set_size_inches((6,3))
    #plt.show()
    plt.savefig('alpha_beta_paper.png')
    plt.savefig('alpha_beta_paper.pdf', pad_inches=0.05, bbox_inches='tight')

    fig.clf()
    fig.set_size_inches((6,8))
    # ----- Primal learning curves -----
    ax = fig.add_subplot(511)
    ax.set_title('Primal learning curves')
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['learning_curve'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Negative log prob')
    #ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(512)
    ax.set_title('Meta learning curves (unregularized losses)')
    losses = ['train_loss', 'valid_loss', 'tests_loss']
    for loss_type in losses:
        ax.plot(results[loss_type], 'o-', label=loss_type)
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Negative log prob')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(513)
    ax.set_title('Meta error curves')
    errors = ['train_err', 'valid_err', 'test_err']
    for err_type in errors:
        ax.plot(results[err_type], 'o-', label=err_type)
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Prediction error')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(514)
    ax.set_title('Meta-gradient magnitude')
    ax.plot(results['meta_grad_magnitude'], 'o-', label='Meta-gradient magnitude')
    ax.plot(results['exact_meta_grad_magnitude'], 'o-', label='Exact meta-gradient magnitude')
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Meta-gradient Magnitude')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(515)
    ax.plot(results['meta_grad_angle'], 'o-', label='Current vs previous Meta-gradient angle')
    ax.plot(results['DrMAD_exact_angle'], 'o-', label='DrMAD vs exact meta-gradient angle')
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Meta-gradient Magnitude')
    ax.legend(loc=1, frameon=False)


    plt.savefig('learning_curves.png')

    # ----- Learning curve info -----
    fig.clf()
    ax = fig.add_subplot(311)
    ax.set_title('Primal learning curves')
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['grad_norm'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    #ax.legend(loc=1, frameon=False)
    ax.set_title('Grad norm')

    ax = fig.add_subplot(312)
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['weight_norm'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    #ax.legend(loc=1, frameon=False)
    ax.set_title('Weight norm')

    ax = fig.add_subplot(313)
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['velocity_norm'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    ax.set_title('Velocity norm')
    #ax.legend(loc=1, frameon=False)
    plt.savefig('extra_learning_curves.png')

    # ----- Alpha and beta schedules -----
    fig.clf()
    ax = fig.add_subplot(211)
    ax.set_title('Alpha learning curves')
    for i, y in enumerate(results['log_alphas']):
        ax.plot(y, 'o-', label="Meta iter {0}".format(i))
    ax.set_xlabel('Primal iter number')
    #ax.set_ylabel('Log alpha')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(212)
    ax.set_title('Beta learning curves')
    for y in results['invlogit_betas']:
        ax.plot(y, 'o-')
    ax.set_xlabel('Primal iter number')
    ax.set_ylabel('Inv logit beta')
    plt.savefig('alpha_beta_curves.png')

    # ----- Init scale and L2 reg -----
    fig.clf()
    # TODO: this didn't change?
    ax = fig.add_subplot(211)
    ax.set_title('Init scale learning curves')
    for i, y in enumerate(zip(*results['log_param_scale'])):
        ax.plot(y, 'o-', label=parser.names[i])
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Log param scale')
    ax.legend(loc=1, frameon=False)

    # TODO: regularized biases???
    ax = fig.add_subplot(212)
    ax.set_title('L2 reg learning curves')
    for i, y in enumerate(zip(*results['log_L2_reg'])):
        ax.plot(y, 'o-', label=parser.names[i])
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Log L2 reg')
    plt.savefig('scale_and_reg.png')
    
    



if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()

#plot()