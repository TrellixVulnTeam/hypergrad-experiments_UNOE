import matplotlib.pyplot as plt


import operator as op
import itertools as it
import numpy as np
from functools import partial
from collections import deque
from funkyyak import grad, Differentiable
from exact_rep import ExactRep
from scipy.optimize import minimize
from nn_utils import fill_parser

RADIX_SCALE = 2**52

#TODO: mad methods use ExactRep unnecessarily

def sgd(loss_fun, batches, N_iter, x, v, alphas, betas, record_learning_curve=False):
    # TODO: Warp alpha and beta to map from real-valued domains (exp and logistic?)
    def print_perf():
        pass
        if (i + 1) % iter_per_epoch == 0:
            print "End of epoch {0}: loss is {1}".format(i / iter_per_epoch,
                                                         loss_fun(X.val, batches.all_idxs))

    X, V = ExactRep(x), ExactRep(v)
    x_orig = X.val
    iter_per_epoch = len(batches)
    num_epochs = N_iter / len(batches) + 1
    iters = zip(range(N_iter), alphas, betas, batches * num_epochs)
    loss_grad = grad(loss_fun)
    loss_hvp = grad(lambda x, d, idxs: np.dot(loss_grad(x, idxs), d))
    learning_curve = [loss_fun(x_orig, batches.all_idxs)]
    for i, alpha, beta, batch in iters:
        V.mul(beta)
        g = loss_grad(X.val, batch)
        V.sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        if record_learning_curve and (i + 1) % iter_per_epoch == 0:
            learning_curve.append(loss_fun(X.val, batches.all_idxs))
            # print_perf()

    x_final = X.val
    d_x = loss_grad(X.val, batches.all_idxs)
    loss_final = loss_fun(x_final, batches.all_idxs)
    d_v = np.zeros(d_x.shape)
    d_alphas = deque()
    d_betas = deque()
    print_perf()

    for i, alpha, beta, batch in iters[::-1]:
        print_perf()
        d_v += d_x * alpha
        X.sub(alpha * V.val)
        g = loss_grad(X.val, batch)
        d_alphas.appendleft(np.dot(d_x, V.val))
        V.add((1.0 - beta) * g)
        V.div(beta)
        d_betas.appendleft(np.dot(d_v, V.val + g))
        d_x = d_x - (1.0 - beta) * loss_hvp(X.val, d_v, batch)
        d_v = d_v * beta

    d_alphas = np.array(d_alphas)
    d_betas = np.array(d_betas)

    # print "-"*80
    assert np.all(x_orig == X.val)
    return {'x_final': x_final,
            'learning_curve': learning_curve,
            'loss_final': loss_final,
            'd_x': d_x,
            'd_v': d_v,
            'd_alphas': d_alphas,
            'd_betas': d_betas}


def sgd2(optimizing_loss, secondary_loss, batches, N_iter, x0, v0, alphas, betas, meta):
    """
    This version takes a secondary loss, and also returns gradients w.r.t. the data.
    :param optimizing_loss: The loss to be optimized by SGD.
    The first argument must be the parameters, the second must be the metaparameters,
    the third is data indicies.
    :param secondary_loss: Another loss we want to compute the gradient wrt.
    It takes parameters and metaparameters.
    :param batches: A list of slices into the data.
    :param N_iter: Number of iterations of SGD.
    :param x0: Starting parameter values.
    :param v0: Starting velocity.  Should probably be zero.
    :param alphas: Stepsize schedule.
    :param betas: Drag schedule.
    :param meta: A second parameter of the loss function that doesn't get optimized here.
    :return:
    a dict containing:
    Gradients wrt x0, v0, alphas, beta, and meta.
    """

    # TODO: Warp alpha and beta to map from real-valued domains (exp and logistic?)
    def print_perf():
        pass
        if (i + 1) % iter_per_epoch == 0:
            print "End of epoch {0}: loss is {1}".format(i / iter_per_epoch,
                                                         optimizing_loss(X.val, meta, batches.all_idxs))

    X, V = ExactRep(x0), ExactRep(v0)
    x_orig = X.val
    iter_per_epoch = len(batches)
    num_epochs = N_iter / len(batches) + 1
    iters = zip(range(N_iter), alphas, betas, batches * num_epochs)
    L_grad = grad(optimizing_loss)  # Gradient wrt parameters.
    M_grad = grad(secondary_loss)  # Gradient wrt parameters.
    L_meta_grad = grad(optimizing_loss, 1)  # Gradient wrt metaparameters.
    M_meta_grad = grad(secondary_loss, 1)  # Gradient wrt metaparameters.
    L_hvp = grad(lambda x, d, idxs:
                 np.dot(L_grad(x, meta, idxs), d))  # Hessian-vector product.
    L_hvp_meta = grad(lambda x, meta, d, idxs:
                      np.dot(L_grad(x, meta, idxs), d), 1)  # Returns a size(meta) output.

    learning_curve = [optimizing_loss(X.val, meta, batches.all_idxs)]
    for i, alpha, beta, batch in iters:
        V.mul(beta)
        g = L_grad(X.val, meta, batch)
        V.sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        learning_curve.append(optimizing_loss(X.val, meta, batches.all_idxs))
        # print_perf()

    x_final = X.val
    dLd_x = L_grad(X.val, meta, batches.all_idxs)
    dMd_x = M_grad(X.val, meta)
    L_final = optimizing_loss(x_final, meta, batches.all_idxs)
    M_final = secondary_loss(x_final, meta)
    dLd_v = np.zeros(dLd_x.shape)
    dMd_v = np.zeros(dMd_x.shape)
    dLd_alphas = deque()
    dLd_betas = deque()
    dMd_alphas = deque()
    dMd_betas = deque()
    dLd_meta = L_meta_grad(X.val, meta, batches.all_idxs)
    dMd_meta = M_meta_grad(X.val, meta)
    print_perf()

    for i, alpha, beta, batch in iters[::-1]:
        # print_perf()
        dLd_v += dLd_x * alpha
        dMd_v += dMd_x * alpha
        X.sub(alpha * V.val)
        g = L_grad(X.val, meta, batch)
        dLd_alphas.appendleft(np.dot(dLd_x, V.val))
        dMd_alphas.appendleft(np.dot(dMd_x, V.val))
        V.add((1.0 - beta) * g)
        V.div(beta)
        dLd_betas.appendleft(np.dot(dLd_v, V.val + g))
        dMd_betas.appendleft(np.dot(dMd_v, V.val + g))
        dLd_x -= (1.0 - beta) * L_hvp(X.val, dLd_v, batch)
        dMd_x -= (1.0 - beta) * L_hvp(X.val, dMd_v, batch)
        dLd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dLd_v, batch)
        dMd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dMd_v, batch)
        dLd_v = dLd_v * beta
        dMd_v = dMd_v * beta

    dLd_alphas = np.array(dLd_alphas)
    dLd_betas = np.array(dLd_betas)

    # print "-"*80
    assert np.all(x_orig == X.val)
    return {'x_final': x_final,
            'learning_curve': learning_curve,
            'L_final': L_final,
            'M_final': M_final,
            'dLd_x': dLd_x,
            'dMd_x': dMd_x,
            'dLd_v': dLd_v,
            'dMd_v': dMd_v,
            'dLd_alphas': dLd_alphas,
            'dMd_alphas': dMd_alphas,
            'dLd_betas': dLd_betas,
            'dMd_betas': dMd_betas,
            'dLd_meta': dLd_meta,
            'dMd_meta': dMd_meta}


def sgd2_short(optimizing_loss, secondary_loss, batches, N_iter, x0, v0, alphas, betas, meta):
    """
    This version takes a secondary loss, and also returns gradients w.r.t. the data.
    :param optimizing_loss: The loss to be optimized by SGD.
    The first argument must be the parameters, the second must be the metaparameters,
    the third is data indicies.
    :param secondary_loss: Another loss we want to compute the gradient wrt.
    It takes parameters and metaparameters.
    :param batches: A list of slices into the data.
    :param N_iter: Number of iterations of SGD.
    :param x0: Starting parameter values.
    :param v0: Starting velocity.  Should probably be zero.
    :param alphas: Stepsize schedule.
    :param betas: Drag schedule.
    :param meta: A second parameter of the loss function that doesn't get optimized here.
    :return:
    a dict containing:
    Gradients wrt x0, v0, alphas, beta, and meta.
    """

    # TODO: Warp alpha and beta to map from real-valued domains (exp and logistic?)
    def print_perf():
        pass
        if (i + 1) % iter_per_epoch == 0:
            print "End of epoch {0}: loss is {1}".format(i / iter_per_epoch,
                                                         optimizing_loss(X.val, meta, batches.all_idxs))

    X, V = ExactRep(x0), ExactRep(v0)
    x_orig = X.val
    iter_per_epoch = len(batches)
    num_epochs = N_iter / len(batches) + 1
    iters = zip(range(N_iter), alphas, betas, batches * num_epochs)
    L_grad = grad(optimizing_loss)  # Gradient wrt parameters.
    M_grad = grad(secondary_loss)  # Gradient wrt parameters.
    L_meta_grad = grad(optimizing_loss, 1)  # Gradient wrt metaparameters.
    M_meta_grad = grad(secondary_loss, 1)  # Gradient wrt metaparameters.
    L_hvp = grad(lambda x, d, idxs:
                 np.dot(L_grad(x, meta, idxs), d))  # Hessian-vector product.
    L_hvp_meta = grad(lambda x, meta, d, idxs:
                      np.dot(L_grad(x, meta, idxs), d), 1)  # Returns a size(meta) output.

    learning_curve = [optimizing_loss(X.val, meta, batches.all_idxs)]
    for i, alpha, beta, batch in iters:
        V.mul(beta)
        g = L_grad(X.val, meta, batch)
        V.sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        learning_curve.append(optimizing_loss(X.val, meta, batches.all_idxs))
        # print_perf()

    x_final = X.val
    dLd_x = L_grad(X.val, meta, batches.all_idxs)
    dMd_x = M_grad(X.val, meta)
    L_final = optimizing_loss(x_final, meta, batches.all_idxs)
    M_final = secondary_loss(x_final, meta)
    dLd_v = np.zeros(dLd_x.shape)
    dMd_v = np.zeros(dMd_x.shape)
    dLd_alphas = deque()
    dLd_betas = deque()
    dMd_alphas = deque()
    dMd_betas = deque()
    dLd_meta = L_meta_grad(X.val, meta, batches.all_idxs)
    dMd_meta = M_meta_grad(X.val, meta)
    print_perf()
    sigma = np.linspace(0.01, 0.99, N_iter)

    for i, alpha, beta, batch in iters[::-1]:
        # print_perf()
        dLd_v += dLd_x * alpha
        dMd_v += dMd_x * alpha
        X.intrep = x_final * RADIX_SCALE
        X.add((1/sigma[i] -1)*x_orig)
        X.mul(sigma[i])
        # g = L_grad(X.val, meta, batch)
        # dLd_alphas.appendleft(np.dot(dLd_x, V.val))
        # dMd_alphas.appendleft(np.dot(dMd_x, V.val))
        # V.add((1.0 - beta) * g)
        # V.div(beta)
        # dLd_betas.appendleft(np.dot(dLd_v, V.val + g))
        # dMd_betas.appendleft(np.dot(dMd_v, V.val + g))
        dLd_x -= (1.0 - beta) * L_hvp(X.val, dLd_v, batch)
        dMd_x -= (1.0 - beta) * L_hvp(X.val, dMd_v, batch)
        dLd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dLd_v, batch)
        dMd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dMd_v, batch)
        dLd_v = dLd_v * beta
        dMd_v = dMd_v * beta

    dLd_alphas = np.array(dLd_alphas)
    dLd_betas = np.array(dLd_betas)

    # print "-"*80
    # assert np.all(x_orig == X.val)
    return {'x_final': x_final,
            'learning_curve': learning_curve,
            'L_final': L_final,
            'M_final': M_final,
            'dLd_x': dLd_x,
            'dMd_x': dMd_x,
            'dLd_v': dLd_v,
            'dMd_v': dMd_v,
            'dLd_alphas': dLd_alphas,
            'dMd_alphas': dMd_alphas,
            'dLd_betas': dLd_betas,
            'dMd_betas': dMd_betas,
            'dLd_meta': dLd_meta,
            'dMd_meta': dMd_meta}



def sgd3(optimizing_loss, secondary_loss, x0, v0, alphas, betas, meta, callback=None):
    """Same as sgd2 but simplifies things by not bothering with grads of
    optimizing loss (can always just pass that in as the secondary loss)"""
    X, V = ExactRep(x0), ExactRep(v0)
    L_grad = grad(optimizing_loss)  # Gradient wrt parameters.
    grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
    L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
    L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
    iters = zip(range(len(alphas)), alphas, betas)
    for i, alpha, beta in iters:
        if callback: callback(X.val, i)
        g = L_grad(X.val, meta, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
    x_final = X.val
    M_grad = grad(secondary_loss, 0)  # Gradient wrt parameters.
    M_meta_grad = grad(secondary_loss, 1)  # Gradient wrt metaparameters.
    dMd_x = M_grad(X.val, meta)
    dMd_v = np.zeros(dMd_x.shape)
    dMd_alphas = deque()
    dMd_betas = deque()
    dMd_meta = M_meta_grad(X.val, meta)
    for i, alpha, beta in iters[::-1]:
        dMd_alphas.appendleft(np.dot(dMd_x, V.val))
        X.sub(alpha * V.val)
        g = L_grad(X.val, meta, i)
        V.add((1.0 - beta) * g).div(beta)
        dMd_v += dMd_x * alpha
        dMd_betas.appendleft(np.dot(dMd_v, V.val + g))
        dMd_x -= (1.0 - beta) * L_hvp_x(X.val, meta, dMd_v, i)
        dMd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dMd_v, i)
        dMd_v *= beta

    assert np.all(ExactRep(x0).val == X.val)
    return {'x_final': x_final,
            'dMd_x': dMd_x,
            'dMd_v': dMd_v,
            'dMd_alphas': dMd_alphas,
            'dMd_betas': dMd_betas,
            'dMd_meta': dMd_meta}

def sgd4(L_grad, hypers, callback=None, forward_pass_only=True):
    x0, alphas, betas, meta = hypers
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, betas)
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_alphas, d_betas = np.zeros(len(alphas)), np.zeros(len(betas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i, alpha, beta in iters[::-1]:
            d_alphas[i] = np.dot(d_x, V.val)
            X.sub(alpha * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - beta) * g).div(beta)  # Reverse momentum update
            d_v += d_x * alpha
            d_betas[i] = np.dot(d_v, V.val + g)
            d_x -= (1.0 - beta) * L_hvp_x(X.val, meta, d_v, i)
            d_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, d_v, i)
            d_v *= beta
        assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_betas, d_meta

    return x_final, [None, hypergrad]

sgd4 = Differentiable(sgd4, partial(sgd4, forward_pass_only=False))



# random projection can be done by choosing two vectors randomly and applying Gram-Schmidt
def sgd4_random_coordinates(L_grad, hypers, callback=None, forward_pass_only=True):
    x0, alphas, betas, meta = hypers
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, betas)
    
    #list of X values
    length = len(X.val)
    coord1 = np.random.randint(0, length)
    coord2 = np.random.randint(0, length)
    X_coord1 = [X.val[coord1]]
    X_coord2 = [X.val[coord2]]
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        
        #add new coordinates
        X_coord1.append(X.val[coord1])
        X_coord2.append(X.val[coord2])
    x_final = X.val
    
    # plot
    plt.plot(X_coord1, X_coord2, marker='o', ms=3.)
    plt.plot(X_coord1[0], X_coord2[0], marker='o')
    plt.plot(X_coord1[len(X_coord1)-1], X_coord2[len(X_coord1)-1], marker='o', ms=10.)
    plt.show()

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_alphas, d_betas = np.zeros(len(alphas)), np.zeros(len(betas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i, alpha, beta in iters[::-1]:
            d_alphas[i] = np.dot(d_x, V.val)
            X.sub(alpha * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - beta) * g).div(beta)  # Reverse momentum update
            d_v += d_x * alpha
            d_betas[i] = np.dot(d_v, V.val + g)
            d_x -= (1.0 - beta) * L_hvp_x(X.val, meta, d_v, i)
            d_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, d_v, i)
            d_v *= beta


        # Michael
        #assert np.all(ExactRep(x0).val == X.val)
        print("||ExactRep(x0).val-X.val|| = ")
        print(np.linalg.norm(ExactRep(x0).val-X.val))
        
        
        return d_x, d_alphas, d_betas, d_meta

    return x_final, [None, hypergrad]

sgd4_random_coordinates = Differentiable(sgd4_random_coordinates, partial(sgd4_random_coordinates, forward_pass_only=False))








from sklearn.decomposition import PCA #, TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
#PCA or SVD???

# want to do PCA on the data, not SVD, because we're interested in how the path changes,
# not its shift away from 0 (uncentred SVD may give too much weight to constant shifts of the path)
"""
L_grad = grad(indexed_loss_fun)
hypers = kylist(W0, alphas, betas, L2_reg)
forward_pass_only=True
"""
def sgd4_PCA(L_grad, hypers, callback=None, forward_pass_only=True):
    x0, alphas, betas, meta = hypers
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, betas)
    
    #list of X values
    Xs = [X.val]
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        
        #add X value
        Xs.append(X.val)
    x_final = X.val
    
    X_matrix = np.array(Xs).T
    pca = PCA(n_components=3, copy=True) #, svd_solver='randomized') #????
    pca.fit(X_matrix)
    # plot
    plt.plot(pca.components_[0], pca.components_[1], marker='o', ms=3.)
    plt.plot(pca.components_[0][0], pca.components_[1][0], marker='o')
    plt.plot(pca.components_[0][len(pca.components_[0])-1], pca.components_[1][len(pca.components_[1])-1], marker='o', ms=10.)
    plt.xlabel('1st: '+str(pca.explained_variance_ratio_[0]))
    plt.ylabel('2nd: '+str(pca.explained_variance_ratio_[1])) 
    plt.title('First 2 prinicipal components')
    plt.show()
    print pca.explained_variance_ratio_

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs=pca.components_[0], ys=pca.components_[1], zs=pca.components_[2], label='first 3 principal components')
    ax.plot(xs=[pca.components_[0][0]], ys=[pca.components_[1][0]], zs=[pca.components_[2][0]], marker='o', label='initial')
    ax.plot(xs=[pca.components_[0][len(pca.components_[0])-1]], ys=[pca.components_[1][len(pca.components_[1])-1]], 
                zs=[pca.components_[2][len(pca.components_[1])-1]], marker='o', ms=10., label='final')
    ax.set_xlabel('1st: '+str(pca.explained_variance_ratio_[0]))
    ax.set_ylabel('2nd: '+str(pca.explained_variance_ratio_[1]))
    ax.set_zlabel('3rd: '+str(pca.explained_variance_ratio_[2]))
    #plt.legend()
    plt.title('First 3 principal components')
    plt.show()


    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_alphas, d_betas = np.zeros(len(alphas)), np.zeros(len(betas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i, alpha, beta in iters[::-1]:
            d_alphas[i] = np.dot(d_x, V.val)
            X.sub(alpha * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - beta) * g).div(beta)  # Reverse momentum update
            d_v += d_x * alpha
            d_betas[i] = np.dot(d_v, V.val + g)
            d_x -= (1.0 - beta) * L_hvp_x(X.val, meta, d_v, i)
            d_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, d_v, i)
            d_v *= beta
        assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_betas, d_meta

    return x_final, [None, hypergrad]

sgd4 = Differentiable(sgd4, partial(sgd4, forward_pass_only=False))




# Michael: plot the distances between current and final weights
def sgd4_distance_plot(L_grad, hypers, callback=None, forward_pass_only=True):
    x0, alphas, betas, meta = hypers
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, betas)
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
    x_final = X.val

    if forward_pass_only:
        return x_final

    #check distances from final point
    dist_from_x_final = [0.]


    def hypergrad(outgrad):
        d_x = outgrad
        d_alphas, d_betas = np.zeros(len(alphas)), np.zeros(len(betas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i, alpha, beta in iters[::-1]:
            d_alphas[i] = np.dot(d_x, V.val)
            X.sub(alpha * V.val)  # Reverse position update

            # add distance
            dist_from_x_final.append(np.linalg.norm(X.val-x_final))
            
            

            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - beta) * g).div(beta)  # Reverse momentum update
            d_v += d_x * alpha
            d_betas[i] = np.dot(d_v, V.val + g)
            d_x -= (1.0 - beta) * L_hvp_x(X.val, meta, d_v, i)
            d_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, d_v, i)
            d_v *= beta
        assert np.all(ExactRep(x0).val == X.val)
        
        #plot distances
        #print(dist_from_x_final)
        plt.plot(range(len(dist_from_x_final),0,-1), dist_from_x_final, 'b', label="||x-x_T||")
        plt.legend()
        plt.show()
                
        
        return d_x, d_alphas, d_betas, d_meta

    return x_final, [None, hypergrad]

#Michael: ???
sgd4 = Differentiable(sgd4, partial(sgd4, forward_pass_only=False))


#Michael: plot grad/velocity/step norms
def sgd4_mad(L_grad, hypers, callback=None, forward_pass_only=True):

    x0, alphas, gammas, meta = hypers
    N_safe_sampling = len(alphas)
    x_init = np.copy(x0)
    x_current = np.copy(x0)
    global  v_current
    v_current = np.zeros(x0.size)
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, gammas)
    
    T = len(alphas)
    ts = range(0,T)
    gnorms = [0. for t in ts]
    gdotg0norms = [0. for t in ts]
    Vnorms = [0. for t in ts]
    VdotV0norms = [0. for t in ts]
    Vagreement = [0. for t in range(1,T)]
    gagreement = [0. for t in range(1,T)]
    t=0
    g0 = 0.
    g = 0.
    aV0 = 0.

    for i, alpha, gamma in iters:
        vprev = V.val
        gprev = g
        
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(gamma).sub((1.0 - gamma) * g)
        X.add(alpha * V.val)
        if i==0.:
            g0 = g
            aV0 = alpha*V.val
        gnorms[t] = np.linalg.norm(g)
        gdotg0norms[t] = np.dot(g,g0)/np.linalg.norm(g0)
        Vnorms[t] = np.linalg.norm(alpha*V.val)
        VdotV0norms[t] = np.dot(alpha*V.val,aV0)/np.linalg.norm(aV0)
        
        
        if t>=1:
            Vagreement[t-1] = np.dot(V.val,vprev)/(np.linalg.norm(V.val)*np.linalg.norm(vprev))
            gagreement[t-1] = np.dot(g,gprev)/(np.linalg.norm(g)*np.linalg.norm(gprev))
        t=t+1
    plt.plot(ts, gnorms, 'b', label="||g||")
    plt.plot(ts, gdotg0norms, 'b--', label="<g,g0>/||g0||")
    plt.plot(ts, [gnorms[0] for t in ts], 'b:', label="||g_0||")
    plt.plot(ts, Vnorms, 'r', label="||alpha * v||")
    plt.plot(ts, VdotV0norms, 'r--', label="<alpha*v,alpha_0*v_0>/||alpha_0*v_0||")
    plt.plot(ts, [Vnorms[0] for t in ts], 'r:', label="||alpha_0*v_0||")
    plt.legend()
    plt.show()
    plt.plot(ts, gnorms, 'b', label="||g||")
    plt.plot(ts, gdotg0norms, 'b--', label="<g,g0>/||g0||")
    plt.plot(ts, [gnorms[0] for t in ts], 'b:', label="||g_0||")
    plt.legend()
    plt.show()
    plt.plot(ts, Vnorms, 'r', label="||alpha * v||")
    plt.plot(ts, VdotV0norms, 'r--', label="<alpha*v,alpha_0*v_0>/||alpha_0*v_0||")
    plt.plot(ts, [Vnorms[0] for t in ts], 'r:', label="||alpha_0*v_0||")
    plt.legend()
    plt.show()
    plt.plot(range(1,T), gagreement, 'b', label="<g,gprev>/(||g||||gprev||)")
    plt.legend()
    plt.show()
    plt.plot(range(1,T), Vagreement, 'r', label="<v,vprev>/(||v||||vprev||)")
    plt.legend()
    plt.show()
    
    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        global v_current
        v=v_current
        d_alphas, d_gammas = np.zeros(len(alphas)), np.zeros(len(gammas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x    = grad(grad_proj, 0) # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1) # Returns a size(gamma) output.
        
        beta = np.linspace(0.001, 0.999, N_safe_sampling) #evenly spaced, Michael
        
        for i, alpha, gamma in iters[::-1]:
            
            # Here is the averaging sequence, Michael
            x = (1 - beta[i])*x_init + beta[i]*x_final
            
            x_previous = (1 - beta[i-1])*x_init + beta[i-1]*x_final
            v = np.subtract(x,x_previous)/alpha #recover velocity
            d_alphas[i] = np.dot(d_x, v)
            g = L_grad(x, meta, i)         # Evaluate gradient
            # v = (v+(1.0 - gamma)*g)/gamma
            d_v += d_x * alpha
            d_gammas[i] = np.dot(d_v, v + g)
            d_x    -= (1.0 - gamma) * L_hvp_x(x, meta, d_v, i) #DrMad paper forgot to mention this line, Michael
            d_meta -= (1.0 - gamma) * L_hvp_meta(x, meta, d_v, i)
            d_v    *= gamma #DrMad paper forgot to mention this line, Michael
        # assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_gammas, d_meta
        # TODO: this returns gradients with respect to initial parameters, learning rate and momentum decay???

    return x_final, [None, hypergrad]

sgd4_mad = Differentiable(sgd4_mad, partial(sgd4_mad, forward_pass_only=False))






# Michael: 
def sgd4_mad_PCA(L_grad, hypers, callback=None, forward_pass_only=True):

    x0, alphas, gammas, meta = hypers
    N_safe_sampling = len(alphas)
    x_init = np.copy(x0)
    x_current = np.copy(x0)
    global  v_current
    v_current = np.zeros(x0.size)
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, gammas)
    
    #list of X values
    Xs = [X.val]
    for i, alpha, gamma in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(gamma).sub((1.0 - gamma) * g)
        X.add(alpha * V.val)
        
        #add X value
        Xs.append(X.val)
    
    X_matrix = np.array(Xs).T
    pca = PCA(n_components=3, copy=True) #, svd_solver='randomized') #????
    pca.fit(X_matrix)
    
    # plot
    print pca.explained_variance_ratio_
    plt.plot(pca.components_[0], pca.components_[1], marker='o', ms=7., color='purple', markevery=40)
    plt.plot(pca.components_[0], pca.components_[1], marker='o', ms=3., color='blue')
    plt.plot(pca.components_[0][0], pca.components_[1][0], marker='o', color='green')
    plt.plot(pca.components_[0][len(pca.components_[0])-1], pca.components_[1][len(pca.components_[1])-1], marker='o', ms=10., color='red')
    plt.xlabel('1st: '+str(pca.explained_variance_ratio_[0]))
    plt.ylabel('2nd: '+str(pca.explained_variance_ratio_[1])) 
    plt.title('First 2 prinicipal components')
    plt.show()
    print pca.explained_variance_ratio_
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs=pca.components_[0], ys=pca.components_[1], zs=pca.components_[2], label='first 3 principal components', marker='o', ms=7., color='purple', markevery=40)
    ax.plot(xs=pca.components_[0], ys=pca.components_[1], zs=pca.components_[2], label='first 3 principal components', marker='o', ms=3., color='blue')
    ax.plot(xs=[pca.components_[0][0]], ys=[pca.components_[1][0]], zs=[pca.components_[2][0]], marker='o', color='green', label='initial')
    ax.plot(xs=[pca.components_[0][len(pca.components_[0])-1]], ys=[pca.components_[1][len(pca.components_[1])-1]], 
                zs=[pca.components_[2][len(pca.components_[1])-1]], marker='o', ms=10., color='red', label='final')
    ax.set_xlabel('1st: '+str(pca.explained_variance_ratio_[0]))
    ax.set_ylabel('2nd: '+str(pca.explained_variance_ratio_[1]))
    ax.set_zlabel('3rd: '+str(pca.explained_variance_ratio_[2]))
    #plt.legend()
    plt.title('First 3 principal components')
    plt.show()



    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        global v_current
        v=v_current
        d_alphas, d_gammas = np.zeros(len(alphas)), np.zeros(len(gammas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x    = grad(grad_proj, 0) # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1) # Returns a size(gamma) output.
        
        beta = np.linspace(0.001, 0.999, N_safe_sampling) #evenly spaced, Michael
        
        for i, alpha, gamma in iters[::-1]:
            
            # Here is the averaging sequence, Michael
            x = (1 - beta[i])*x_init + beta[i]*x_final
            
            x_previous = (1 - beta[i-1])*x_init + beta[i-1]*x_final
            v = np.subtract(x,x_previous)/alpha #recover velocity
            d_alphas[i] = np.dot(d_x, v)
            g = L_grad(x, meta, i)         # Evaluate gradient
            # v = (v+(1.0 - gamma)*g)/gamma
            d_v += d_x * alpha
            d_gammas[i] = np.dot(d_v, v + g)
            d_x    -= (1.0 - gamma) * L_hvp_x(x, meta, d_v, i) #DrMad paper forgot to mention this line, Michael
            d_meta -= (1.0 - gamma) * L_hvp_meta(x, meta, d_v, i)
            d_v    *= gamma #DrMad paper forgot to mention this line, Michael
        # assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_gammas, d_meta

    return x_final, [None, hypergrad]

sgd4_mad_PCA = Differentiable(sgd4_mad_PCA, partial(sgd4_mad_PCA, forward_pass_only=False))










#TODO: need to find a way to return exact hypergradient
#Or, have exact hypergradient as a global variable to be modified inside the function
"""
def sgd4_mad_with_exact_old(L_grad, hypers, callback=None, forward_pass_only=True):
    x0, alphas, gammas, meta = hypers
    N_safe_sampling = len(alphas)
    x_init = np.copy(x0)
    #x_current = np.copy(x0)
    global  v_current
    v_current = np.zeros(x0.size)
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, gammas)
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad_exact(outgrad):
        d_x = outgrad
        d_alphas, d_gammas = np.zeros(len(alphas)), np.zeros(len(gammas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i, alpha, gamma in iters[::-1]:
            d_alphas[i] = np.dot(d_x, V.val)
            X.sub(alpha * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - gamma) * g).div(gamma)  # Reverse momentum update
            d_v += d_x * alpha
            d_gammas[i] = np.dot(d_v, V.val + g)
            d_x -= (1.0 - gamma) * L_hvp_x(X.val, meta, d_v, i)
            d_meta -= (1.0 - gamma) * L_hvp_meta(X.val, meta, d_v, i)
            d_v *= gamma
        assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_gammas, d_meta
        
    def hypergrad_mad(outgrad):
        d_x = outgrad
        global v_current
        v=v_current
        d_alphas, d_gammas = np.zeros(len(alphas)), np.zeros(len(gammas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x    = grad(grad_proj, 0) # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1) # Returns a size(gamma) output.
        beta = np.linspace(0.001, 0.999, N_safe_sampling) #evenly spaced, Michael
        for i, alpha, gamma in iters[::-1]:
            # Here is the averaging sequence, Michael
            x = (1 - beta[i])*x_init + beta[i]*x_final
            x_previous = (1 - beta[i-1])*x_init + beta[i-1]*x_final
            v = np.subtract(x,x_previous)/alpha #recover velocity
            d_alphas[i] = np.dot(d_x, v)
            g = L_grad(x, meta, i)         # Evaluate gradient
            # v = (v+(1.0 - gamma)*g)/gamma
            d_v += d_x * alpha
            d_gammas[i] = np.dot(d_v, v + g)
            d_x    -= (1.0 - gamma) * L_hvp_x(x, meta, d_v, i) #DrMad paper forgot to mention this line, Michael
            d_meta -= (1.0 - gamma) * L_hvp_meta(x, meta, d_v, i)
            d_v    *= gamma #DrMad paper forgot to mention this line, Michael
        # assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_gammas, d_meta

    return x_final, [None, [hypergrad_mad, hypergrad_exact]] #TODO: Does this make sense?
"""
    







def sgd4_mad_with_exact(L_grad, hypers, exact_hypergrad, callback=None, forward_pass_only=True):
    x0, alphas, gammas, meta = hypers
    #print("Elementary SGD")
    #print("x0", len(x0))
    #print("alphas", len(alphas))
    #print("gammas", len(gammas))
    #print("meta", len(meta))
    #x0, alphas, gammas, meta, x0_dummy, alphas_dummy, gammas_dummy, meta_dummy = hypers
    N_safe_sampling = len(alphas)
    x_init = np.copy(x0)
    #x_current = np.copy(x0)
    global  v_current
    v_current = np.zeros(x0.size)
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, gammas)
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad, exact_hypergrad=exact_hypergrad):
        #DrMAD
        d_x = outgrad
        global v_current
        v=v_current
        d_alphas, d_gammas = np.zeros(len(alphas)), np.zeros(len(gammas))
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x    = grad(grad_proj, 0) # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1) # Returns a size(gamma) output.
        beta = np.linspace(0.001, 0.999, N_safe_sampling) #evenly spaced, Michael
        for i, alpha, gamma in iters[::-1]:
            # Here is the averaging sequence, Michael
            x = (1 - beta[i])*x_init + beta[i]*x_final
            x_previous = (1 - beta[i-1])*x_init + beta[i-1]*x_final
            v = np.subtract(x,x_previous)/alpha #recover velocity
            d_alphas[i] = np.dot(d_x, v)
            g = L_grad(x, meta, i)         # Evaluate gradient
            # v = (v+(1.0 - gamma)*g)/gamma
            d_v += d_x * alpha
            d_gammas[i] = np.dot(d_v, v + g)
            d_x    -= (1.0 - gamma) * L_hvp_x(x, meta, d_v, i) #DrMad paper forgot to mention this line, Michael
            d_meta -= (1.0 - gamma) * L_hvp_meta(x, meta, d_v, i)
            d_v    *= gamma #DrMad paper forgot to mention this line, Michael

        #Exact
        d_x_exact = outgrad
        d_alphas_exact, d_gammas_exact = np.zeros(len(alphas)), np.zeros(len(gammas))
        d_v, d_meta_exact = np.zeros(d_x_exact.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i, alpha, gamma in iters[::-1]:
            d_alphas_exact[i] = np.dot(d_x_exact, V.val)
            X.sub(alpha * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - gamma) * g).div(gamma)  # Reverse momentum update
            d_v += d_x * alpha
            d_gammas_exact[i] = np.dot(d_v, V.val + g)
            d_x_exact -= (1.0 - gamma) * L_hvp_x(X.val, meta, d_v, i)
            d_meta_exact -= (1.0 - gamma) * L_hvp_meta(X.val, meta, d_v, i)
            d_v *= gamma
        assert np.all(ExactRep(x0).val == X.val)
        # 'tuple' object does not support item assignment, so use list instead
        #print(len(exact_hypergrad['log_L2_reg'])) #2
        #print(len(d_x_exact)) #7850
        # TODO: get rid of ugly hardcoding indices
        exact_hypergrad['log_alphas'] = d_alphas
        exact_hypergrad['invlogit_betas'] = d_gammas
        
        """print("log_param_scale")
        print(d_x_exact)
        print(exact_hypergrad['log_param_scale'])
        print("L2_reg")
        print(d_meta_exact)
        print(exact_hypergrad['log_L2_reg'])"""
        
        exact_hypergrad['log_param_scale'] = d_x_exact
        exact_hypergrad['log_L2_reg'] = d_meta_exact
        #exact_hypergrad['log_param_scale'][0] = d_x_exact[0]
        #exact_hypergrad['log_param_scale'][-1] = d_x_exact[-1]
        #exact_hypergrad['log_L2_reg'][0] = d_meta_exact[0]
        #exact_hypergrad['log_L2_reg'][-1] = d_meta_exact[-1]
        
        # TODO: probably has to have the same dimensions as input
        #print("Hypergrad")
        #print("d_x", len(d_x), "d_x_exact", len(d_x_exact))
        #print("d_alphas", len(d_alphas), "d_alphas_exact", len(d_alphas_exact))
        #print("d_gammas", len(d_gammas), "d_gammas_exact", len(d_gammas_exact))
        #print("d_meta", len(d_meta), "d_meta_exact", len(d_meta_exact))
        
        
        return d_x, d_alphas, d_gammas, d_meta

    return x_final, [None, hypergrad]


sgd4_mad_with_exact = Differentiable(sgd4_mad_with_exact, partial(sgd4_mad_with_exact, forward_pass_only=False))










def sgd_meta_only_mad_with_exact(L_grad, meta, x0, exact_hypergrad, alpha, gamma, N_iters,
                callback=None, forward_pass_only=True):
    #the tricky part is the denominator, still investigating the effect
    N_safe_sampling = N_iters/10
    x_init = x0
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))

    for i in range(N_iters):
        g = L_grad(X.val, meta, i, record_results=True)
        if callback: callback(X.val, V.val, g, i)
        V.mul(gamma).sub((1.0 - gamma) * g)
        X.add(alpha * V.val)
    x_final = X.val
    if forward_pass_only:
        return x_final

    def hypergrad(outgrad, exact_hypergrad=exact_hypergrad):
        #DrMAD
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        beta = np.linspace(0.001,0.999,N_safe_sampling)
        for i in range(N_safe_sampling)[::-1]:
            x_current = (1-beta[i])*x_init + beta[i]*x_final
            d_v += d_x * alpha
            d_x -= (1.0 - gamma) * L_hvp_x(x_current, meta, d_v, i)
            d_meta -= (1.0 - gamma) * L_hvp_meta(x_current, meta, d_v, i)
            d_v *= gamma
        # assert np.all(ExactRep(x0).val == X.val)

        #Exact
        #print exact_hypergrad
        d_x = outgrad
        d_v_exact, d_meta_exact = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i in range(N_iters)[::-1]:
            X.sub(alpha * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - gamma) * g).div(gamma)  # Reverse momentum update
            d_v_exact += d_x * alpha
            d_x -= (1.0 - gamma) * L_hvp_x(X.val, meta, d_v_exact, i)
            d_meta_exact -= (1.0 - gamma) * L_hvp_meta(X.val, meta, d_v_exact, i)
            d_v_exact *= gamma
        #print np.linalg.norm(ExactRep(x0).val-X.val)
        assert np.all(ExactRep(x0).val == X.val)
        exact_hypergrad[0] = d_meta_exact
        #print exact_hypergrad
        
        return d_meta

    return x_final, [None, hypergrad]

sgd_meta_only_mad_with_exact = Differentiable(sgd_meta_only_mad_with_exact,
                               partial(sgd_meta_only_mad_with_exact, forward_pass_only=False))

















# Michael: plot distance from final
def sgd_meta_only(L_grad, meta, x0, alpha, beta, N_iters,
                  callback=None, forward_pass_only=True, meta_iteration=0):
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    for i in range(N_iters):
        g = L_grad(X.val, meta, i, record_results=True)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
    x_final = X.val
    if forward_pass_only:
        return x_final


    #check distances from final point
    #dist_from_x_final = [0.]

    hypergrad_norms = []


    def hypergrad(outgrad):
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i in range(N_iters)[::-1]:
            X.sub(alpha * V.val)  # Reverse position update
            

            # add distance
            #dist_from_x_final.append(np.linalg.norm(X.val-x_final))            
            
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - beta) * g).div(beta)  # Reverse momentum update
            #d_v += d_x * alpha
            d_v = beta*d_v + d_x * alpha #correction?
            d_x -= (1.0 - beta) * L_hvp_x(X.val, meta, d_v, i)
            d_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, d_v, i)
            #d_v *= beta
            
            hypergrad_norms.append(np.linalg.norm(d_meta))
            
            if i%1000 == 0:
                print(i, hypergrad_norms[-1])
                plt.plot(range(N_iters-1, N_iters-len(hypergrad_norms)-1,-1), hypergrad_norms)
                plt.show()
        assert np.all(ExactRep(x0).val == X.val)
        
        
        #plot distances
        #print(dist_from_x_final)
        #plt.plot(range(len(dist_from_x_final),0,-1), dist_from_x_final, 'b', label="||x-x_T||")
        #plt.legend()
        #plt.show()
        
        plt.plot(range(N_iters)[::-1], hypergrad_norms)
        plt.savefig('exact_norms200000_'+str(meta_iteration)+'_corrected.png')
        plt.show()
        return d_meta
        
    return x_final, [None, hypergrad]


sgd_meta_only = Differentiable(sgd_meta_only,
                               partial(sgd_meta_only, forward_pass_only=False))

def sgd_meta_only_mad(L_grad, meta, x0, alpha, gamma, N_iters,
                callback=None, forward_pass_only=True):
    #the tricky part is the denominator, still investigating the effect
    #N_safe_sampling = N_iters/10
    #N_safe_sampling = N_iters #leads to exploding hypergradient
    #N_safe_sampling = N_iters/100
    N_safe_sampling = min(N_iters, 200)
    x_init = x0
    x_current = x0
    v_current = np.zeros(x0.size)
    
    """T = N_iters
    ts = range(0,T)
    gnorms = [0. for t in ts]
    gdotg0norms = [0. for t in ts]
    Vnorms = [0. for t in ts]
    VdotV0norms = [0. for t in ts]
    t=0
    g0 = 0.
    aV0 = 0."""
    """X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    for i in range(N_iters):
        g = L_grad(X.val, meta, i, record_results=True)
        if callback: callback(X.val, V.val, g, i)
        V.mul(gamma).sub((1.0 - gamma) * g)
        X.add(alpha * V.val)
    x_final = X.val    """
    
    dist_from_x0 = [] #will be normalized by iterate number
    for i in range(N_iters):
        g = L_grad(x_current, meta, i, record_results=True)
        if callback: callback(x_current, v_current, g, i)
        v_current = v_current*gamma -(1.0 - gamma)*g
        x_current = x_current +alpha*v_current
        if i > 3 and i < min(20,N_iters):
            dist_from_x0.append(np.linalg.norm(x_current-x0)/(i+1))
            #TODO: try max of these instead?
        
        #if i==0.:
        #    g0 = g
        #    aV0 = alpha*v_current
        #gnorms[t] = np.linalg.norm(g)
        #gdotg0norms[t] = np.dot(g,g0)/np.linalg.norm(g0)
        #Vnorms[t] = np.linalg.norm(alpha*v_current)
        #VdotV0norms[t] = np.dot(alpha*v_current,aV0)/np.linalg.norm(aV0)
        #t=t+1
    """plt.plot(ts, gnorms, 'b', label="||g||")
    plt.plot(ts, gdotg0norms, 'b--', label="<g,g0>/||g0||")
    plt.plot(ts, [gnorms[0] for t in ts], 'b:', label="||g_0||")
    plt.plot(ts, Vnorms, 'r', label="||alpha * v||")
    plt.plot(ts, VdotV0norms, 'r--', label="<alpha*v,alpha_0*v_0>/||alpha_0*v_0||")
    plt.plot(ts, [Vnorms[0] for t in ts], 'r:', label="||alpha_0*v_0||")
    plt.legend()
    plt.show()
    plt.plot(ts, gnorms, 'b', label="||g||")
    plt.plot(ts, gdotg0norms, 'b--', label="<g,g0>/||g0||")
    plt.plot(ts, [gnorms[0] for t in ts], 'b:', label="||g_0||")
    plt.legend()
    plt.show()
    plt.plot(ts, Vnorms, 'r', label="||alpha * v||")
    plt.plot(ts, VdotV0norms, 'r--', label="<alpha*v,alpha_0*v_0>/||alpha_0*v_0||")
    plt.plot(ts, [Vnorms[0] for t in ts], 'r:', label="||alpha_0*v_0||")
    plt.legend()
    plt.show()"""
    
    
    x_final = x_current
    
    if forward_pass_only:
        return x_final


    #2nd gets 2^-2, 3rd gets 2^-3, 4th gets 2^-4 so on; first gets what's left
    weights = [0.]+[2.**-(i+1) for i in range(1,len(dist_from_x0))]
    weights[0] = 1.-sum(weights[1:]) #there might be a closed form here
    beta = 0.
    for i in range(0,len(weights)):
        beta = beta + weights[i]*dist_from_x0[i]
    beta = 10.*beta/np.linalg.norm(x_final-x0)
    N_safe_sampling = int(np.ceil(1./beta)) #max(N,int(np.ceil(1./beta)))
    print("weights", weights)
    print("dist_from_x0", dist_from_x0)
    print("N_safe_sampling", N_safe_sampling)
    

    def hypergrad(outgrad):
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        beta = np.linspace(0.001,0.999,N_safe_sampling)
        # beta = np.linspace(0.,1.,N_safe_sampling) #TODO: remove point corresponding to final parameters
        for i in range(N_safe_sampling)[::-1]:
            x_current = (1-beta[i])*x_init + beta[i]*x_final
            #d_v += d_x * alpha
            d_v = gamma*d_v + d_x * alpha #correction?
            d_x -= (1.0 - gamma) * L_hvp_x(x_current, meta, d_v, i)
            d_meta -= (1.0 - gamma) * L_hvp_meta(x_current, meta, d_v, i)
            #d_v *= gamma
            #if i%50 == 0:
            #    print(i, np.linalg.norm(d_meta))
        # assert np.all(ExactRep(x0).val == X.val)
        return d_meta

    return x_final, [None, hypergrad]


sgd_meta_only_mad = Differentiable(sgd_meta_only_mad,
                               partial(sgd_meta_only_mad, forward_pass_only=False))











def sgd_meta_only_mad_random_coordinates(L_grad, meta, x0, alpha, gamma, N_iters,
                callback=None, forward_pass_only=True):
    #the tricky part is the denominator, still investigating the effect

    N_safe_sampling = N_iters/10
    x_init = x0
    x_current = x0
    v_current = np.zeros(x0.size)

    length = len(x0)
    coord1 = np.random.randint(0, length)
    coord2 = np.random.randint(0, length)
    X_coord1 = [x0[coord1]]
    X_coord2 = [x0[coord2]]

    for i in range(N_iters):
        g = L_grad(x_current, meta, i, record_results=True)
        if callback: callback(x_current, v_current, g, i)
        v_current = v_current*gamma -(1.0 - gamma)*g
        x_current = x_current +alpha*v_current
        X_coord1.append(x_current[coord1])
        X_coord2.append(x_current[coord2])

    # plot
    plt.plot(X_coord1, X_coord2, marker='o', ms=3.)
    plt.plot(X_coord1[0], X_coord2[0], marker='o')
    plt.plot(X_coord1[len(X_coord1)-1], X_coord2[len(X_coord1)-1], marker='o', ms=10.)
    plt.show()


    x_final = x_current
    
    if forward_pass_only:
        return x_final


    #check distances from final point
    dist_from_x_final = [0.]


    def hypergrad(outgrad):
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        beta = np.linspace(0.001,0.999,N_safe_sampling)
        for i in range(N_safe_sampling)[::-1]:
            x_current = (1-beta[i])*x_init + beta[i]*x_final
        
            # add distance
            dist_from_x_final.append(np.linalg.norm(x_current-x_final))
            
            d_v += d_x * alpha
            d_x -= (1.0 - gamma) * L_hvp_x(x_current, meta, d_v, i)
            d_meta -= (1.0 - gamma) * L_hvp_meta(x_current, meta, d_v, i)
            d_v *= gamma
        # assert np.all(ExactRep(x0).val == X.val)


        #plot distances
        #print(dist_from_x_final)
        plt.plot(range(len(dist_from_x_final),0,-1), dist_from_x_final, 'b', label="||x-x_T||")
        plt.legend()
        plt.show()

        return d_meta

    return x_final, [None, hypergrad]


sgd_meta_only_mad_random_coordinates = Differentiable(sgd_meta_only_mad_random_coordinates,
                               partial(sgd_meta_only_mad_random_coordinates, forward_pass_only=False))














def sgd_short_safe(L_grad, meta, x0, alpha, gamma, N_iters,
                callback=None, forward_pass_only=True):

    N_safe_sampling = 1000
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    x_init = X.val
    for i in range(N_iters):
        g = L_grad(X.val, meta, i, record_results=True)
        if callback: callback(X.val, V.val, g, i)
        V.mul(gamma).sub((1.0 - gamma) * g)
        X.add(alpha * V.val)
    x_final = X.val
    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        beta = np.linspace(0.001,0.999,N_safe_sampling)
        for i in range(N_safe_sampling)[::-1]:
            x_current = (1-beta[i])*x_init + beta[i]*x_final
            d_v += d_x * alpha
            d_x -= (1.0 - gamma) * L_hvp_x(x_current, meta, d_v, i)
            d_meta -= (1.0 - gamma) * L_hvp_meta(x_current, meta, d_v, i)
            d_v *= gamma
        # assert np.all(ExactRep(x0).val == X.val)
        return d_meta

    return x_final, [None, hypergrad]


sgd_short_safe = Differentiable(sgd_short_safe,
                               partial(sgd_short_safe, forward_pass_only=False))

def sgd_short_min(L_grad, meta, x0, alpha, gamma, N_iters,
                callback=None, forward_pass_only=True):

    N_safe_sampling = 1000
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    x_init = X.val
    for i in range(N_iters):
        g = L_grad(X.val, meta, i, record_results=True)
        if callback: callback(X.val, V.val, g, i)
        V.mul(gamma).sub((1.0 - gamma) * g)
        X.add(alpha * V.val)
    x_final = X.val
    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        beta = np.linspace(0.001,0.999,N_safe_sampling)
        for i in range(N_safe_sampling)[::-1]:
            x_current = (1-beta[i])*x_init + beta[i]*x_final
            d_v += d_x * alpha
            d_x -= (1.0 - gamma) * L_hvp_x(x_current, meta, d_v, i)
            d_meta -= (1.0 - gamma) * L_hvp_meta(x_current, meta, d_v, i)
            d_v *= gamma
        # assert np.all(ExactRep(x0).val == X.val)
        return d_meta

    return x_final, [None, hypergrad]


sgd_short_min = Differentiable(sgd_short_min,
                               partial(sgd_short_min, forward_pass_only=False))



def sum_args(fun, arglist):
    def sum_fun(*args):
        partial_fun = lambda i: fun(*(list(args) + [i]))
        return reduce(op.add, it.imap(partial_fun, arglist))

    return sum_fun


def sgd_meta_only_sub(L_grad_sub, meta, x0, alpha, beta, N_iters,
                      N_sub, callback=None, forward_pass_only=True):
    # Allows adding the gradient of multiple sub-batches (minibatch within a minibatch)
    # Signature of L_grad_sub is x, meta, i_batch, i_sub
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    L_grad = sum_args(L_grad_sub, range(N_sub))
    for i in range(N_iters):
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj_sub = lambda x, meta, d, i, i_sub: np.dot(L_grad_sub(x, meta, i, i_sub), d)
        L_hvp_x_sub = grad(grad_proj_sub, 0)  # Returns a size(x) output.
        L_hvp_meta_sub = grad(grad_proj_sub, 1)  # Returns a size(meta) output.
        L_hvp_x = sum_args(L_hvp_x_sub, range(N_sub))
        L_hvp_meta = sum_args(L_hvp_x_sub, range(N_sub))
        for i in range(N_iters)[::-1]:
            X.sub(alpha * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - beta) * g).div(beta)  # Reverse momentum update
            d_v += d_x * alpha
            d_x -= (1.0 - beta) * L_hvp_x(X.val, meta, d_v, i)
            d_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, d_v, i)
            d_v *= beta
        assert np.all(ExactRep(x0).val == X.val)
        return d_meta

    return x_final, [None, hypergrad]


sgd_meta_only_sub = Differentiable(sgd_meta_only_sub,
                               partial(sgd_meta_only_sub, forward_pass_only=False))


def sgd_parsed(L_grad, hypers, parser, callback=None, forward_pass_only=True):
    """This version has alphas and betas be TxN_weight_types matrices.
       parser is a dict containing the indices for the different types of weights."""
    x0, alphas, betas, meta = hypers
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, betas)
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_beta_vect = fill_parser(parser, beta)
        V.mul(cur_beta_vect).sub((1.0 - cur_beta_vect) * g)
        X.add(cur_alpha_vect * V.val)
    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_alphas, d_betas = np.zeros(alphas.shape), np.zeros(betas.shape)
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        for i, alpha, beta in iters[::-1]:

            # build alpha and beta vector
            cur_alpha_vect = fill_parser(parser, alpha)
            cur_beta_vect = fill_parser(parser, beta)
            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_alphas[i, j] = np.dot(d_x[ixs], V.val[ixs])

            X.sub(cur_alpha_vect * V.val)  # Reverse position update
            g = L_grad(X.val, meta, i)  # Evaluate gradient
            V.add((1.0 - cur_beta_vect) * g).div(cur_beta_vect)  # Reverse momentum update

            d_v += d_x * cur_alpha_vect

            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_betas[i, j] = np.dot(d_v[ixs], V.val[ixs] + g[ixs])

            d_x -= L_hvp_x(X.val, meta, (1.0 - cur_beta_vect) * d_v, i)
            d_meta -= L_hvp_meta(X.val, meta, (1.0 - cur_beta_vect) * d_v, i)
            d_v *= cur_beta_vect
        assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_betas, d_meta

    return x_final, [None, hypergrad]


sgd_parsed = Differentiable(sgd_parsed,
                            partial(sgd_parsed, forward_pass_only=False))


def sgd_parsed_mad(L_grad, hypers, parser, callback=None, forward_pass_only=True):
    """This version has alphas and betas be TxN_weight_types matrices.
       parser is a dict containing the indices for the different types of weights."""
    x0, alphas, gammas, meta = hypers
    N_safe_sampling = len(alphas)
    x_init = np.copy(x0)
    x_current = np.copy(x0)
    global v_current
    v_current = np.zeros(x0.size)
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, gammas)
    for i, alpha, gamma in iters:
        g = L_grad(X.val, meta, i)
        if callback: callback(X.val, V.val, g, i)
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect  = fill_parser(parser, gamma)
        V.mul(cur_gamma_vect).sub((1.0 - cur_gamma_vect) * g)
        X.add(cur_alpha_vect * V.val)
    x_final = X.val

    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        global v_current
        v = v_current
        d_alphas, d_gammas = np.zeros(alphas.shape), np.zeros(gammas.shape)
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x    = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        beta = np.linspace(0.001, 0.999, N_safe_sampling)
        for i, alpha, gamma in iters[::-1]:
            # build alpha and beta vector
            cur_alpha_vect = fill_parser(parser, alpha)
            cur_gamma_vect  = fill_parser(parser, gamma)

            x = (1 - beta[i]) * x_init + beta[i] * x_final
            x_previous = (1 - beta[i - 1]) * x_init + beta[i - 1] * x_final
            v = (np.subtract(x, x_previous)) / cur_alpha_vect  # recover velocity
            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_alphas[i,j] = np.dot(d_x[ixs], v[ixs])
            g = L_grad(x, meta, i)                           # Evaluate gradient

            d_v += d_x * cur_alpha_vect

            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_gammas[i,j] = np.dot(d_v[ixs], v[ixs] + g[ixs])

            d_x    -= L_hvp_x(x, meta, (1.0 - cur_gamma_vect)*d_v, i)
            d_meta -= L_hvp_meta(x, meta, (1.0 - cur_gamma_vect)* d_v, i)
            d_v    *= cur_gamma_vect
        # assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_gammas, d_meta

    return x_final, [None, hypergrad]

sgd_parsed_mad = Differentiable(sgd_parsed_mad,
                            partial(sgd_parsed_mad, forward_pass_only=False))


# Non-reversible code
###############################################

def simple_sgd(grad, x, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() has signature grad(x, i), where i is the iteration."""
    velocity = np.zeros(len(x))
    for i in xrange(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        x += step_size * velocity
    return x


def rms_prop(grad, x, callback=None, num_iters=100, step_size=0.1, gamma=0.9,
             eps=10 ** -8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))  # Is this really a sensible initialization?
    for i in xrange(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        avg_sq_grad = avg_sq_grad * gamma + g ** 2 * (1 - gamma)
        x -= step_size * g / (np.sqrt(avg_sq_grad) + eps)
    return x


def adam(grad, x, callback=None, num_iters=100,
         step_size=0.1, b1=0.1, b2=0.01, eps=10 ** -4, lam=10 ** -4):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in xrange(num_iters):
        b1t = 1 - (1 - b1) * (lam ** i)
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = b1t * g + (1 - b1t) * m  # First  moment estimate
        v = b2 * (g ** 2) + (1 - b2) * v  # Second moment estimate
        mhat = m / (1 - (1 - b1) ** (i + 1))  # Bias correction
        vhat = v / (1 - (1 - b2) ** (i + 1))
        x -= step_size * mhat / (np.sqrt(vhat) + eps)
    return x

def adam_compare(grad, x, extra, callback=None, num_iters=100,
         step_size=0.1, b1=0.1, b2=0.01, eps=10 ** -4, lam=10 ** -4):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in xrange(num_iters):
        b1t = 1 - (1 - b1) * (lam ** i)
        g = grad(x, i)
        #print("g", len(g))
        #print("x", len(x))
        
        if callback: callback(x, i, g)
        m = b1t * g + (1 - b1t) * m  # First  moment estimate
        v = b2 * (g ** 2) + (1 - b2) * v  # Second moment estimate
        mhat = m / (1 - (1 - b1) ** (i + 1))  # Bias correction
        vhat = v / (1 - (1 - b2) ** (i + 1))
        x -= step_size * mhat / (np.sqrt(vhat) + eps)
    return x


def bfgs(obj_and_grad, x, callback=None, num_iters=100):
    def epoch_counter():
        epoch = 0
        while True:
            yield epoch
            epoch += 1

    ec = epoch_counter()

    wrapped_callback = None
    if callback:
        def wrapped_callback(x):
            callback(x, next(ec))

    res = minimize(fun=obj_and_grad, x0=x, jac=True, callback=wrapped_callback,
                   options={'maxiter': num_iters, 'disp': True})
    return res.x
