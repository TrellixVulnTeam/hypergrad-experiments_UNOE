import theano.tensor as T
import theano
# TODO: more robust implementation: address problems with Lop
from theano.compat import OrderedDict, izip
import numpy as np

# hypers influencing only penalty term (cause Theano)
penalty_list = ['L1', 'L2', 'Lmax', 'LmaxSlope', 'LmaxCutoff', 'LmaxHard']
# hypers influencing only NLL (cause Theano)
noise_list = ['addNoise', 'inputNoise']


def format_as(use_list, use_tuple, outputs):
    """
    Formats the outputs according to the flags `use_list` and `use_tuple`.
    If `use_list` is True, `outputs` is returned as a list (if `outputs`
    is not a list or a tuple then it is converted in a one element list).
    If `use_tuple` is True, `outputs` is returned as a tuple (if `outputs`
    is not a list or a tuple then it is converted into a one element tuple).
    Otherwise (if both flags are false), `outputs` is returned.
    """
    assert not (use_list and use_tuple), \
        "Both flags cannot be simultaneously True"
    if (use_list or use_tuple) and not isinstance(outputs, (list, tuple)):
        if use_list:
            return [outputs]
        else:
            return (outputs,)
    elif not (use_list or use_tuple) and isinstance(outputs, (list, tuple)):
        assert len(outputs) == 1, \
            "Wrong arguments. Expected a one element list"
        return outputs[0]
    elif use_list or use_tuple:
        if use_list:
            return list(outputs)
        else:
            return tuple(outputs)
    else:
        return outputs


def Lop2(f, wrt, eval_points):
    res = []
    for t1, t2, t3 in zip(f, wrt, eval_points):
        res.append(T.grad(T.sum(t1 * t3), t2, disconnected_inputs='warn')) #, disconnected_inputs='ignore')
    return res

"""
    parser.add_argument('--classes',
                        type=int, default=10)
    parser.add_argument('--depth',
                        type=int, default=4) #40 #"depth must be num_blocks * n + 1 for some n"
    parser.add_argument('--first_output',
                        type=int, default=8) #16
    parser.add_argument('--growth_rate',
                        type=int, default=12)
    parser.add_argument('--num_blocks',
                        type=int, default=3)
    parser.add_argument('--dropout',
                        type=float, default=0)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: block1_trs_conv.L2
  handle_disconnected(elem)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:605: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: <DisconnectedType>
  handle_disconnected(rval[i])
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: block2_trs_conv.L2
  handle_disconnected(elem)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: output.L2
  handle_disconnected(elem)


    parser.add_argument('--classes',
                        type=int, default=10)
    parser.add_argument('--depth',
                        type=int, default=7) #40 #"depth must be num_blocks * n + 1 for some n"
    parser.add_argument('--first_output',
                        type=int, default=8) #16
    parser.add_argument('--growth_rate',
                        type=int, default=12)
    parser.add_argument('--num_blocks',
                        type=int, default=3)
    parser.add_argument('--dropout',
                        type=float, default=0)


[pre_conv.W, block1_l01_scale.scales, block1_l01_shift.b, block1_l01_conv.W, block1_trs_scale.scales, block1_trs_shift.b, block1_trs_conv.W, block2_l01_scale.scales, block2_l01_shift.b, block2_l01_conv.W, block2_trs_scale.scales, block2_trs_shift.b, block2_trs_conv.W, block3_l01_scale.scales, block3_l01_shift.b, block3_l01_conv.W, post_scale.scales, post_shift.b, output.W, output.b]
[pre_conv.W, block1_l01_scale.scales, block1_l01_conv.W, block1_trs_scale.scales, block1_trs_conv.W, block2_l01_scale.scales, block2_l01_conv.W, block2_trs_scale.scales, block2_trs_conv.W, block3_l01_scale.scales, block3_l01_conv.W, post_scale.scales, output.W]
[pre_conv.L2, block1_l01_conv.L2, block1_trs_conv.L2, block2_l01_conv.L2, block2_trs_conv.L2, block3_l01_conv.L2, output.L2]
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: block1_l01_conv.L2
  handle_disconnected(elem)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:605: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: <DisconnectedType>
  handle_disconnected(rval[i])
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: block1_trs_conv.L2
  handle_disconnected(elem)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: block2_l01_conv.L2
  handle_disconnected(elem)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: block2_trs_conv.L2
  handle_disconnected(elem)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: block3_l01_conv.L2
  handle_disconnected(elem)
//anaconda/lib/python2.7/site-packages/theano/gradient.py:579: UserWarning: grad method was asked to compute the gradient with respect to a variable that is not part of the computational graph of the cost, or is used only by a non-differentiable operator: output.L2
  handle_disconnected(elem)
"""

def Lop(f, wrt, eval_points, consider_constant=None,
        disconnected_inputs='ignore'): #TODO: disconnected_inputs='raise'):
    """

        This Lop() has the same functionality of Theano.Lop()
    """
    if type(eval_points) not in (list, tuple):
        eval_points = [eval_points]

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if not isinstance(f, (list, tuple)):
        f = [f]

    # make copies of f and grads so we don't modify the client's copy
    f = list(f)
    grads = list(eval_points)

    # var_grads = []
    # for grad in grads:
    #     var_grad = theano.shared(grad, name="let me see", allow_downcast=True)
    #     var_grads += [var_grad]

    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]

    assert len(f) == len(grads)
    known = dict(izip(f, grads))

    # print "I know nothing.", known, "\n\n\n", f, "\n\n\n", grads, type(grads[0])

    ret = T.grad(cost=None, known_grads=known,
               consider_constant=consider_constant, wrt=wrt,
               disconnected_inputs=disconnected_inputs)

    # print "return value", ret[0], type(ret[0])

    return format_as(using_list, using_tuple, ret)

def hypergrad(params_lambda, params_weight, dL_dweight, grad_valid_weight):
    #TODO: grad_valid_weight should be dv_t ??

    # theano.gradient.Lop(f, wrt, eval_points, consider_constant=None,
    #                     disconnected_inputs='raise')
    #     Computes the L operation on `f` wrt to `wrt` evaluated at points given
    # in `eval_points`. Mathematically this stands for the jacobian of `f` wrt
    # to `wrt` left muliplied by the eval points.

    #print("hypergrad_penalty_weight")
    hypergrad_penalty_weight = Lop2(dL_dweight, params_weight, grad_valid_weight)
    #TODO: Disconnected weights. 
    #print("hypergrad_penalty_lambda")
    hypergrad_penalty_lambda = Lop2(dL_dweight, params_lambda, grad_valid_weight)
    #for param, lamb, hypergrad in zip(params_weight, params_lambda, hypergrad_penalty_lambda):
        #print(lamb.type)
        #print(type(lamb))
        #print(np.linalg.norm(10.**lamb*np.log(10.)*2.*param-hypergrad))
    #print("lambda", len(params_lambda),len(hypergrad_penalty_lambda))
    # For L2 loss, dw dlambda L(w, lambda) = dw dlambda 10^lambda w^2 (pointwise) = 2w * 10^lambda * ln(10) (diagonal)
    # For L1 loss, it's 10^lambda * ln(10)
    #hypergrad is only called once
    return hypergrad_penalty_weight, hypergrad_penalty_lambda