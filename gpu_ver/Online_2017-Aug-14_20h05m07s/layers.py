import numpy as np
import theano.tensor as T

#from lasagne.layers import Layer
from lasagne import nonlinearities, init
eps = np.float32(1e-8)
zero = np.float32(0.)
one = np.float32(1.)


from lasagne.utils import as_tuple
#from lasagne.theano_extensions import conv

from lasagne.layers.base import Layer
from lasagne.layers.conv import conv_output_length

#Gaussian noise
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# TODO: updates Jul 30 5pm: added clipping to weights and scales, between -10^3 and 10^3
# switched scales back to regular scale (no longer log-scale) and regularized them
# This is because if x is regularized and y is not, then x*y=c has solutions with x->0 and y unbounded
# which may cause overflow in y and underflow in x
# OTOH, if (WLOG) xy=1, then the regularization x^2+y^2=x^2+1/x^2 is minimized at x=y=1
# but x^2 and y^2 have hyperparameter coefficients which may change, 
# and one of x or y could no longer be strongly penalized, allowing it to overflow
# and the other to underflow. Hence clipping (in the SGD steps in updates.py). 
# TODO: Jul 31 2:36 AM: all errors and all losses but the *regularized* training loss are periodic and not decreasing significantly
# So remove regularization from ScaleLayers?
# Or it could be the W=0 initialization, or both
# Fix initialization first


def nan_o_meter(x):
    n_nans = T.sum(T.isnan(x))
    n_infs = T.sum(T.isinf(x))
    print('# Nans ', n_nans, '# Infs', n_infs) 
    return
    

def relu(x):
    return T.maximum(x, 0)






class DenseLayerWithReg(Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=1, **kwargs)
    A fully connected layer.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        The number of units of the layer
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    num_leading_axes : int
        Number of leading axes to distribute the dot product over. These axes
        will be kept in the output tensor, remaining axes will be collapsed and
        multiplied against the weight matrix. A negative number gives the
        (negated) number of trailing axes to involve in the dot product.
    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    If the input has more than two axes, by default, all trailing axes will be
    flattened. This is useful when a dense layer follows a convolutional layer.
    >>> l_in = InputLayer((None, 10, 20, 30))
    >>> DenseLayer(l_in, num_units=50).output_shape
    (None, 50)
    Using the `num_leading_axes` argument, you can specify to keep more than
    just the first axis. E.g., to apply the same dot product to each step of a
    batch of time sequences, you would want to keep the first two axes.
    >>> DenseLayer(l_in, num_units=50, num_leading_axes=2).output_shape
    (None, 10, 50)
    >>> DenseLayer(l_in, num_units=50, num_leading_axes=-1).output_shape
    (None, 10, 20, 50)
    """
    def __init__(self, args, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        super(DenseLayerWithReg, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "leaving no trailing axes for the dot product." %
                    (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "requesting more trailing axes than there are input "
                    "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                    "A DenseLayer requires a fixed input shape (except for "
                    "the leading axes). Got %r for num_leading_axes=%d." %
                    (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

        self.L1 = None
        self.L2 = None
        #self.initScale = None
        if args.regL1 is True:
            self.L1 = self.add_param(init.Constant(args.regInit['L1']),
                                     (num_inputs, num_units), name="L1")
        if args.regL2 is True:
            self.L2 = self.add_param(init.Constant(args.regInit['L2']),
                                     (num_inputs, num_units), name="L2")




    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)














class BaseConvLayerWithReg(Layer):
    """
    lasagne.layers.BaseConvLayer(incoming, num_filters, filter_size,
    stride=1, pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    n=None, **kwargs)
    Convolutional layer base class
    Base class for performing an `n`-dimensional convolution on its input,
    optionally adding a bias and applying an elementwise nonlinearity. Note
    that this class cannot be used in a Lasagne network, only its subclasses
    can (e.g., :class:`Conv1DLayer`, :class:`Conv2DLayer`).
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`n` dimensions:
        ``(batch_size, num_input_channels, <n spatial dimensions>)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or an `n`-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or an `n`-element tuple specifying the stride of the
        convolution operation.
    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of `n` integers allows different symmetric padding
        per dimension.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be an
        `n`-dimensional tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`n` dimensions with shape
        ``(num_filters, num_input_channels, <n spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <n spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.
    n : int or None
        The dimensionality of the convolution (i.e., the number of spatial
        dimensions of each feature map and each convolutional filter). If
        ``None``, will be inferred from the input shape.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, args, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 n=None, **kwargs):
        super(BaseConvLayerWithReg, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)
        
        self.L1 = None
        self.L2 = None
        #self.initScale = None
        if args.regL1 is True:
            self.L1 = self.add_param(init.Constant(args.regInit['L1']),
                                     self.get_W_shape() , name="L1") # Michael: This is a "TensorSharedVariable"
        if args.regL2 is True:
            self.L2 = self.add_param(init.Constant(args.regInit['L2']),
                                     self.get_W_shape() , name="L2") # Michael: This is a "TensorSharedVariable"
        

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.
        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.
        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`
        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
                                  "use a subclass such as Conv2DLayer.")



class Conv2DLayerWithReg(BaseConvLayerWithReg):
    """
    lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    convolution=theano.tensor.nnet.conv2d, **kwargs)
    2D convolutional layer
    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.
    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.
    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, args, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DLayerWithReg, self).__init__(args, incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=2,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved


# Replaced 'scale' with 'b' (not actually a bias, but b is now used to just represent unregularized weights)
# These are on a logarithmic scale instead (so always positive)
# Note that between each Scale Layer and the previous convolution are only linear layers
# which themselves can be positive or negative, so having negative ScaleLayer parameters shouldn't be as useful

#update Jul30 4:15pm: 'scale' is now W, and (log-scale) regularized and clipped between -5 and 1 (10^-5 and 10)

#TODO: update Jul30 5pm, 'scale' is now W and regularized, but no longer log-scale
#if x=conv weight and y=scale, then for constant xy=c, the output is the same,
#but with L2 regularization, and c=1, x^2+y^2 = x^2+1/x^2, which is minimized at x=y=1
#so this should help keep the scales from exploding


#TODO: regularized scale layer
class ScaleLayer(Layer):
    """
    lasagne.layers.ScaleLayer(incoming, scales=lasagne.init.Constant(1),
    shared_axes='auto', **kwargs)
    A layer that scales its inputs by learned coefficients.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    scales : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for the scale.  The scale
        shape must match the incoming shape, skipping those axes the scales are
        shared over (see the example below).  See
        :func:`lasagne.utils.create_param` for more information.
    shared_axes : 'auto', int or tuple of int
        The axis or axes to share scales over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share scales over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    Notes
    -----
    The scales parameter dimensionality is the input dimensionality minus the
    number of axes the scales are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:
    >>> layer = ScaleLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.scales.get_value().shape
    (30, 50)
    """
    #def __init__(self, args, incoming, scale=init.Constant(1), shared_axes='auto',
    #             **kwargs):
    def __init__(self, args, incoming, W=init.Constant(1), shared_axes='auto',
                 **kwargs):
        super(ScaleLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            # default: share scales over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        # create scales parameter, ignoring all dimensions in shared_axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ScaleLayer needs specified input sizes for "
                             "all axes that scales are not shared over.")
        #self.b = self.add_param(init.Constant(), shape, 'scales', regularizable=False)
        #self.initScale = self.add_param(init.Constant(args.initScaleInit), shape, 'initScales', regularizable=False)
        #TODO: need to reinitialize W every meta-iteration
        #self.W = self.add_param(self.initScale.get_value(), shape, 'scales', regularizable=False) #on a logarithmic scale
        #print(self.initScale.get_value().shape, self.W.get_value().shape)
        self.W = self.add_param(W, shape, 'scales', regularizable=True) #, regularizable=False)
        self.b = None
        self.L1 = None
        self.L2 = None
        if args.regL1 is True:
            self.L1 = self.add_param(init.Constant(args.regInit['L1']),
                                     shape, name="L1")
        if args.regL2 is True:
            self.L2 = self.add_param(init.Constant(args.regInit['L2']),
                                     shape, name="L2")


    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.W.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes) for input_axis in range(input.ndim)]
        #return input * self.W.dimshuffle(*pattern)
        #return input * (10.**self.W.dimshuffle(*pattern)) #base 10 logarithmic scale
        return input * (self.W.dimshuffle(*pattern))


#TODO: without regularization
class ScaleLayerUnreg(Layer):
    """
    lasagne.layers.ScaleLayer(incoming, scales=lasagne.init.Constant(1),
    shared_axes='auto', **kwargs)
    A layer that scales its inputs by learned coefficients.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    scales : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for the scale.  The scale
        shape must match the incoming shape, skipping those axes the scales are
        shared over (see the example below).  See
        :func:`lasagne.utils.create_param` for more information.
    shared_axes : 'auto', int or tuple of int
        The axis or axes to share scales over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share scales over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    Notes
    -----
    The scales parameter dimensionality is the input dimensionality minus the
    number of axes the scales are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:
    >>> layer = ScaleLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.scales.get_value().shape
    (30, 50)
    """
    #def __init__(self, args, incoming, scale=init.Constant(1), shared_axes='auto',
    #             **kwargs):
    def __init__(self, args, incoming, b=init.Constant(1), shared_axes='auto',
                 **kwargs):
        super(ScaleLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            # default: share scales over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        # create scales parameter, ignoring all dimensions in shared_axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ScaleLayer needs specified input sizes for "
                             "all axes that scales are not shared over.")
        #self.b = self.add_param(init.Constant(), shape, 'scales', regularizable=False)
        #self.initScale = self.add_param(init.Constant(args.initScaleInit), shape, 'initScales', regularizable=False)
        #TODO: need to reinitialize W every meta-iteration
        #self.W = self.add_param(self.initScale.get_value(), shape, 'scales', regularizable=False) #on a logarithmic scale
        #print(self.initScale.get_value().shape, self.W.get_value().shape)
        self.b = self.add_param(b, shape, 'scales', regularizable=False)
        self.W = None
        self.L1 = None
        self.L2 = None
        """if args.regL1 is True:
            self.L1 = self.add_param(init.Constant(args.regInit['L1']),
                                     shape, name="L1")
        if args.regL2 is True:
            self.L2 = self.add_param(init.Constant(args.regInit['L2']),
                                     shape, name="L2")"""


    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.b.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes) for input_axis in range(input.ndim)]
        #return input * self.W.dimshuffle(*pattern)
        #return input * (10.**self.W.dimshuffle(*pattern)) #base 10 logarithmic scale
        return input * (self.b.dimshuffle(*pattern))







class BiasLayer(Layer):
    """
    lasagne.layers.BiasLayer(incoming, b=lasagne.init.Constant(0),
    shared_axes='auto', **kwargs)
    A layer that just adds a (trainable) bias term.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases and pass through its input
        unchanged. Otherwise, the bias shape must match the incoming shape,
        skipping those axes the biases are shared over (see the example below).
        See :func:`lasagne.utils.create_param` for more information.
    shared_axes : 'auto', int or tuple of int
        The axis or axes to share biases over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share biases over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    Notes
    -----
    The bias parameter dimensionality is the input dimensionality minus the
    number of axes the biases are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:
    >>> layer = BiasLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.b.get_value().shape
    (30, 50)
    """
    def __init__(self, args, incoming, b=init.Constant(0), shared_axes='auto',
                 **kwargs):
        super(BiasLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            # default: share biases over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        if b is None:
            self.b = None
        else:
            # create bias parameter, ignoring all dimensions in shared_axes
            shape = [size for axis, size in enumerate(self.input_shape)
                     if axis not in self.shared_axes]
            if any(size is None for size in shape):
                raise ValueError("BiasLayer needs specified input sizes for "
                                 "all axes that biases are not shared over.")
            self.b = self.add_param(b, shape, 'b', regularizable=False)
        self.W = None
        self.L1 = None
        self.L2 = None
        #self.initScale = None


    def get_output_for(self, input, **kwargs):
        if self.b is not None:
            bias_axes = iter(range(self.b.ndim))
            pattern = ['x' if input_axis in self.shared_axes
                       else next(bias_axes)
                       for input_axis in range(input.ndim)]
            return input + self.b.dimshuffle(*pattern)
        else:
            return input









class GaussianNoiseLayer(Layer):
    """Gaussian noise layer.
    Add zero-mean Gaussian noise of given standard deviation to the input [1]_.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape
    sigma : float or tensor scalar
            Standard deviation of added Gaussian noise
    Notes
    -----
    The Gaussian noise layer is a regularizer. During training you should set
    deterministic to false and during testing you should set deterministic to
    true.
    References
    ----------
    .. [1] K.-C. Jim, C. Giles, and B. Horne (1996):
           An analysis of noise in recurrent neural networks: convergence and
           generalization.
           IEEE Transactions on Neural Networks, 7(6):1424-1438.
    """
    def __init__(self, incoming, sigma=init.Constant(-1), shared_axes='all', **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)        
        if shared_axes == 'auto': #share sigma over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all': #share sigma over all axes (e.g. for input)
            shared_axes = tuple(range(0, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        else:
            shared_axes = ()

        self.shared_axes = shared_axes
        
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("GaussianNoiseLayer needs specified input sizes for "
                             "all axes that sigmas are not shared over.")
        #sigma on log10 scale
        self.sigma = self.add_param(sigma, shape, 'sigma', regularizable=False)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        if deterministic or self.sigma == 0:
            return input
        else:
            noise_axes = iter(range(self.sigma.ndim))
            pattern = ['x' if input_axis in self.shared_axes
                       else next(noise_axes)
                       for input_axis in range(input.ndim)]
            return input + ((1/(1+T.exp(-self.sigma))).dimshuffle(*pattern))*self._srng.normal(input.shape, avg=0.0, std=1.0)
            #TODO: broadcastable dimensions
    def reinit(self):
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
























# add reinitialize function

class DropoutLayer(Layer):
    """Dropout layer
    Sets values to zero with probability p. See notes for disabling dropout
    during testing.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If ``True`` (the default), scale the input by ``1 / (1 - p)`` when
        dropout is enabled, to keep the expected output mean the same.
    shared_axes : tuple of int
        Axes to share the dropout mask over. By default, each value can be
        dropped individually. ``shared_axes=(0,)`` uses the same mask across
        the batch. ``shared_axes=(2, 3)`` uses the same mask across the
        spatial dimensions of 2D feature maps.
    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    The behaviour of the layer depends on the ``deterministic`` keyword
    argument passed to :func:`lasagne.layers.get_output`. If ``True``, the
    layer behaves deterministically, and passes on the input unchanged. If
    ``False`` or not specified, dropout (and possibly scaling) is enabled.
    Usually, you would use ``deterministic=False`` at train time and
    ``deterministic=True`` at test time.
    See also
    --------
    dropout_channels : Drops full channels of feature maps
    spatial_dropout : Alias for :func:`dropout_channels`
    dropout_locations : Drops full pixels or voxels of feature maps
    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.
    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, incoming, p=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        #TODO: use same random
        #self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        r = get_rng().randint(1, 2147462579)
        self._srng = RandomStreams(r)
        print(self, r)
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1, dtype='int8')

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            return input * mask
    def reinit(self):
        r = get_rng().randint(1, 2147462579)
        self._srng = RandomStreams(r)
        print(self, r)
        #self._srgn = RandomStreams(get_rng().randint(1, 2147462579))

dropout = DropoutLayer  # shortcut