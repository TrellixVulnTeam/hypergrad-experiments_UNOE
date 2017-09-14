# Experiments with gradient-based hyperparameter optimization 

For [my master's project](https://uwaterloo.ca/computational-mathematics/sites/ca.computational-mathematics/files/uploads/files/michael_st._jules.pdf).

### Abstract

Gradient-based hyperparameter optimization algorithms have the potential to scale to numbers of individual hyperparameters proportional to the number of elementary parameters, unlike other current approaches. Some candidate completions of DrMAD, one such algorithm that updates the hyperparameters after fully training the parameters of the model, are explored, with experiments tuning per-parameter L2 regularization coefficients on CIFAR10 with the DenseNet architecture. Experiments with DenseNets on CIFAR10 are also conducted with an adaptive method, which updates the hyperparameters during the training of elementary parameters, tuning per-parameter learning rates and L2 regularization. The experiments do not establish the utility of either method, but the adaptive method shows some promise, with further experiments required.

### Primary references: 

[DrMAD: Distilling Reverse-Mode Automatic Differentiation
for Optimizing Hyperparameters of Deep Neural Networks](https://www.ijcai.org/Proceedings/16/Papers/211.pdf) ([code](https://github.com/bigaidream-projects/drmad))

[Gradient-based Hyperparameter Optimization through Reversible Learning](http://proceedings.mlr.press/v37/maclaurin15.pdf) ([code](https://github.com/HIPS/hypergrad))

[Scalable Gradient-Based Tuning of
Continuous Regularization Hyperparameters](http://proceedings.mlr.press/v48/luketina16.pdf) ([code](https://github.com/jelennal/t1t2))

[Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782)

[On-Line Step Size Adaptation](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.8519)

[Parameter Adaptation in Stochastic Optimization](https://www.cambridge.org/core/books/on-line-learning-in-neural-networks/parameter-adaptation-in-stochastic-optimization/4E8D5E86F4E2634EE29CC363C0568222)

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

The code here is edited from https://github.com/bigaidream-projects/drmad (and https://github.com/HIPS/hypergrad), and https://github.com/Lasagne/Recipes/tree/master/papers/densenet.

The CPU version (cpu-ver) is written in Python and uses https://github.com/HIPS/autograd, while the GPU version (gpu-ver) is written in Theano. 
