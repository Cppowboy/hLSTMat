import theano
import numpy
import theano.tensor as tensor
from utils import _p, norm_weight, ortho_weight, tanh, linear


class NonLocalLayers(object):
    def __init__(self):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'non_local_layer': ('self.param_init_non_local_layer', 'self.non_local_layer')
        }

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return (eval(fns[0]), eval(fns[1]))

    def param_init_non_local_layer(self, options, params, prefix='non_local_layer', twh=7 * 7 * 28, c=512):
        params[_p(prefix, 'W_theta')] = norm_weight(c, c, scale=0.01)
        params[_p(prefix, 'W_phi')] = norm_weight(c, c, scale=0.01)
        params[_p(prefix, 'W_g')] = norm_weight(twh, twh, scale=0.01)
        return params

    def non_local_layer(self, tparams, state_below, options,
                        prefix='non_local_layer', **kwargs):
        W_theta = tparams[_p(prefix, 'W_theta')]
        W_phi = tparams[_p(prefix, 'W_phi')]
        W_g = tparams[_p(prefix, 'W_g')]

        c = W_theta.shape[0]
        twh = W_g.shape[0]

        g = tensor.dot(W_g, state_below)
        if state_below.ndim == 2:
            y = tensor.dot(state_below, W_theta)
            y = tensor.dot(y, W_phi)
            y = tensor.dot(y, state_below.T)
            y = tensor.nnet.softmax(y)
            y = tensor.dot(y, g)
        elif state_below.ndim == 3:
            y = tensor.dot(state_below, W_theta)
            y = tensor.dot(y, W_phi)
            y = tensor.batched_dot(y, state_below.transpose(0, 2, 1))
            y = y.reshape((-1, twh))
            y = tensor.nnet.softmax(y)
            y = y.reshape(-1, twh, twh)
            y = tensor.dot(y, g)
        else:
            raise Exception('bad input dim %d' % (state_below.ndim))
        return y
