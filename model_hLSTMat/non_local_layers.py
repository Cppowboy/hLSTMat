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
        params[_p(prefix, 'W_g')] = norm_weight(c, c, scale=0.01)
        return params

    def non_local_layer(self, tparams, state_below, options,
                        prefix='non_local_layer', **kwargs):
        W_theta = tparams[_p(prefix, 'W_theta')]
        W_phi = tparams[_p(prefix, 'W_phi')]
        W_g = tparams[_p(prefix, 'W_g')]

        c = W_theta.shape[0]
        twh = W_g.shape[0]

        if state_below.ndim == 2:
            x_theta = tensor.dot(state_below, W_theta)  # (twh, c) * (c, c) = (twh, c)
            x_phi = tensor.dot(state_below, W_phi)  # (twh, c) * (c, c) = (twh, c)

            twh_mat = tensor.dot(x_theta, x_phi.T)  # (twh, c) * (c, twh) = (twh, twh)
            twh_mat = tensor.nnet.softmax(twh_mat)  # (twh, twh) -> (twh, twh)

            g_x = tensor.dot(W_g, state_below)  # (twh, twh) * (twh, c) = (twh, c)

            y = tensor.dot(twh_mat, g_x)  # (twh, twh) * (twh, c) = (twh, c)
        elif state_below.ndim == 3:
            x_theta = tensor.dot(state_below, W_theta)  # (batch, twh, c) * (c, c) = (batch, twh, c)
            x_phi = tensor.dot(state_below, W_phi)  # (batch, twh, c) * (c, c) = (batch, twh, c)

            twh_mat = tensor.batched_dot(x_theta, tensor.transpose(x_phi, (
                0, 2, 1)))  # (batch, twh, c) .* (batch, c, twh) = (batch, twh, twh)
            twh_mat = tensor.reshape(twh_mat, (-1, twh))  # (batch, twh, twh) -> (batch * twh, twh)
            twh_mat = tensor.nnet.softmax(twh_mat)  # (batch * twh, twh)
            twh_mat = tensor.reshape(twh_mat, (-1, twh, twh))  # (batch * twh, twh) -> (batch, twh, twh)

            g_x = tensor.dot(state_below, W_g)  # (batch, twh, c) * (c, c) = (batch, twh, c)

            y = tensor.batched_dot(twh_mat, g_x)  # (batch, twh, twh) .* (batch, twh, c) = (batch, twh, c)
        else:
            raise Exception('bad input dim %d' % (state_below.ndim))
        return y
