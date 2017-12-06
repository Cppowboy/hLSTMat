import theano
import numpy
import theano.tensor as tensor
from utils import _p, norm_weight, ortho_weight, tanh, linear
from layers import Layers
from non_local_layers import NonLocalLayers


class LSTMNonLocalLayers(object):
    def __init__(self, lstm_layers):
        self.lstm_layers = lstm_layers
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'lstm_non_local_layer': ('self.param_init_lstm_non_local_layer', 'self.lstm_non_local_layer')
        }

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return (eval(fns[0]), eval(fns[1]))

    def param_init_lstm_non_local_layer(self, options, params, prefix='lstm_non_local_layer', twh=7 * 7 * 28, c=512):
        params[_p(prefix, 'W_theta')] = norm_weight(c, c, scale=0.01)
        params[_p(prefix, 'W_phi')] = norm_weight(c, c, scale=0.01)
        params[_p(prefix, 'W_g')] = norm_weight(c, c, scale=0.01)
        params = self.lstm_layers.get_layer('lstm')[0](params, nin=options['ctx_dim'], dim=options['ctx_dim'],
                                                       prefix='self_att_lstm')
        return params

    def lstm_non_local_layer(self, tparams, state_below, options,
                             prefix='lstm_non_local_layer', **kwargs):
        W_theta = tparams[_p(prefix, 'W_theta')]
        W_phi = tparams[_p(prefix, 'W_phi')]
        W_g = tparams[_p(prefix, 'W_g')]

        c = W_theta.shape[0]
        twh = options['K']
        T = options['T']
        wh = options['wh']
        batch = options['batch_size']
        if state_below.ndim == 2:
            # state_below: twh * c
            x_theta = tensor.dot(state_below, W_theta)  # (twh, c) * (c, c) = (twh, c)
            x_phi = tensor.dot(state_below, W_phi)  # (twh, c) * (c, c) = (twh, c)
            g_x = tensor.dot(state_below, W_g)  # (twh, c) * (c, c) = (twh, c)

            twh_mat = tensor.dot(x_theta, x_phi.T)  # (twh, c) * (c, twh) = (twh, twh)
            twh_mat = tensor.nnet.softmax(twh_mat)  # (twh, twh) -> (twh, twh)

            twh_mat = tensor.reshape(twh_mat, (twh, T, wh))  # (twh, twh) -> (twh, t, wh)
            twh_mat = tensor.transpose(twh_mat, (1, 0, 2))  # (twh, t, wh) -> (t, twh, wh)
            g_x = tensor.reshape(g_x, (T, wh, c))  # (twh, c) -> (t, wh, c)

            y = tensor.batched_dot(twh_mat, g_x)  # (t, twh, wh) .* (t, wh, c) = (t, twh, c)
            y = tensor.transpose(y, (1, 0, 2))  # (t, twh, c) -> (twh, t, c)

            self_att_lstm = self.lstm_layers.get_layer('lstm')[1](tparams, y,
                                                                  mask=None,
                                                                  one_step=False,
                                                                  prefix='self_att_lstm'
                                                                  )  # (twh, t, c)
            ans = self_att_lstm[0][-1]  # (twh, -1, c) -> (twh, c)
        elif state_below.ndim == 3:
            # state_below: batch * twh * c
            x_theta = tensor.dot(state_below, W_theta)  # (batch, twh, c) * (c, c) = (batch, twh, c)
            x_phi = tensor.dot(state_below, W_phi)  # (batch, twh, c) * (c, c) = (batch, twh, c)
            g_x = tensor.dot(state_below, W_g)  # (batch, twh, c) * (c, c) = (batch, twh, c)

            twh_mat = tensor.batched_dot(x_theta, tensor.transpose(x_phi, (
                0, 2, 1)))  # (batch, twh, c) * (batch, c, twh) = (batch, twh, twh)
            twh_mat = tensor.reshape(twh_mat, (-1, twh))  # (batch, twh, twh) -> (batch * twh, twh)
            twh_mat = tensor.nnet.softmax(twh_mat)  # (batch * twh, twh) -> (batch * twh, twh)

            twh_mat = tensor.reshape(twh_mat, (-1, twh, T, wh))  # (batch, twh, twh) -> (batch, twh, t, wh)
            twh_mat = tensor.transpose(twh_mat, (0, 2, 1, 3))  # (batch, twh, t, wh) -> (batch, t, twh, wh)
            twh_mat = tensor.reshape(twh_mat, (-1, twh, wh))  # (batch, t, twh, wh) -> (batch * t, twh, wh)

            g_x = tensor.reshape(g_x, (-1, wh, c))  # (batch, twh, c) -> (batch * t, wh, c)

            y = tensor.batched_dot(twh_mat, g_x)  # (batch * t, twh, wh) .* (batch * t, wh, c) = (batch * t, twh, c)
            y = tensor.reshape(y, (batch, T, twh, c))  # (batch * t, twh, c) -> (batch, t, twh, c)
            y = tensor.transpose(y, (0, 2, 1, 3))  # (batch, t, twh, c) -> (batch, twh, t, c)
            y = tensor.reshape(y, (batch * twh, T, c))  # (batch, twh, t, c) -> (batch * twh, t, c)

            self_att_lstm = self.lstm_layers.get_layer('lstm')[1](tparams, y,
                                                                  mask=None,
                                                                  one_step=False,
                                                                  prefix='self_att_lstm')  # (batch * twh, t, c)
            ans = self_att_lstm[0][:, -1, :]  # (batch * twh, -1, c) -> (batch * twh, c)
            ans = tensor.reshape(ans, (batch, twh, c))  # (batch * twh, c) -> (batch, twh, c)
        else:
            raise Exception('bad input dim %d' % (state_below.ndim))
        return ans
