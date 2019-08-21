"""
/*******************************************
** This is a file created by CNALeon007
** Name: TrajGRU
** Date: 8/21/19
** BSD license
********************************************/
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# S: seq_len
# B: batch
# H: height
# W: width
# C: filter

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B, H, W)
    - y: flattened tensor of shape (B, H, W)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def wrap(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - x of shape (B, H, W): elements of x are in [-1,1].
    - y of shape (B, H, W): elements of y are in [-1,1]

    Returns
    -------
    - out of shape (B, H, W, C): interpolated images according to grids x and y.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


class BaseConvRNN(keras.Model):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type=tf.math.tanh,
                 prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)

        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]

        self._batch_size, self._height, self._width = b_h_w

        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h) \
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                            // self._i2h_stride[1] + 1

class TrajGRU(BaseConvRNN):
    def __init__(self, num_filter, b_h_w, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type=tf.nn.leaky_relu):
        """
        Input
        ------
        - num_filter: hidden size
        - b_h_w: tuple of (B, H, W)
        - zoneout: dropout
        - L: number of links
        - i2h_kernel: 2D tuple, input to hidden
        - i2h_stride: 2D tuple, input to hidden
        - i2h_pad: 2D tuple, input to hidden
        - h2h_kernel: 2D tuple, hidden to hidden
        - h2h_dilate: 2D tuple, hidden to hidden
        - act_type: type of activation function
        """
        super(TrajGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU')
        self._L = L
        self._zoneout = zoneout

        # Computational part of Xt
        self.i2h = keras.layers.Conv2D(filters=self._num_filter * 3,
                                       kernel_size=self._i2h_kernel,
                                       strides=self._i2h_stride,
                                       padding="same",
                                       dilation_rate=self._i2h_dilate)

        # inputs to flow
        self.i2f_conv1 = keras.layers.Conv2D(filters=32,
                                             kernel_size=(5, 5),
                                             strides=1,
                                             padding="same",
                                             dilation_rate=(1, 1))

        # hidden to flow
        self.h2f_conv1 = keras.layers.Conv2D(filters=32,
                                             kernel_size=(5, 5),
                                             strides=1,
                                             padding="same",
                                             dilation_rate=(1, 1))

        # generate flow
        self.flows_conv = keras.layers.Conv2D(filters=self._L * 2,
                                              kernel_size=(5, 5),
                                              strides=1,
                                              padding="same")

        # Computational part of Ht
        self.ret = keras.layers.Conv2D(filters=self._num_filter * 3,
                                       kernel_size=(1, 1),
                                       strides=1)

    def _flow_generator(self, inputs, states):
        """
        Generating flow images, which are also called sampling grids.

        Input
        ------
        - inputs of shape (B, H, W, C): inputs
        - states of shape (B, H, W, C): previous hidden states

        Return
        ------
        - flow of shape (L, B, H, W, 2): L*2 flow images, which are also called sampling grids
        """
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1  # 这里看，inputs的H，W和state的H，W应该一样
        f_conv1 = self._act_type(f_conv1, alpha=0.2)

        flows = self.flows_conv(f_conv1)  # size<B, H, W, 2*L>
        flows = tf.split(flows, num_or_size_splits=self._L, axis=3)  # 第1维分割，2个一组 size<L, B, H, W, 2>
        return flows

    def call(self, inputs=None, states=None):
        """
        TrajGRU

        Input
        ------
        - inputs of shape (S, B, H, W, C): inputs
        - state of shape (B, H, W, C): previous hidden states

        Return
        ------
        - tf.stack(outputs) of shape (S, B, H, W, hidden_size): history hidden states
        - next_h of shape (B, H, W, hidden_size): output
        ------

        """
        seq_len = tf.shape(inputs).numpy()[0]
        if states is None:
            states = tf.zeros((tf.shape(inputs).numpy()[1], self._state_height,
                               self._state_width, self._num_filter))
            states = tf.cast(states, 'float32')
        if inputs is not None:
            S, B, H, W, C = tf.shape(inputs).numpy()
            i2h = self.i2h(tf.reshape(inputs, (-1, H, W, C)))
            i2h = tf.reshape(i2h, (S, B, tf.shape(i2h).numpy()[1], tf.shape(i2h).numpy()[2],
                                   tf.shape(i2h).numpy()[3]))
            i2h_slice = tf.split(i2h, num_or_size_splits=3, axis=4)

        else:
            i2h_slice = None

        prev_h = states
        outputs = []
        for i in range(seq_len):
            if inputs is not None:
                flows = self._flow_generator(inputs[i, ...], prev_h)
            else:
                flows = self._flow_generator(None, prev_h)
            wrapped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                wrapped_data.append(wrap(prev_h, -flow[:, :, :, 0], -flow[:, :, :, 1]))

            wrapped_data = tf.concat(wrapped_data, axis=3)
            h2h = self.ret(wrapped_data)
            h2h_slice = tf.split(h2h, 3, axis=3)

            if i2h_slice is not None:
                reset_gate = tf.math.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = tf.math.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2], alpha=0.2)
            else:
                reset_gate = tf.math.sigmoid(h2h_slice[0])
                update_gate = tf.math.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2], alpha=0.2)
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem

            if self._zoneout > 0.0:
                mask = tf.nn.dropout(tf.zeros_like(prev_h), rate=self._zoneout)
                next_h = tf.where(mask, next_h, prev_h)
            outputs.append(next_h)
            prev_h = next_h

        return tf.stack(outputs), next_h