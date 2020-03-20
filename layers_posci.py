# core python
import sys

# third party
import keras.models
import numpy as np
from keras import backend as K
from keras.layers import Layer, Activation, LeakyReLU
from keras.layers import Input, AveragePooling2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Add


from keras.layers import Dense, Lambda, Reshape
import tensorflow as tf
from keras.initializers import RandomUniform,Identity,RandomNormal


def _unet_from_tensor(tensor, filt, kern, acti, trainable=True):
    """
    UNet used in POSCI, adapted from LOUPE package for MRI

    TODO: this is quite rigid right now and hardcoded (# layers, features, etc)
    - use a richer library for this, perhaps neuron
    """

    # start first convolution of UNet
    conv1 = Conv2D(filt, kern, activation = acti, padding = 'same', trainable=trainable)(tensor)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization(trainable=trainable)(conv1)
    conv1 = Conv2D(filt, kern, activation = acti, padding = 'same', trainable=trainable)(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization(trainable=trainable)(conv1)
    
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filt*2, kern, activation = acti, padding = 'same', trainable=trainable)(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization(trainable=trainable)(conv2)
    conv2 = Conv2D(filt*2, kern, activation = acti, padding = 'same', trainable=trainable)(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization(trainable=trainable)(conv2)
    
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(filt*4, kern, activation = acti, padding = 'same', trainable=trainable)(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization(trainable=trainable)(conv3)
    conv3 = Conv2D(filt*4, kern, activation = acti, padding = 'same', trainable=trainable)(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization(trainable=trainable)(conv3)
    
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(filt*8, kern, activation = acti, padding = 'same', trainable=trainable)(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization(trainable=trainable)(conv4)
    conv4 = Conv2D(filt*8, kern, activation = acti, padding = 'same', trainable=trainable)(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization(trainable=trainable)(conv4)
    
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(filt*16, kern, activation = acti, padding = 'same', trainable=trainable)(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization(trainable=trainable)(conv5)
    conv5 = Conv2D(filt*16, kern, activation = acti, padding = 'same', trainable=trainable)(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization(trainable=trainable)(conv5)

    sub1 = UpSampling2D(size=(2, 2))(conv5)
    concat1 = Concatenate(axis=-1)([conv4,sub1])
    
    conv6 = Conv2D(filt*8, kern, activation = acti, padding = 'same', trainable=trainable)(concat1)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization(trainable=trainable)(conv6)
    conv6 = Conv2D(filt*8, kern, activation = acti, padding = 'same', trainable=trainable)(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization(trainable=trainable)(conv6)

    sub2 = UpSampling2D(size=(2, 2))(conv6)
    concat2 = Concatenate(axis=-1)([conv3,sub2])
    
    conv7 = Conv2D(filt*4, kern, activation = acti, padding = 'same', trainable=trainable)(concat2)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization(trainable=trainable)(conv7)
    conv7 = Conv2D(filt*4, kern, activation = acti, padding = 'same', trainable=trainable)(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization(trainable=trainable)(conv7)

    sub3 = UpSampling2D(size=(2, 2))(conv7)
    concat3 = Concatenate(axis=-1)([conv2,sub3])
    
    conv8 = Conv2D(filt*2, kern, activation = acti, padding = 'same', trainable=trainable)(concat3)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization(trainable=trainable)(conv8)
    conv8 = Conv2D(filt*2, kern, activation = acti, padding = 'same', trainable=trainable)(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization(trainable=trainable)(conv8)

    sub4 = UpSampling2D(size=(2, 2))(conv8)
    concat4 = Concatenate(axis=-1)([conv1,sub4])
    
    conv9 = Conv2D(filt, kern, activation = acti, padding = 'same', trainable=trainable)(concat4)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization(trainable=trainable)(conv9)
    conv9 = Conv2D(filt, kern, activation = acti, padding = 'same', trainable=trainable)(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization(trainable=trainable)(conv9)
    conv9 = Conv2D(1, 1, padding = 'same', trainable=trainable)(conv9)
    
    return conv9


def Lambda_intermat(dim):
    indices = []
    for k1 in range(dim):
        for k2 in range(k1):
            indices.append([k1, k2])
    def func(Q):
        M = tf.scatter_nd(indices, Q, (dim, dim))
        return tf.transpose(M) + M
    return func

def Lambda_ising_energy(M, delta):
    def func(x):
        return tf.reduce_sum(0.5 * tf.matmul(x, M) * x  + delta * x, -1)
    return func


# def Lambda_gibbs_layer3(delta, M, dim, const = 10, first=0):
#     def func(inputs):
#         x = inputs[0:dim]
#         u = []
#         for k in range(dim):
#             u.append(tf.random.uniform(shape=tf.shape(x[k])))
#         x_new = x
#         order_list = list(np.random.permutation(dim))
#         for k in order_list:
#             k = k % dim
#             alpha = delta[k]*tf.ones_like(x[k])
#             for j in range(k):
#                 alpha += M[k, j] * x_new[j]
#             for j in range(k+1, dim):
#                 alpha += M[k, j] * x[j] 
#             x_new[k] = tf.tanh(const*(tf.exp(alpha) / (tf.exp(alpha)+tf.exp(-alpha)) - u[k]))
#         return x_new
#     return func


def Lambda_gibbs_layer2(delta, M, dim, const = 10):
    def func(inputs):
        x = inputs[0:dim]
        u = []
        for k in range(dim):
            u.append(tf.random.uniform(shape=tf.shape(x[k])))
        x_new = x
        # order_list = list(np.random.permutation(dim))
        order_list = []
        for k in range(dim):
            order_list.append(k)
        for i in range(dim):
            k = order_list[i] % dim
            updated_indices = order_list[0:i]
            non_updated_indices = order_list[i+1:]
            alpha = delta[k]*tf.ones_like(x[k])
            for j in updated_indices:
                alpha += M[k, j] * x_new[j]
            for j in non_updated_indices:
                alpha += M[k, j] * x[j] 
            x_new[k] = tf.tanh(const*(tf.exp(alpha) / (tf.exp(alpha)+tf.exp(-alpha)) - u[k]))
        return x_new

    return func


class Ising_sampling2(Layer):
    '''
    Gibbs sampling layer
    '''
    def __init__(self, output_dim=16, my_initializer=RandomUniform(minval=-1, maxval=1, seed=None), **kwargs):
        self.initializer = my_initializer
        self.output_dim = output_dim
        super(Ising_sampling2, self).__init__(**kwargs)
    def build(self, input_shape):
        # create trainable weights which describe the Ising model distribution
        self.Q = self.add_weight(name='inter_potential',
                                shape=(self.output_dim*(self.output_dim-1)//2, ),
                                initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None),
                                trainable=True)

        self.delta = self.add_weight(name='self_potential',
                                shape=(self.output_dim,),
                                initializer=self.initializer,
                                trainable=True)


        self.M = Lambda(Lambda_intermat(self.output_dim))(self.Q)
        super(Ising_sampling2, self).build(input_shape)
    def call(self, inputs, n_layers=10, const=10):
        n_batch = tf.shape(inputs)[0]
        self.n_batch = n_batch
        random_seeds = []
        for k in range(self.output_dim):
            random_seeds.append(tf.random.uniform(shape=(n_batch, 1), minval=-1, maxval=1))
        x = []
        for k in range(self.output_dim):
            x.append(K.sign(random_seeds[k]))
            # x.append(K.tanh(const * random_seeds[k]))
        for k in range(1, n_layers+1):
            x = Lambda(Lambda_gibbs_layer2(self.delta, self.M, self.output_dim, const))(x)
        z = K.concatenate(x, -1)
        energy = Lambda(Lambda_ising_energy(self.M, self.delta))(z)
        return [z, energy]
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], )]



