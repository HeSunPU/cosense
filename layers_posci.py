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
#from flow_generator_helpers import FCResnet, Split, Concat, Coupling, Permute2


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


def Lambda_gibbs_layer2(delta, M, dim, const = 10, order_flag='forward'):
    def func(inputs):
        x = inputs[0:dim]
        u = []
        for k in range(dim):
            u.append(tf.random.uniform(shape=tf.shape(x[k])))
        x_new = x
        if order_flag == 'forward':
            order_list = list(np.arange(dim))
        elif order_flag == 'reverse':
            order_list = list(np.arange(dim)[::-1])
        elif order_flag == 'random':
            order_list = list(np.random.permutation(dim))
        else:
            print('''The order flag should be 'forward', 'reverse' or 'random'!''')
        # order_list = list(np.random.permutation(dim))
        
        # order_list = []
        # for k in range(dim):
        #     order_list.append(k)
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


@tf.custom_gradient
def straight_through_estimator_tanh(x):
    y = tf.math.sign(tf.math.sign(x) - 1e-6)
    temperature = 1
    y2 = tf.math.tanh(x/temperature)
    def grad(dy):
        return dy * (1 - tf.math.square(y2))
    return y, grad

class STE_tanh_layer(Layer):
    def __init__(self):
        super(STE_tanh_layer, self).__init__()
    def call(self, x):
        return straight_through_estimator_tanh(x)

def Lambda_inverse_tanh(x):
    y = 0.5 * (tf.math.log(1+x) - tf.math.log(1-x))
    return y


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
        y = Lambda_inverse_tanh(z)
        x = straight_through_estimator_tanh(y)
        energy = Lambda(Lambda_ising_energy(self.M, self.delta))(x)
        return [x, energy]
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], )]


class Ising_sampling(Layer):
    '''
    Hybird/Hamilton Monte Carlo sampling layer
    '''
    def __init__(self, output_dim=16, my_initializer=RandomUniform(minval=-1, maxval=1, seed=None), **kwargs):
        self.initializer = my_initializer
        self.output_dim = output_dim
        super(Ising_sampling, self).__init__(**kwargs)
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
        super(Ising_sampling, self).build(input_shape)
    def call(self, inputs, n_layers=3, const=1, xi=1, L=10):
        n_batch = tf.shape(inputs)[0]
        self.n_batch = n_batch

        # update the Ising sample using hybrid/Hamiltonian Monte Carlo
        y = tf.random.normal(shape=(n_batch, self.output_dim))
        x = tf.math.tanh(const * y)
        for k in range(n_layers):
            vel = tf.random.normal(shape=(n_batch, self.output_dim))
            for i in range(L):
                vel_half = vel + 0.5 * np.sqrt(xi) * const * (self.delta + tf.matmul(x, self.M)) * (1 - tf.math.square(x))
                y += np.sqrt(xi) * vel_half
                x = tf.math.tanh(const * y)
                vel = vel_half + 0.5 * np.sqrt(xi) * const * (self.delta + tf.matmul(x, self.M)) * (1 - tf.math.square(x))
        x = straight_through_estimator_tanh(const * y)
        energy = Lambda(Lambda_ising_energy(self.M, self.delta))(x)
        return [x, energy]
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], )]


class Site_mask_prob(Layer):
    '''
    Site mask prob
    '''
    def __init__(self, obs_prob, **kwargs):
        self.obs_prob = obs_prob
        super(Site_mask_prob, self).__init__(**kwargs)
    def build(self, input_shape):
        # create trainable weights which describe the Ising model distribution
        super(Site_mask_prob, self).build(input_shape)
    def call(self, inputs):
        mask_shape = tf.shape(inputs)
        random_seeds = tf.random.uniform(shape=mask_shape, minval=0, maxval=1)
        prob_mask = 2 * (tf.math.sign(self.obs_prob - random_seeds) + 1)
        return prob_mask * inputs
    def compute_output_shape(self, input_shape):
        return input_shape


@tf.custom_gradient
def straight_through_estimator(x):
    y = 0.5 * (tf.math.sign(tf.math.sign(x) - 1e-6) + 1.0)
    temperature = 1
    y2 = tf.math.sigmoid(x/temperature)
    def grad(dy):
        return dy * (1 - y2) * y2
    return y, grad

class STE_layer(Layer):
    def __init__(self, **kwargs):
        super(STE_layer, self).__init__(**kwargs)
    def call(self, x):
        return straight_through_estimator(x)


class Gaussian_sampling(Layer):
    '''
    Gaussian sampling layer
    '''
    def __init__(self, output_dim=16, my_initializer=keras.initializers.Zeros(), **kwargs):
        self.initializer = my_initializer
        self.output_dim = output_dim
        super(Gaussian_sampling, self).__init__(**kwargs)
    def build(self, input_shape):
        # create trainable weights which describe the Ising model distribution
        W0 = np.random.normal(loc=0, scale=1e-3, size=(self.output_dim, self.output_dim))
        # W0 = W0 / np.abs(np.linalg.det(W0))**(1/self.output_dim)
        self.W0 = self.add_weight(name='std',
                                shape=(self.output_dim, self.output_dim),
                                initializer=keras.initializers.Constant(W0),
                                trainable=True)

        self.b = self.add_weight(name='mean',
                                shape=(self.output_dim,),
                                initializer=self.initializer,
                                trainable=True)

        self.W = tf.eye(self.output_dim) + self.W0


        super(Gaussian_sampling, self).build(input_shape)
    def call(self, inputs):
        n_batch = tf.shape(inputs)[0]
        self.n_batch = n_batch
        # update the Ising sample using hybrid/Hamiltonian Monte Carlo
        y = tf.random.normal(shape=(n_batch, self.output_dim))
        x = tf.matmul(y, self.W) + self.b
        logdet = tf.linalg.logdet(self.W) * tf.ones((n_batch, ), dtype=tf.float32)
        return [x, logdet]
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], )]



def CouplingBlock(x1, x2, depth=2):
    logdet_list = []
    resnet_list = []

    dim1 = int(x1.get_shape()[-1])
    dim2 = int(x2.get_shape()[-1])
    for k in range(depth):
        # affine coupling (x1, x2) ========> (x1, s(x1)*x2+t(x1))
        res = FCResnet(dim1, dim2, n_res_blocks=3, n_hid=dim1)
        log_scale_shift = res(x1)

        coupling = Coupling(shift_only=False)

        x1, x2, logdet = coupling([x1, x2, log_scale_shift])

        logdet_list.append(logdet)
        resnet_list.append(res)

        # affine coupling (x1, x2) ========> (s(x2)*x1+t(x2), x2)
        res = FCResnet(dim2, dim1, n_res_blocks=3, n_hid=dim2)
        log_scale_shift = res(x2)

        coupling = Coupling(shift_only=False)

        x2, x1, logdet = coupling([x2, x1, log_scale_shift])

        logdet_list.append(logdet)
        resnet_list.append(res)

    logdet = keras.layers.Add()(logdet_list)

    return x1, x2, logdet, resnet_list



def CouplingBlockInverse(x1, x2, resnet_layers):
    depth=len(resnet_layers)//2
    logdet_list = []

    for k in range(depth):
        # inverse affine coupling (s(x2)*x1+t(x2), x2) ========> (x1, x2)
        res = resnet_layers[-1-k*2]
        log_scale_shift = res(x2)

        coupling_inverse = Coupling(shift_only=False).inverse()
        x2, x1, logdet = coupling_inverse([x2, x1, log_scale_shift])

        logdet_list.append(logdet)

        # inverse affine coupling (x1, s(x1)*x2+t(x1)) ========> (x1, x2)
        res = resnet_layers[-2-k*2]
        log_scale_shift = res(x1)

        coupling_inverse = Coupling(shift_only=False).inverse()

        x1, x2, logdet = coupling_inverse([x1, x2, log_scale_shift])

        logdet_list.append(logdet)

    logdet = keras.layers.Add()(logdet_list)

    return x1, x2, logdet



def realnvp_encoder(output_dim=16, n_coupling=5):
    x_in = Input(shape=(output_dim, ), name='input_sample')
    x = x_in

    logdet_list = []
    resnet_list = []
    permute_list = []

    for k in range(n_coupling):
        # split the samples into two part
        split = Split()
        x1, x2 = split(x)

        # affine coupling
        x1, x2, logdet, resnet_layers = CouplingBlock(x1, x2, depth=2)
        logdet_list.append(logdet)
        resnet_list.append(resnet_layers)

        # concate the splitted part
        concat = Concat()
        x = concat([x1, x2])

        # permute the dimensions
        permute = Permute2(mode='random')
        x = permute(x)
        permute_list.append(permute)

    z = x
    logdet = keras.layers.Add(name='logdet')(logdet_list)
    inputs = x_in
    outputs = [z, logdet]

    return keras.models.Model(inputs, outputs), resnet_list, permute_list


def realnvp_decoder(resnet_list, permute_list, output_dim=16, n_coupling=5):
    z = Input(shape=(output_dim, ))

    x = z

    logdet_list = []
    for k in range(n_coupling-1, -1, -1):
        # inverse permute
        permute = permute_list[k]
        x = permute.inverse()(x)

        # inverse concat
        concat = Concat()
        x1, x2 = concat.inverse()(x)

        # inverse coupling
        resnet_layers = resnet_list[k]
        x1, x2, logdet = CouplingBlockInverse(x1, x2, resnet_layers)
        logdet_list.append(logdet)

        # inverse split
        split = Split()
        x = split.inverse()([x1, x2])

    logdet = keras.layers.Add(name='logdet')(logdet_list)

    inputs = z
    outputs = [x, logdet]

    return keras.models.Model(inputs, outputs)



def Lambda_Gaussian(output_dim=16):
    def func(x):
        x_shape = tf.shape(x)
        n_batch = x_shape[0]
        random_seeds = tf.random.normal((n_batch, output_dim), dtype=tf.float32)
        return random_seeds
    return func


def Lambda_logdet_sigmoid(slope=1):
    def func(x):
        output_mask = tf.math.sigmoid(slope*x)
        derivative = slope * output_mask * (1 - output_mask)
        return tf.reduce_sum(tf.math.log(derivative + 1e-8), -1)
    return func






