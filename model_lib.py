import numpy as np
import tensorflow as tf

# set trainable variable initialization routines
KERNEL_INIT = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32, uniform=True)
WEIGHT_INIT = tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=True)
BIAS_INIT = tf.constant_initializer(0.0, dtype=tf.float32)

# set minimum value for all variables that are > 0
MIN_POSITIVE_VAL = 1e-6

# log epsilon
LOG_EPSILON = 1e-6

# standard activation
STANDARD_ACTIVATION = tf.nn.elu


def positive_activation_elu(x):

    # apply elu, adjust for elu on (-1, inf), and add minimum value
    return tf.nn.elu(x) + tf.constant(1.0 + MIN_POSITIVE_VAL, dtype=tf.float32)


def positive_activation_soft_plus(x):

    # apply soft-plus and and add minimum value
    return tf.nn.softplus(x) + tf.constant(MIN_POSITIVE_VAL, dtype=tf.float32)


# standard positive activation
STANDARD_POSITIVE_ACTIVATION = positive_activation_soft_plus


def reparameterization_trick(mu, sigma, softmax=False):
    """
    :param mu: mean
    :param sigma: variance vector (implies diagonal covariance)
    :param softmax: optional softmax application to constrain result to the simplex
    :return: z ~ N(mu, sigma^2)
    """
    # if no input, return None
    if mu is None or sigma is None:
        return None

    # apply reparameterization trick for a Gaussian
    z = mu + tf.random_normal(tf.shape(mu)) * sigma

    # apply activation if one was specified
    if softmax:
        z = tf.nn.softmax(z, axis=1)

    return z


def convolution_layer(x, training, kernel_dim, n_out_chan, name):

    # run convolution
    x = tf.layers.conv2d(inputs=x,
                         filters=n_out_chan,
                         kernel_size=kernel_dim,
                         strides=[1, 1],
                         padding='same',
                         activation=STANDARD_ACTIVATION,
                         use_bias=True,
                         kernel_initializer=KERNEL_INIT,
                         bias_initializer=BIAS_INIT,
                         name=name)

    # run max pooling
    x = tf.layers.max_pooling2d(inputs=x,
                                pool_size=3,
                                strides=2,
                                padding='same',
                                name=name)

    # apply batch norm
    # x = tf.layers.batch_normalization(x, training=training)

    return x


def deconvolution_layer(x, kernel_dim, n_out_chan, name):

    # run convolution transpose layers
    x = tf.layers.Conv2DTranspose(filters=n_out_chan,
                                  kernel_size=kernel_dim,
                                  strides=[1, 1],
                                  padding='SAME',
                                  activation=STANDARD_ACTIVATION,
                                  use_bias=True,
                                  kernel_initializer=KERNEL_INIT,
                                  bias_initializer=BIAS_INIT,
                                  name=name)(x)

    # up-sample data
    size = [int(x.shape[1] * 2), int(x.shape[2] * 2)]
    x = tf.image.resize_bilinear(x, size=size, align_corners=None)

    return x


def base_encoder_network(x, training, dropout, enc_arch, y=None):

    with tf.variable_scope('BaseEncoderNetwork') as scope:

        # loop over the convolution layers
        for i in range(len(enc_arch['conv'])):

            # run convolution layer
            x = convolution_layer(x,
                                  training=training,
                                  kernel_dim=enc_arch['conv'][i]['k_size'],
                                  n_out_chan=enc_arch['conv'][i]['out_chan'],
                                  name='conv_layer{:d}'.format(i + 1))

        # flatten features to vector
        x = tf.contrib.layers.flatten(x)

        # append labels if they were provided
        if y is not None:
            x = tf.concat((x, y), axis=-1)

        # loop over fully connected layers
        for i in range(len(enc_arch['full'])):

            # run fully connected layers
            x = tf.layers.dense(inputs=x,
                                units=enc_arch['full'][i],
                                activation=STANDARD_ACTIVATION,
                                use_bias=True,
                                kernel_initializer=WEIGHT_INIT,
                                bias_initializer=BIAS_INIT,
                                name='full_layer{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, rate=dropout)

    return x


def dirichlet_encoder_layer(x, K):

    with tf.variable_scope('DirichletEncodingLayer') as scope:

        # compute alpha
        alpha = tf.layers.dense(inputs=x,
                                units=K,
                                activation=None,
                                use_bias=True,
                                kernel_initializer=WEIGHT_INIT,
                                bias_initializer=BIAS_INIT,
                                name='alpha')

        # apply positive-valued activation
        alpha = STANDARD_POSITIVE_ACTIVATION(alpha)

        return alpha


def gaussian_encoder_layer(x, dim_z):

    with tf.variable_scope('GaussianEncodingLayer') as scope:

        # compute mean
        mu = tf.layers.dense(inputs=x,
                             units=dim_z,
                             activation=None,
                             use_bias=True,
                             kernel_initializer=WEIGHT_INIT,
                             bias_initializer=BIAS_INIT,
                             name='mu_z')

        # compute covariance
        sigma = tf.layers.dense(inputs=x,
                                units=dim_z,
                                activation=None,
                                use_bias=True,
                                kernel_initializer=WEIGHT_INIT,
                                bias_initializer=BIAS_INIT,
                                name='sigma_z')

        # apply positive-valued activation
        sigma = STANDARD_POSITIVE_ACTIVATION(sigma)

        return mu, sigma


def base_decoder_network_dense(x, dropout, dim_mu, dec_arch, covariance_structure, name):

    with tf.variable_scope(name) as scope:

        # loop over fully connected layers
        for i in range(len(dec_arch['full'])):

            # run fully connected layer
            x = tf.layers.dense(inputs=x,
                                units=dec_arch['full'][i],
                                activation=STANDARD_ACTIVATION,
                                use_bias=True,
                                kernel_initializer=WEIGHT_INIT,
                                bias_initializer=BIAS_INIT,
                                name='full_layer{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, rate=dropout)

        # determine final fully connected layer's output dimensions
        n_conv_layers = len(dec_arch['conv'])
        conv_start_dim1 = int(dim_mu[0] / 2 ** n_conv_layers)
        conv_start_dim2 = int(dim_mu[1] / 2 ** n_conv_layers)
        if len(dec_arch['conv']):
            conv_start_chans = dec_arch['conv_start_chans']
        else:
            conv_start_chans = 1
        total_dims = int(conv_start_dim1 * conv_start_dim2 * conv_start_chans)

        # adjust total dimensions as required by architecture
        if len(dec_arch['conv']) == 0 and covariance_structure == 'scalar':
            total_dims += 1
        elif len(dec_arch['conv']) == 0 and covariance_structure == 'diag':
            total_dims *= 2

        # run final fully connected layer
        x = tf.layers.dense(inputs=x,
                            units=total_dims,
                            activation=None,
                            use_bias=True,
                            kernel_initializer=WEIGHT_INIT,
                            bias_initializer=BIAS_INIT,
                            name='full_layer_final')

        # apply non-linearity and drop out only if deconvolution layers will follow
        if len(dec_arch['conv']):
            x = STANDARD_ACTIVATION(x)
            x = tf.layers.dropout(x, rate=dropout)

    return x


def base_decoder_network_deconvolution(x, dropout, dim_x, dec_arch, final_activation, name):

    with tf.variable_scope(name) as scope:

        # determine final fully connected layer's output dimensions
        n_conv_layers = len(dec_arch['conv'])
        conv_start_dim1 = int(dim_x[0] / 2 ** n_conv_layers)
        conv_start_dim2 = int(dim_x[1] / 2 ** n_conv_layers)

        # reshape for convolution layers
        x = tf.reshape(x, shape=(-1, conv_start_dim1, conv_start_dim2, int(dec_arch['conv_start_chans'])))

        # loop over the de-convolution layers
        for i in range(len(dec_arch['conv'])):

            # run de-convolution layer
            x = deconvolution_layer(x,
                                    kernel_dim=dec_arch['conv'][i]['k_size'],
                                    n_out_chan=dec_arch['conv'][i]['out_chan'],
                                    name='deconv_layer{:d}'.format(i + 1))

        # is there a channel mis-match?
        if x.get_shape().as_list()[-1] != dim_x[-1]:

            # run final convolution layer to ensure requisite number of channels
            x = tf.layers.conv2d(inputs=x,
                                 filters=dim_x[-1],
                                 kernel_size=[1, 1],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=KERNEL_INIT,
                                 bias_initializer=BIAS_INIT,
                                 name='deconv_layer_final')

        # apply final activation, if one was provided
        if final_activation is not None:
            x = final_activation(x)

        # ensure dimensions match
        dim_out = x.get_shape().as_list()[1:]
        assert dim_x == dim_out

    return x


def bernoulli_decoder_network(z, dropout, dim_x, dec_arch, k=1):

    with tf.variable_scope('BernoulliDecoder_k{:d}.'.format(k)) as scope:

        # run fully-connected network
        output = base_decoder_network_dense(z, dropout, dim_x, dec_arch, None, 'mu')

        # running deconvolution layers?
        if len(dec_arch['conv']):

            # run deconvolution layers with a final sigmoid to ensure mean is on (0, 1)
            mu_x = base_decoder_network_deconvolution(output, dropout, dim_x, dec_arch, tf.nn.sigmoid, 'mu')

        # no deconvolution layers
        else:

            # apply a sigmoid to ensure mean is on (0, 1) and reshape it to original image dimensions
            mu_x = tf.reshape(tf.nn.sigmoid(output), shape=[-1] + dim_x)

        # no variance for this model
        sigma_x = None

        return mu_x, sigma_x


def gaussian_decoder_network(z, dropout, dim_x, dec_arch, covariance_structure, k=1):

    with tf.variable_scope('GaussianDecoder_k{:d}.'.format(k)) as scope:

        # run fully-connected network
        output = base_decoder_network_dense(z, dropout, dim_x, dec_arch, covariance_structure, 'mu_sigma')

        # running deconvolution layers
        if len(dec_arch['conv']):

            # run deconvolution layers
            mu_x = base_decoder_network_deconvolution(output, dropout, dim_x, dec_arch, None, 'mu')

            # diagonal covariance
            if covariance_structure == 'diag':

                # run deconvolution layers
                sigma_x = base_decoder_network_deconvolution(output, dropout, dim_x, dec_arch, None, 'sigma')

            # scalar covariance
            elif covariance_structure == 'scalar':

                # run fully-connected layer
                sigma_x = tf.layers.dense(inputs=output,
                                          units=1,
                                          activation=None,
                                          use_bias=True,
                                          kernel_initializer=WEIGHT_INIT,
                                          bias_initializer=BIAS_INIT,
                                          name='sigma')

            else:
                assert False, 'not supported'

        # no deconvolution layers
        else:

            # bifurcate the mean and covariance
            mu_x = output[:, :np.prod(dim_x)]
            sigma_x = output[:, np.prod(dim_x):]

            # reshape mean to original image dimensions
            mu_x = tf.reshape(mu_x, shape=[-1] + dim_x)

            # reshape covariance to image dimensions, if using diagonal covariance
            if covariance_structure == 'diag':
                sigma_x = tf.reshape(sigma_x, shape=[-1] + dim_x)

        # apply positive-valued activation
        sigma_x = STANDARD_POSITIVE_ACTIVATION(sigma_x)

        return mu_x, sigma_x


def bernoulli_log_likelihood(x, mu_x):

    with tf.variable_scope('BernoulliLogLikelihood') as scope:

        # flatten input and reconstruction
        x = tf.layers.flatten(x)
        mu_x = tf.layers.flatten(mu_x)

        # compute reconstruction loss: E[ln p(x|z)]
        ll = tf.reduce_sum(x * tf.log(mu_x + 1e-6) + (1 - x) * tf.log(1 - mu_x + 1e-6), axis=1)

    return ll


def gaussian_log_likelihood(x, mu_x, sigma_x, covariance_structure):

    with tf.variable_scope('GaussianLogLikelihood') as scope:

        # flatten input and reconstruction
        x = tf.layers.flatten(x)
        mu_x = tf.layers.flatten(mu_x)

        # compute log determinant portion of reconstruction loss: -E[ln p(x|z)]
        if covariance_structure == 'diag':
            sigma_x = tf.layers.flatten(sigma_x)
            log_det = -0.5 * tf.log(2 * np.pi * sigma_x)
        elif covariance_structure == 'scalar':
            log_det = -0.5 * tf.log(2 * np.pi * sigma_x) * tf.constant(x.get_shape().as_list()[1], dtype=tf.float32)
        else:
            assert False

        # sum over the dimensions
        log_det = tf.reduce_sum(log_det, axis=1)

        # compute log exponential portion of reconstruction loss: -E[ln p(x|z)]
        log_exp = -0.5 * tf.reduce_sum(tf.squared_difference(x, mu_x) / sigma_x, axis=1)

        # combine loss terms
        ll = log_exp + log_det

    return ll


def kl_dirichlet(alpha, alpha_prior):

    # compute convenient terms
    alpha_0 = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    alpha_prior_0 = tf.reduce_sum(alpha_prior, axis=-1, keepdims=True)

    # compute KL(q || p)
    kl = \
        tf.lgamma(alpha_0) - \
        tf.reduce_sum(tf.lgamma(alpha), axis=-1) - \
        tf.lgamma(alpha_prior_0) + \
        tf.reduce_sum(tf.lgamma(alpha_prior)) + \
        tf.reduce_sum((alpha - alpha_prior) * (tf.digamma(alpha) - tf.digamma(alpha_0)), axis=-1)

    return kl


def kl_gaussian(q_mu, q_sigma, p_mu, p_sigma):

    # convert standard deviation to diagonal covariance matrices
    q_sigma2 = tf.square(q_sigma)
    p_sigma2 = tf.square(p_sigma)

    # compute trace(p_sigma2^-1 q_sigma2
    tr = tf.reduce_sum(q_sigma2 / p_sigma2, axis=-1)

    # compute (p_mu - q_mu)^T p_sigma2^(-1) (p_mu - q_mu)
    quad = tf.reduce_sum(tf.squared_difference(p_mu, q_mu) / p_sigma2, axis=-1)

    # compute k
    k = tf.constant(q_sigma.shape.as_list()[-1], dtype=tf.float32)

    # compute log(|p_sigma2| / |q_sigma2|)
    log_det_ratio = tf.reduce_sum(tf.log(p_sigma2) - tf.log(q_sigma2), axis=-1)

    # complete the KL-Divergence
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    kl = 0.5 * (tr + quad - k + log_det_ratio)

    return kl


def dirichlet_prior_laplace_approx(alpha, K):

    # compute mu
    mu = tf.log(alpha) - tf.reduce_sum(tf.log(alpha), axis=-1, keepdims=True) / K

    # compute sigma^2
    sigma2 = (1 - 2 / K) / alpha + tf.reduce_sum(1 / alpha, axis=-1, keepdims=True) / K ** 2

    # convert to standard deviation vector
    sigma = tf.sqrt(sigma2)

    return mu, sigma
