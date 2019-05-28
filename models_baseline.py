import os
from matplotlib import pyplot as plt

# import model library file -- contains common architectural pieces and probabilistic evaluations
from model_lib import *

# import Kumaraswamy-Dirichlet sampler
from kumar_sampler import KumaraswamyDirichletSampler

# plot settings
DPI = 600


class KingmaM2(object):

    def __init__(self, common, dim_z=2):
        """
        :param common: an AutoEncodingCommon object:
        :param dim_z: Normal latent space dimensions
        """
        # save common object
        self.common = common

        # save latent space dimension
        self.dim_z = int(dim_z)

        # duplicate data for MC sampling
        x = tf.tile(self.common.x, [self.common.M] + [1] * len(self.common.dim_x))
        if self.common.x_super is not None and self.common.y_super is not None:
            x_super = tf.tile(self.common.x_super, [self.common.M] + [1] * len(self.common.dim_x))
            y_super = tf.tile(self.common.y_super, [self.common.M, 1])
        else:
            raise Exception('Not supported!')

        # labelled loss
        self.loss_supervised, self.neg_ll_supervised, self.dkl_supervised = self.__loss_labelled(x_super, y_super)
        self.loss_supervised = tf.reduce_mean(self.loss_supervised)
        self.neg_ll_supervised = tf.reduce_mean(self.neg_ll_supervised)
        self.dkl_supervised = tf.reduce_mean(self.dkl_supervised)

        # unlabelled loss
        self.loss_unsupervised = self.__loss_unlabelled(x)
        self.loss_unsupervised = tf.reduce_mean(self.loss_unsupervised)
        self.neg_ll_unsupervised = tf.constant(0, dtype=tf.float32)
        self.dkl_unsupervised = tf.constant(0, dtype=tf.float32)

        # bonus loss
        self.qy_x = tf.reduce_sum(y_super * self.__recognition_network_pi(x_super, training=True), axis=-1)
        self.loss_bonus = tf.reduce_mean(-tf.log(self.qy_x + LOG_EPSILON))

        # compute objective
        self.loss = self.loss_unsupervised + self.loss_supervised + 0.1 * self.loss_bonus

        # configure training operation
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        global_step=self.common.global_step,
                                                        learning_rate=self.common.learning_rate,
                                                        optimizer=self.common.optimizer,
                                                        summaries=['loss', 'gradients'])

        # declare test operations (deactivates neural network regularization methods such as drop out)
        self.alpha_test = self.__recognition_network_pi(self.common.x, training=False)
        self.mu_z_test, _ = self.__recognition_network_z(self.common.x,
                                                         tf.one_hot(tf.argmax(self.alpha_test, axis=-1), self.common.K),
                                                         training=False)
        self.alpha_super_test = self.__recognition_network_pi(self.common.x_super, training=False)
        self.x_recon, _ = self.__generative_network_x(self.mu_z_test, training=False)

        # no latent space plotting
        self.fig_latent = None

    def __recognition_network_pi(self, x, training):

        # if no input, return None
        if x is None:
            return [None] * 3

        with tf.variable_scope('qy_x', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.common.enc_arch)

            # compute pi
            pi = tf.layers.dense(inputs=x,
                                 units=self.common.K,
                                 activation=tf.nn.softmax,
                                 use_bias=True,
                                 kernel_initializer=WEIGHT_INIT,
                                 bias_initializer=BIAS_INIT,
                                 name='pi')

        return pi

    def __recognition_network_z(self, x, y, training):

        # if no input, return None
        if x is None:
            return [None] * 3

        with tf.variable_scope('qz_xy', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.common.enc_arch, y=y)

            # add Gaussian recognition layer
            mu_z, sigma_z = gaussian_encoder_layer(x=x, dim_z=self.dim_z)

        return mu_z, sigma_z

    def __generative_network_x(self, z, training):

        with tf.variable_scope('px_z', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # build decoder network according to generative data distribution family
            if self.common.px_z == 'Gaussian':
                mu_x, sigma_x = gaussian_decoder_network(z,
                                                         dropout=dropout,
                                                         dim_x=self.common.dim_x,
                                                         dec_arch=self.common.dec_arch,
                                                         covariance_structure=self.common.covariance_structure,
                                                         k=1)
            elif self.common.px_z == 'Bernoulli':
                mu_x, sigma_x = bernoulli_decoder_network(z,
                                                          dropout=dropout,
                                                          dim_x=self.common.dim_x,
                                                          dec_arch=self.common.dec_arch,
                                                          k=1)
            else:
                mu_x = sigma_x = None

        return mu_x, sigma_x

    def __generative_network_y(self, x, z, training):

        with tf.variable_scope('py_xz', reuse=tf.AUTO_REUSE) as scope:

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # network input
            x = tf.concat((tf.layers.flatten(x), z), axis=-1)

            # loop over fully connected layers
            for i in range(len(self.common.dec_arch['full'])):
                # run fully connected layer
                x = tf.layers.dense(inputs=x,
                                    units=self.common.dec_arch['full'][i],
                                    activation=STANDARD_ACTIVATION,
                                    use_bias=True,
                                    kernel_initializer=WEIGHT_INIT,
                                    bias_initializer=BIAS_INIT,
                                    name='full_layer{:d}'.format(i + 1))

                # apply drop out
                x = tf.layers.dropout(x, rate=dropout)

            # final layer
            y_hat = tf.layers.dense(inputs=x,
                                    units=self.common.K,
                                    activation=tf.nn.softmax,
                                    use_bias=True,
                                    kernel_initializer=WEIGHT_INIT,
                                    bias_initializer=BIAS_INIT,
                                    name='y_hat')

        return y_hat

    def __loss_labelled(self, x, y):

        # run encoder
        mu_z, sigma_z = self.__recognition_network_z(x, y, training=True)

        # run sampler
        z = reparameterization_trick(mu_z, sigma_z)

        # run decoder
        mu_x, sigma_x = self.__generative_network_x(z, training=True)
        y_hat = self.__generative_network_y(x, z, training=True)

        # compute the negative log likelihood
        ln_px = self.common.log_likelihood_decoder(x, mu_x, sigma_x)
        ln_py = tf.log(tf.reduce_sum(y * y_hat, axis=-1) + LOG_EPSILON)
        neg_ll = -ln_px - ln_py

        # take the KL divergence
        d_kl = kl_gaussian(q_mu=mu_z,
                           q_sigma=sigma_z,
                           p_mu=tf.constant(0, dtype=tf.float32),
                           p_sigma=tf.constant(1, dtype=tf.float32))

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl

    def __loss_unlabelled(self, x):

        # run labelled loss for all possible labels
        y_all = tf.transpose(tf.eye(self.common.K, batch_shape=tf.shape(x)[:1]), [1, 0, 2])
        loss = tf.map_fn(lambda y: self.__loss_labelled(x, y)[0], y_all)

        # run encoder
        pi = self.__recognition_network_pi(x, training=True)

        # take the expectation
        loss = tf.einsum('ik,ki->i', pi, loss)

        # subtract the entropy
        loss += tf.reduce_sum(pi * tf.log(pi + LOG_EPSILON), axis=-1)

        return loss

    def concentration_and_reconstruction(self):

        # return concentration parameters and data reconstruction
        return self.alpha_test, self.x_recon

    def plot_latent_representation(self, sess, epoch=0):

        return


class AutoEncodingSoftmax(object):

    def __init__(self, common):
        """
        :param common: an AutoEncodingCommon object
        """
        # save common object
        self.common = common

        # duplicate data for MC sampling
        x = tf.tile(self.common.x, [self.common.M] + [1] * len(self.common.dim_x))
        if self.common.x_super is not None and self.common.y_super is not None:
            x_super = tf.tile(self.common.x_super, [self.common.M] + [1] * len(self.common.dim_x))
            y_super = tf.tile(self.common.y_super, [self.common.M, 1])
        else:
            x_super = None
            y_super = None

        # construct softmax approximation of Dirichlet priors
        self.mu_pi_prior, self.sigma_pi_prior = dirichlet_prior_laplace_approx(self.common.alpha_prior, self.common.K)

        # declare recognition network training operations
        self.mu_pi, self.sigma_pi = self.__recognition_network(x, training=True)
        self.mu_pi_super, self.sigma_pi_super = self.__recognition_network(x_super, training=True)

        # declare sampler and sample operations
        self.pi = reparameterization_trick(self.mu_pi, self.sigma_pi, softmax=True)
        self.pi_super = reparameterization_trick(self.mu_pi_super, self.sigma_pi_super, softmax=True)

        # declare generative network training operations
        y_all = tf.transpose(tf.eye(self.common.K, batch_shape=tf.shape(self.pi)[:1]), [1, 0, 2])
        if self.common.covariance_structure is None:
            self.mu_x_y = tf.map_fn(lambda y: self.__generative_network(y, training=True)[0], y_all)
            self.sigma_x_y = None
        else:
            self.mu_x_y, self.sigma_x_y = tf.map_fn(lambda y: self.__generative_network(y, training=True),
                                                    y_all,
                                                    dtype=(tf.float32, tf.float32))
        self.mu_x_super, self.sigma_x_super = self.__generative_network(y_super, training=True)

        # unlabelled loss
        self.loss_unsupervised, self.neg_ll_unsupervised, self.dkl_unsupervised =\
            self.__loss_operation(x=x,
                                  mu_x=self.mu_x_y,
                                  sigma_x=self.sigma_x_y,
                                  pi=self.pi,
                                  mu_pi=self.mu_pi,
                                  sigma_pi=self.sigma_pi,
                                  y=None)

        # labelled loss
        self.loss_supervised, self.neg_ll_supervised, self.dkl_supervised =\
            self.__loss_operation(x=x_super,
                                  mu_x=self.mu_x_super,
                                  sigma_x=self.sigma_x_super,
                                  pi=self.pi_super,
                                  mu_pi=self.mu_pi,
                                  sigma_pi=self.sigma_pi,
                                  y=y_super)

        # compute objective
        self.loss = self.loss_unsupervised + self.loss_supervised

        # configure training operation
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        global_step=self.common.global_step,
                                                        learning_rate=self.common.learning_rate,
                                                        optimizer=self.common.optimizer,
                                                        summaries=['loss', 'gradients'])

        # declare test operations (deactivates neural network regularization methods such as drop out)
        self.alpha_test, _ = self.__recognition_network(self.common.x, training=False)
        self.alpha_super_test, _ = self.__recognition_network(self.common.x_super, training=False)
        self.x_recon, _ = self.__generative_network(tf.one_hot(tf.argmax(self.alpha_test, axis=-1), self.common.K),
                                                    training=False)

        # configure latent space plotting
        if self.common.n_channels == 3:
            fig_size = (self.common.K, 1.5)
            self.fig_latent, self.ax_latent = plt.subplots(1, self.common.K, figsize=fig_size)
        else:
            fig_size = (self.common.K, self.common.n_channels * 1.5)
            self.fig_latent, self.ax_latent = plt.subplots(self.common.n_channels, self.common.K, figsize=fig_size)
            if np.ndim(self.ax_latent) < 2:
                self.ax_latent = np.expand_dims(self.ax_latent, axis=0)

        # eliminate those pesky margins
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)

    def __recognition_network(self, x, training):

        # if no input, return None
        if x is None:
            return [None] * 2

        with tf.variable_scope('RecognitionModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.common.enc_arch)

            # add Gaussian recognition layer
            mu_pi, sigma_pi = gaussian_encoder_layer(x=x, dim_z=self.common.K)

        return mu_pi, sigma_pi

    def __generative_network(self, pi_or_y, training):

        # if no input, return None
        if pi_or_y is None:
            return [None] * 2

        with tf.variable_scope('GenerativeModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # build decoder network according to generative data distribution family
            if self.common.px_z == 'Gaussian':
                mu_x, sigma_x = gaussian_decoder_network(pi_or_y,
                                                         dropout=dropout,
                                                         dim_x=self.common.dim_x,
                                                         dec_arch=self.common.dec_arch,
                                                         covariance_structure=self.common.covariance_structure,
                                                         k=1)
            elif self.common.px_z == 'Bernoulli':
                mu_x, sigma_x = bernoulli_decoder_network(pi_or_y,
                                                          dropout=dropout,
                                                          dim_x=self.common.dim_x,
                                                          dec_arch=self.common.dec_arch,
                                                          k=1)
            else:
                mu_x = sigma_x = None

        return mu_x, sigma_x

    def __loss_operation(self, x, mu_x, sigma_x, pi, mu_pi, sigma_pi, y=None):

        # if no input, return None
        if x is None:
            return [tf.constant(0, dtype=tf.float32)] * 3

        # label missing
        if y is None:

            # compute the log likelihood
            if sigma_x is None:
                ln_px = tf.transpose(tf.map_fn(lambda p: self.common.log_likelihood_decoder(x, p, None),
                                               mu_x,
                                               dtype=tf.float32))
            else:
                ln_px = tf.transpose(tf.map_fn(lambda p: self.common.log_likelihood_decoder(x, p[0], p[1]),
                                               (mu_x, sigma_x),
                                               dtype=tf.float32))

            ln_py = tf.log(pi + LOG_EPSILON)
            ll = tf.reduce_logsumexp(ln_px + ln_py, axis=-1)

        # label present
        else:

            # compute the log likelihood
            ln_px = self.common.log_likelihood_decoder(x, mu_x, sigma_x)
            ln_py = tf.reduce_sum(y * tf.log(pi + LOG_EPSILON), axis=1)
            ll = ln_px + ln_py

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # take the mean across the batch of the KL Divergence
        d_kl = tf.reduce_mean(kl_gaussian(q_mu=mu_pi,
                                          q_sigma=sigma_pi,
                                          p_mu=self.mu_pi_prior,
                                          p_sigma=self.sigma_pi_prior))

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl

    def concentration_and_reconstruction(self):

        # return concentration parameters and data reconstruction
        return self.alpha_test, self.x_recon

    def plot_latent_representation(self, sess, epoch=0):

        # generate reconstruction
        x_latent, _ = self.__generative_network(tf.eye(self.common.K), training=False)
        x_latent = sess.run(x_latent)

        # loop over the classes
        for j in range(x_latent.shape[0]):

            # rgb images
            if self.common.n_channels == 3:

                # generate subplots for original data
                sp = self.ax_latent[j]
                sp.cla()
                sp.imshow(x_latent[j], origin='upper', vmin=0, vmax=1)
                sp.set_xticks([])
                sp.set_yticks([])
                sp.set_title('{:d}'.format(j))

            # not rgb images
            else:

                # loop over the channels
                for i in range(self.common.n_channels):

                    # generate subplots for original data
                    sp = self.ax_latent[i, j]
                    sp.cla()
                    sp.imshow(x_latent[j, :, :, i], origin='upper', vmin=0, vmax=1)
                    sp.set_xticks([])
                    sp.set_yticks([])
                    sp.set_title('{:d}'.format(j))

        # save, if we are saving
        if self.common.latent_fig_dir is not None:
            fig_name = os.path.join(self.common.latent_fig_dir, '{:d}.png'.format(epoch))
            self.fig_latent.savefig(fig_name, dpi=DPI)


class AutoEncodingSoftmaxNormal(object):

    def __init__(self, common, dim_z=2):
        """
        :param common: an AutoEncodingCommon object:
        :param dim_z: Normal latent space dimensions
        """
        # save common object
        self.common = common

        # save latent space dimension
        self.dim_z = int(dim_z)

        # duplicate data for MC sampling
        x = tf.tile(self.common.x, [self.common.M] + [1] * len(self.common.dim_x))
        if self.common.x_super is not None and self.common.y_super is not None:
            x_super = tf.tile(self.common.x_super, [self.common.M] + [1] * len(self.common.dim_x))
            y_super = tf.tile(self.common.y_super, [self.common.M, 1])
        else:
            x_super = None
            y_super = None

        # construct softmax approximation of Dirichlet priors
        self.mu_pi_prior, self.sigma_pi_prior = dirichlet_prior_laplace_approx(self.common.alpha_prior, self.common.K)

        # declare recognition network training operations
        self.mu_pi, self.sigma_pi, self.mu_z, self.sigma_z = self.__recognition_network(x, training=True)
        self.mu_pi_super, self.sigma_pi_super, self.mu_z_super, self.sigma_z_super =\
            self.__recognition_network(x_super, training=True)

        # declare sampler and sample operations
        self.pi = reparameterization_trick(self.mu_pi, self.sigma_pi, softmax=True)
        self.pi_super = reparameterization_trick(self.mu_pi_super, self.sigma_pi_super, softmax=True)
        self.z = reparameterization_trick(self.mu_z, self.sigma_z)
        self.z_super = reparameterization_trick(self.mu_z_super, self.sigma_z_super)

        # declare generative network training operations
        y_all = tf.transpose(tf.eye(self.common.K, batch_shape=tf.shape(self.z)[:1]), [1, 0, 2])
        if self.common.covariance_structure is None:
            self.mu_x_y = tf.map_fn(lambda y: self.__generative_network(y, self.z, training=True)[0], y_all)
            self.sigma_x_y = None
        else:
            self.mu_x_y, self.sigma_x_y = tf.map_fn(lambda y: self.__generative_network(y, self.z, training=True),
                                                    y_all,
                                                    dtype=(tf.float32, tf.float32))
        self.mu_x_super, self.sigma_x_super = self.__generative_network(y_super, self.z_super, training=True)

        # unlabelled loss
        self.loss_unsupervised, self.neg_ll_unsupervised, self.dkl_unsupervised =\
            self.__loss_operation(x=x,
                                  mu_x=self.mu_x_y,
                                  sigma_x=self.sigma_x_y,
                                  pi=self.pi,
                                  mu_pi=self.mu_pi,
                                  sigma_pi=self.sigma_pi,
                                  mu_z=self.mu_z,
                                  sigma_z=self.sigma_z,
                                  y=None)
        # labelled loss
        self.loss_supervised, self.neg_ll_supervised, self.dkl_supervised =\
            self.__loss_operation(x=x_super,
                                  mu_x=self.mu_x_super,
                                  sigma_x=self.sigma_x_super,
                                  pi=self.pi_super,
                                  mu_pi=self.mu_pi,
                                  sigma_pi=self.sigma_pi,
                                  mu_z=self.mu_z_super,
                                  sigma_z=self.sigma_z_super,
                                  y=y_super)

        # compute objective
        self.loss = self.loss_unsupervised + self.loss_supervised

        # configure training operation
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        global_step=self.common.global_step,
                                                        learning_rate=self.common.learning_rate,
                                                        optimizer=self.common.optimizer,
                                                        summaries=['loss', 'gradients'])

        # declare test operations (deactivates neural network regularization methods such as drop out)
        self.alpha_test, _, self.mu_x_test, _ = self.__recognition_network(self.common.x, training=False)
        self.alpha_super_test, _, _, _ = self.__recognition_network(self.common.x_super, training=False)
        self.x_recon, _ = self.__generative_network(tf.one_hot(tf.argmax(self.alpha_test, axis=-1), self.common.K),
                                                    self.mu_x_test,
                                                    training=False)

        # configure latent space plotting
        if self.dim_z == 2 and (self.common.n_channels == 1 or self.common.n_channels == 3):
            fig_size = (self.common.K, 4)
            self.fig_latent, self.ax_latent = plt.subplots(2, int(self.common.K / 2 + 0.5), figsize=fig_size)
            self.ax_latent = np.reshape(self.ax_latent, -1)
        elif self.common.n_channels == 3:
            fig_size = (self.common.K, 1.5)
            self.fig_latent, self.ax_latent = plt.subplots(1, self.common.K, figsize=fig_size)
        else:
            fig_size = (self.common.K, self.common.n_channels * 1.5)
            self.fig_latent, self.ax_latent = plt.subplots(self.common.n_channels, self.common.K, figsize=fig_size)
            if np.ndim(self.ax_latent) < 2:
                self.ax_latent = np.expand_dims(self.ax_latent, axis=0)

        # eliminate those pesky margins
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)

    def __recognition_network(self, x, training):

        # if no input, return None
        if x is None:
            return [None] * 4

        with tf.variable_scope('RecognitionModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.common.enc_arch)

            # add Gaussian recognition layer
            mu, sigma = gaussian_encoder_layer(x=x, dim_z=self.common.K + self.dim_z)

            # partition results
            mu_pi = mu[:, :self.common.K]
            sigma_pi = sigma[:, :self.common.K]
            mu_z = mu[:, self.common.K:]
            sigma_z = sigma[:, self.common.K:]

        return mu_pi, sigma_pi, mu_z, sigma_z

    def __generative_network(self, pi_or_y, z, training):

        # if no input, return None
        if pi_or_y is None:
            return [None] * 2

        with tf.variable_scope('GenerativeModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # network input
            network_input = tf.concat((pi_or_y, z), axis=1)

            # build decoder network according to generative data distribution family
            if self.common.px_z == 'Gaussian':
                mu_x, sigma_x = gaussian_decoder_network(network_input,
                                                         dropout=dropout,
                                                         dim_x=self.common.dim_x,
                                                         dec_arch=self.common.dec_arch,
                                                         covariance_structure=self.common.covariance_structure,
                                                         k=1)
            elif self.common.px_z == 'Bernoulli':
                mu_x, sigma_x = bernoulli_decoder_network(network_input,
                                                          dropout=dropout,
                                                          dim_x=self.common.dim_x,
                                                          dec_arch=self.common.dec_arch,
                                                          k=1)
            else:
                mu_x = sigma_x = None

        return mu_x, sigma_x

    def __loss_operation(self, x, mu_x, sigma_x, pi, mu_pi, sigma_pi, mu_z, sigma_z, y=None):

        # if no input, return None
        if x is None:
            return [tf.constant(0, dtype=tf.float32)] * 3

        # unsupervised likelihood
        if y is None:

            # compute the log likelihood
            if sigma_x is None:
                ln_px = tf.transpose(tf.map_fn(lambda p: self.common.log_likelihood_decoder(x, p, None),
                                               mu_x,
                                               dtype=tf.float32))
            else:
                ln_px = tf.transpose(tf.map_fn(lambda p: self.common.log_likelihood_decoder(x, p[0], p[1]),
                                               (mu_x, sigma_x),
                                               dtype=tf.float32))
            ln_py = tf.log(pi + LOG_EPSILON)
            ll = tf.reduce_logsumexp(ln_px + ln_py, axis=-1)

        # supervised likelihood
        else:

            # compute the log likelihood
            ln_px = self.common.log_likelihood_decoder(x, mu_x, sigma_x)
            ln_py = tf.reduce_sum(y * tf.log(pi + LOG_EPSILON), axis=1)
            ll = ln_px + ln_py

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # take the mean across the batch of the KL Divergence
        d_kl = tf.reduce_mean(kl_gaussian(q_mu=mu_pi,
                                          q_sigma=sigma_pi,
                                          p_mu=self.mu_pi_prior,
                                          p_sigma=self.sigma_pi_prior))
        d_kl += tf.reduce_mean(kl_gaussian(q_mu=mu_z,
                                           q_sigma=sigma_z,
                                           p_mu=tf.constant(0, dtype=tf.float32),
                                           p_sigma=tf.constant(1, dtype=tf.float32)))

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl

    def concentration_and_reconstruction(self):

        # return concentration parameters and data reconstruction
        return self.alpha_test, self.x_recon

    def __plot_latent_representation_single(self, sess):

        # generate reconstruction
        x_latent, _ = self.__generative_network(tf.eye(self.common.K),
                                                tf.zeros([self.common.K, self.dim_z]),
                                                training=False)
        x_latent = sess.run(x_latent)

        # loop over the classes
        for j in range(x_latent.shape[0]):

            # rgb images
            if self.common.n_channels == 3:

                # generate subplots for original data
                sp = self.ax_latent[j]
                sp.cla()
                sp.imshow(x_latent[j], origin='upper', vmin=0, vmax=1)
                sp.set_xticks([])
                sp.set_yticks([])
                sp.set_title('{:d}'.format(j))

            # not rgb images
            else:

                # loop over the channels
                for i in range(self.common.n_channels):

                    # generate subplots for original data
                    sp = self.ax_latent[i, j]
                    sp.cla()
                    sp.imshow(x_latent[j, :, :, i], origin='upper', vmin=0, vmax=1)
                    sp.set_xticks([])
                    sp.set_yticks([])
                    sp.set_title('{:d}'.format(j))

    def __plot_latent_representation_2_dims(self, sess):

        # generate the latent encoding test points
        extreme = 1.5  # corresponds to the number of standard deviations away from the prior's mean
        num = 5
        lp = np.linspace(-extreme, extreme, num)
        zx, zy = np.meshgrid(lp, lp, indexing='xy')
        z = np.zeros([0, 2])
        for x in range(num):
            for y in range(num):
                z_new = np.array([[zx[x, y], zy[x, y]]])
                z = np.concatenate((z, z_new), axis=0)

        # copy it K times
        z = tf.constant(np.tile(z, [self.common.K, 1]), dtype=tf.float32)

        # generate latent labels
        y = [tf.tile(tf.expand_dims(tf.one_hot(i, self.common.K), axis=0), [num ** 2, 1]) for i in range(self.common.K)]
        y = tf.reshape(tf.stack(y), [-1, self.common.K])

        # generate reconstruction
        x_latent = sess.run(self.__generative_network(y, z, training=False)[0])

        # loop over the classes
        for k in range(self.common.K):

            # loop over the channels
            x_plot = []
            for c in range(self.common.n_channels):

                # grab all reconstructions for this class/channel
                start = k * num ** 2
                stop = (k + 1) * num ** 2
                x_kc = x_latent[start:stop, :, :, c]

                # turn them into a block
                x_block = []
                for y in range(num):
                    x_row = []
                    for x in range(num):
                        x_row.append(x_kc[y * num + x])
                    x_block.insert(0, x_row)
                x_plot.append(np.block(x_block))

            # generate subplots for original data
            sp = self.ax_latent[k]
            sp.cla()
            sp.imshow(np.squeeze(np.stack(x_plot, axis=2)), origin='upper', vmin=0, vmax=1)
            sp.set_xticks([])
            sp.set_yticks([])

    def plot_latent_representation(self, sess, epoch=0):

        # make the cool plot if we have a 2-D latent Normal vector
        if self.dim_z == 2:
            self.__plot_latent_representation_2_dims(sess)

        # otherwise just make the simple one that receives one-hot labels (for y) and the 0 vector (for z)
        else:
            self.__plot_latent_representation_single(sess)

        # save, if we are saving
        if self.common.latent_fig_dir is not None:
            fig_name = os.path.join(self.common.latent_fig_dir, '{:d}.png'.format(epoch))
            self.fig_latent.savefig(fig_name, dpi=DPI)
