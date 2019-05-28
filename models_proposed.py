import os
import shutil
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# import model library file -- contains common architectural pieces and probabilistic evaluations
from model_lib import *

# import Kumaraswamy-Dirichlet sampler
from kumar_sampler import KumaraswamyDirichletSampler

# plot settings
DPI = 600


class AutoEncodingCommon(object):

    def __init__(self,
                 dim_x,
                 K,
                 enc_arch,
                 dec_arch,
                 learn_rate,
                 px_z,
                 covariance_structure,
                 dropout_rate=0,
                 num_classes=None,
                 save_dir=None):
        """
        :param dim_x: list containing input shape [length, width, channels]
        :param K: number of latent clusters

        :param enc_arch: encoder architecture
        :param dec_arch: decoder architecture
        :param px_z: data generating distribution family {Bernoulli, Gaussian}
        :param covariance_structure: {scalar, diag}
        :param dropout_rate: drop out rate
        :param num_classes: number of real classes (defaults to K if None)
        :param save_dir: save directory
        """
        # initialize placeholders
        self.x = None
        self.x_super = None
        self.y_super = None
        self.M = tf.placeholder(dtype=tf.int32)

        # initialize mode (will either be supervised or semi-supervised once link_inputs is called)
        self.mode = None

        # save data dimensions
        assert isinstance(dim_x, list) and len(dim_x) == 3
        self.dim_x = dim_x
        self.n_channels = self.dim_x[-1]

        # save cluster priors
        self.K = K
        self.alpha_prior = tf.constant(np.ones(self.K) / self.K, dtype=tf.float32)

        # save number of real classes
        if num_classes is None:
            num_classes = K
        self.num_classes = num_classes

        # save encoder/decoder architectures
        self.enc_arch = enc_arch
        self.dec_arch = dec_arch

        # Monte-Carlo samples
        self.monte_carlo_samples = 5

        # save output type
        assert px_z == 'Gaussian' or px_z == 'Bernoulli'
        self.px_z = px_z

        # save variance type
        if px_z == 'Gaussian':
            assert covariance_structure == 'scalar' or covariance_structure == 'diag'
        elif px_z == 'Bernoulli':
            assert covariance_structure is None
        self.covariance_structure = covariance_structure

        # regularization hyper-parameters
        self.dropout_prob = tf.constant(dropout_rate, tf.float32)

        # configure training
        self.learning_rate = learn_rate
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # configure logging
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.save_dir = save_dir

        # configure reconstruction plotting
        if self.n_channels == 3:
            self.fig_recon, self.ax_recon = plt.subplots(3, self.num_classes, figsize=(8, 2.75))
        else:
            self.fig_recon, self.ax_recon = plt.subplots(2 * self.n_channels + 1, self.num_classes, figsize=(8, 2.75))
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.95, top=1, wspace=0.05, hspace=0)

        # set up plot directories if a save director was provided
        if self.save_dir is not None:

            # set the base directory
            self.recon_fig_dir = os.path.join(self.save_dir, 'reconstruction_figures')
            self.latent_fig_dir = os.path.join(self.save_dir, 'latent_space_figures')

            # remove those directories if they already exist
            if os.path.exists(self.recon_fig_dir):
                shutil.rmtree(self.recon_fig_dir)
            if os.path.exists(self.latent_fig_dir):
                shutil.rmtree(self.latent_fig_dir)

            # make fresh directories
            os.mkdir(self.recon_fig_dir)
            os.mkdir(self.latent_fig_dir)

        # otherwise, set them to None
        else:
            self.recon_fig_dir = None
            self.latent_fig_dir = None

    def link_inputs(self, x, x_super=None, y_super=None):
        """
        :param x: unlabelled input data of shape [batch, length, width, channels]
        :param x_super: labelled input data of shape [batch, length, width, channels]
        :param y_super: labels of shape [batch] corresponding to x_super
        :return: None
        """
        # save placeholders and training mode
        self.x = x
        self.mode = 'unsupervised'
        if x_super is not None and y_super is not None:
            self.x_super = x_super
            self.y_super = tf.one_hot(y_super, self.K)
            self.mode = 'semi-supervised'

    def log_likelihood_decoder(self, x, mu_x, sigma_x):

        # Gaussian output treatment
        if self.px_z == 'Gaussian':
            ln_p_x = gaussian_log_likelihood(x, mu_x, sigma_x ** 2, self.covariance_structure)

        # Bernoulli output treatment
        elif self.px_z == 'Bernoulli':
            ln_p_x = bernoulli_log_likelihood(x, mu_x)

        else:
            raise Exception('Unsupported likelihood function!')

        return ln_p_x

    def plot_random_reconstruction(self, x, x_recon, alpha, epoch=0):

        # get min/max of alpha parameter
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)

        # loop over the classes
        for i in range(self.num_classes):

            # rgb images
            if self.n_channels == 3:

                # generate subplots for original data
                sp = self.ax_recon[0, i]
                sp.cla()
                sp.imshow(x[i], origin='upper', vmin=0, vmax=1)
                sp.set_xticks([])
                sp.set_yticks([])
                if i == 0:
                    sp.set_ylabel('Original')

                # generate subplots for reconstructed data
                sp = self.ax_recon[1, i]
                sp.cla()
                sp.imshow(x_recon[i], origin='upper', vmin=0, vmax=1)
                sp.set_xticks([])
                sp.set_yticks([])
                if i == 0:
                    sp.set_ylabel('Generated')

                # plot alpha parameters
                sp = self.ax_recon[2, i]
                sp.cla()
                markerline, stemlines, _ = sp.stem(alpha[i], markerfmt='k.', linefmt='--', bottom=0)
                markerline.set_zorder(10)
                for s in stemlines:
                    plt.setp(s, alpha=0.75)
                sp.set_xticks([])
                sp.set_yticks([])
                sp.set_ylim([alpha_min, alpha_max])
                if i == 0:
                    sp.set_ylabel('Parameters')
                if i == self.K - 1:
                    sp.set_yticks([alpha_min, alpha_min + (alpha_max - alpha_min) / 2, alpha_max])
                    sp.yaxis.tick_right()
                    sp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # not rgb images
            else:

                # loop over the channels
                for j in range(self.n_channels):

                    # generate subplots for original data
                    sp = self.ax_recon[2 * j, i]
                    sp.cla()
                    sp.imshow(x[i, :, :, j], origin='upper', vmin=0, vmax=1)
                    sp.set_xticks([])
                    sp.set_yticks([])
                    if i == 0:
                        sp.set_ylabel('Original')

                    # generate subplots for reconstructed data
                    sp = self.ax_recon[2 * j + 1, i]
                    sp.cla()
                    sp.imshow(x_recon[i, :, :, j], origin='upper', vmin=0, vmax=1)
                    sp.set_xticks([])
                    sp.set_yticks([])
                    if i == 0:
                        sp.set_ylabel('Generated')

                    # plot alpha parameters
                    sp = self.ax_recon[2 * j + 2, i]
                    sp.cla()
                    markerline, stemlines, _ = sp.stem(alpha[i], markerfmt='k.', linefmt='--', bottom=0)
                    markerline.set_zorder(10)
                    for s in stemlines:
                        plt.setp(s, alpha=0.75)
                    sp.set_xticks([])
                    sp.set_yticks([])
                    sp.set_ylim([alpha_min, alpha_max])
                    if i == 0:
                        sp.set_ylabel('Parameters')
                    if i == self.K - 1:
                        sp.set_yticks([alpha_min, alpha_min + (alpha_max - alpha_min) / 2, alpha_max])
                        sp.yaxis.tick_right()
                        sp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # save, if we are saving
        if self.recon_fig_dir is not None:
            fig_name = os.path.join(self.recon_fig_dir, '{:d}.png'.format(epoch))
            self.fig_recon.savefig(fig_name, dpi=DPI)


class AutoEncodingDirichlet(object):

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

        # declare recognition network training operations
        self.alpha = self.__recognition_network(x, training=True)
        self.alpha_super = self.__recognition_network(x_super, training=True)

        # declare sampler and sample operations
        self.sampler = KumaraswamyDirichletSampler(K=self.common.K, taylor_order=5)
        self.pi, self.i_perm = self.sampler.sample(self.alpha)
        self.pi_super, self.i_perm_super = self.sampler.sample(self.alpha_super)

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
                                  alpha=self.alpha,
                                  i_perm=self.i_perm,
                                  y=None)

        # labelled loss
        self.loss_supervised, self.neg_ll_supervised, self.dkl_supervised =\
            self.__loss_operation(x=x_super,
                                  mu_x=self.mu_x_super,
                                  sigma_x=self.sigma_x_super,
                                  pi=self.pi_super,
                                  alpha=self.alpha_super,
                                  i_perm=self.i_perm_super,
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
        self.alpha_test = self.__recognition_network(self.common.x, training=False)
        self.alpha_super_test = self.__recognition_network(self.common.x_super, training=False)
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
            return None

        with tf.variable_scope('RecognitionModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.common.enc_arch)

            # add Dirichlet recognition layer
            alpha = dirichlet_encoder_layer(x=x, K=self.common.K)

        return alpha

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

    def __loss_operation(self, x, mu_x, sigma_x, pi, alpha, i_perm, y=None):

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
        d_kl = tf.reduce_mean(self.sampler.kl_divergence(alpha, self.common.alpha_prior, wrt='BetaSticks', i_perm=i_perm))

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


class AutoEncodingDirichletNormal(object):

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

        # declare recognition network training operations
        self.alpha, self.mu_z, self.sigma_z = self.__recognition_network(x, training=True)
        self.alpha_super, self.mu_z_super, self.sigma_z_super = self.__recognition_network(x_super, training=True)

        # declare sampler and sample operations
        self.sampler = KumaraswamyDirichletSampler(K=self.common.K, taylor_order=5)
        self.pi, self.i_perm = self.sampler.sample(self.alpha)
        self.pi_super, self.i_perm_super = self.sampler.sample(self.alpha_super)
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
                                  alpha=self.alpha,
                                  i_perm=self.i_perm,
                                  mu_z=self.mu_z,
                                  sigma_z=self.sigma_z,
                                  y=None)
        # labelled loss
        self.loss_supervised, self.neg_ll_supervised, self.dkl_supervised =\
            self.__loss_operation(x=x_super,
                                  mu_x=self.mu_x_super,
                                  sigma_x=self.sigma_x_super,
                                  pi=self.pi_super,
                                  alpha=self.alpha_super,
                                  i_perm=self.i_perm_super,
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
        self.alpha_test, self.mu_x_test, _ = self.__recognition_network(self.common.x, training=False)
        self.alpha_super_test, _, _ = self.__recognition_network(self.common.x_super, training=False)
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
            return [None] * 3

        with tf.variable_scope('RecognitionModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.common.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.common.enc_arch)

            # add Dirichlet recognition layer
            alpha = dirichlet_encoder_layer(x=x, K=self.common.K)

            # add Gaussian recognition layer
            mu_z, sigma_z = gaussian_encoder_layer(x=x, dim_z=self.dim_z)

        return alpha, mu_z, sigma_z

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

    def __loss_operation(self, x, mu_x, sigma_x, pi, alpha, i_perm, mu_z, sigma_z, y=None):

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
        d_kl = tf.reduce_mean(self.sampler.kl_divergence(alpha, self.common.alpha_prior, wrt='BetaSticks', i_perm=i_perm))
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
