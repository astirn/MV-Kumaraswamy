import os
import shutil
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# import model library file -- contains common architectural pieces and probabilistic evaluations
from model_lib import *

# import Kumaraswamy-Dirichlet sampler
from mv_kumaraswamy_sampler import KumaraswamyStickBreakingProcess

# plot settings
DPI = 600


class VariationalAutoEncoder(object):

    def __init__(self,
                 dim_x,
                 num_classes,
                 dim_z,
                 K,
                 enc_arch,
                 dec_arch,
                 learn_rate,
                 px_z,
                 covariance_structure,
                 dropout_rate=0,
                 save_dir=None,
                 save_plots=False):
        """
        :param dim_x: list containing input shape [length, width, channels]
        :param num_classes: number of real classes
        :param dim_z: number of normally distributed latent dimensions
        :param K: number of latent clusters
        :param enc_arch: encoder architecture
        :param dec_arch: decoder architecture
        :param px_z: data likelihood family {Bernoulli, Gaussian}
        :param covariance_structure: {scalar, diag}
        :param dropout_rate: drop out rate
        :param save_dir: save directory
        :param save_plots: whether to save plots
        """
        # initialize placeholders
        self.M = tf.placeholder(dtype=tf.int32)

        # initialize task type (will either be supervised or semi-supervised once set_task_type is called)
        self.task_type = None

        # save observable variable dimensions
        assert isinstance(dim_x, list) and len(dim_x) == 3
        self.dim_x = dim_x
        self.n_channels = self.dim_x[-1]

        # save latent variable dimensions
        self.dim_z = dim_z
        self.K = K

        # save cluster priors
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
        assert px_z in {'Gaussian', 'Bernoulli'}
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

        # configure latent space plotting
        self.x_latent = None
        if self.dim_z == 2 and (self.n_channels == 1 or self.n_channels == 3):
            fig_size = (self.K, 4)
            self.fig_latent, self.ax_latent = plt.subplots(2, int(self.K / 2 + 0.5), figsize=fig_size)
            self.ax_latent = np.reshape(self.ax_latent, -1)
        elif self.n_channels == 3:
            fig_size = (self.K, 1.5)
            self.fig_latent, self.ax_latent = plt.subplots(1, self.K, figsize=fig_size)
        else:
            fig_size = (self.K, self.n_channels * 1.5)
            self.fig_latent, self.ax_latent = plt.subplots(self.n_channels, self.K, figsize=fig_size)
            if np.ndim(self.ax_latent) < 2:
                self.ax_latent = np.expand_dims(self.ax_latent, axis=0)

        # eliminate those pesky margins
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)

        # set up plot directories if a save director was provided
        if self.save_dir is not None and save_plots:

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

    def set_task_type(self, x_obs, y_obs):
        """
        :param x_obs: labelled input data of shape [batch, length, width, channels]
        :param y_obs: labels of shape [batch] corresponding to x_obs
        :return: None
        """
        # determine and save training task type
        if x_obs is None and y_obs is None:
            self.task_type = 'unsupervised'
        elif x_obs is not None and y_obs is not None:
            self.task_type = 'semi-supervised'
        else:
            raise Exception('undetermined training task')

    def duplicate_training_data(self, x_lat, x_obs=None, y_obs=None):
        """
        :param x_lat: unlabelled input data of shape [batch, length, width, channels]
        :param x_obs: labelled input data of shape [batch, length, width, channels]
        :param y_obs: labels of shape [batch] corresponding to x_obs
        :return: None
        """
        # duplicate data for MC sampling
        x_lat = tf.tile(x_lat, [self.M] + [1] * len(self.dim_x))
        if x_obs is not None and y_obs is not None:
            x_obs = tf.tile(x_obs, [self.M] + [1] * len(self.dim_x))
            y_obs = tf.tile(tf.one_hot(y_obs, self.num_classes), [self.M, 1])
        else:
            x_obs = y_obs = None

        return x_lat, x_obs, y_obs

    def log_likelihood_decoder(self, x, mu_x, sigma_x):
        """
        :param x: input image
        :param mu_x: generative network's mean parameter
        :param sigma_x: generative network's covariance parameter (None for Bernoulli)
        :return: log likelihood
        """
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
        """
        :param x: original data of shape [num classes, length, width, channels]
        :param x_recon: reconstructed data of shape [num classes, length, width, channels]
        :param alpha: concentration parameter of shape [num classes, K]
        :param epoch: current epoch number for plot text
        :return: None
        """
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

    def latent_representation(self, generative_network):
        """
        :param generative_network: model's generative network function
        :return: latent representation tensor according to dim(z)
        """
        # for dim(z) = 2, generate a manifold
        if self.dim_z == 2:

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
            z = tf.constant(np.tile(z, [self.K, 1]), dtype=tf.float32)

            # generate latent labels
            y = [tf.tile(tf.expand_dims(tf.one_hot(i, self.K), axis=0), [num ** 2, 1]) for i in range(self.K)]
            y = tf.reshape(tf.stack(y), [-1, self.K])

            # generate reconstruction
            x_latent = generative_network(y, z, training=False)[0]

        # for dim(z) != 2, use the prior
        else:

            # generate reconstruction
            x_latent = generative_network(tf.eye(self.K), tf.zeros([self.K, self.dim_z]), training=False)[0]

        return x_latent

    def __plot_latent_representation_single(self, sess):
        """
        :param sess: TensorFlow session
        :return: None
        """
        # compute latent space representation
        x_latent = sess.run(self.x_latent)

        # loop over the classes
        for j in range(x_latent.shape[0]):

            # rgb images
            if self.n_channels == 3:

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
                for i in range(self.n_channels):

                    # generate subplots for original data
                    sp = self.ax_latent[i, j]
                    sp.cla()
                    sp.imshow(x_latent[j, :, :, i], origin='upper', vmin=0, vmax=1)
                    sp.set_xticks([])
                    sp.set_yticks([])
                    sp.set_title('{:d}'.format(j))

    def __plot_latent_representation_2_dims(self, sess):
        """
        :param sess: TensorFlow session
        :return: None
        """
        # compute latent space representation
        x_latent = sess.run(self.x_latent)
        num = int((x_latent.shape[0] / self.K) ** 0.5)

        # loop over the classes
        for k in range(self.K):

            # loop over the channels
            x_plot = []
            for c in range(self.n_channels):

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
        """
        :param sess: TensorFlow session
        :param epoch: current epoch number for plot text
        :return: None
        """
        # return for models without a latent representation implementation
        if self.x_latent is None:
            return

        # make the cool plot if we have a 2-D latent Normal vector
        if self.dim_z == 2:
            self.__plot_latent_representation_2_dims(sess)

        # otherwise just make the simple one that receives one-hot labels (for y) and the 0 vector (for z)
        else:
            self.__plot_latent_representation_single(sess)

        # save, if we are saving
        if self.latent_fig_dir is not None:
            fig_name = os.path.join(self.latent_fig_dir, '{:d}.png'.format(epoch))
            self.fig_latent.savefig(fig_name, dpi=DPI)


class AutoEncodingKumaraswamy(VariationalAutoEncoder):

    def __init__(self, x_lat, x_obs=None, y_obs=None, use_rand_perm=True, **kwargs):
        """
        :param x_lat: image data with latent labels
        :param x_obs: image data with observed labels
        :param y_obs: labels for x_obs
        :param kwargs: configuration dictionary for the base VAE class
        """
        # initialize base class
        VariationalAutoEncoder.__init__(self, **kwargs)

        # set the training task type (i.e. unsupervised or semi-supervised)
        self.set_task_type(x_obs=x_obs, y_obs=y_obs)

        # duplicate data for MC sampling
        x_lat, x_obs, y_obs = self.duplicate_training_data(x_lat, x_obs, y_obs)

        # declare MV-Kumaraswamy sampler
        self.use_rand_perm = use_rand_perm
        self.sampler = KumaraswamyStickBreakingProcess(dkl_taylor_order=5)

        # labelled loss
        self.loss_labelled, self.neg_ll_labelled, self.dkl_labelled = self.__loss_labelled(x_obs, y_obs)

        # unlabelled loss
        self.loss_unlabelled, self.neg_ll_unlabelled, self.dkl_unlabelled = self.__loss_unlabelled(x_lat)

        # compute re-weighted objective
        self.loss = self.loss_unlabelled + self.loss_labelled

        # configure training operation
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        global_step=self.global_step,
                                                        learning_rate=self.learning_rate,
                                                        optimizer=self.optimizer,
                                                        summaries=['loss', 'gradients'])

        # declare test operations (deactivates neural network regularization methods such as drop out)
        self.alpha_lat_test, self.mu_x_test, _ = self.__recognition_network(x_lat, training=False)
        self.alpha_obs_test, _, _ = self.__recognition_network(x_obs, training=False)
        self.x_recon, _ = self.__generative_network(tf.one_hot(tf.argmax(self.alpha_lat_test, axis=-1), self.K),
                                                    self.mu_x_test,
                                                    training=False)
        self.x_latent = self.latent_representation(self.__generative_network)

    def __recognition_network(self, x, training):
        """
        :param x: image data
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: posterior approximation parameters
        """
        # if no input, return None
        if x is None:
            return [None] * 3

        with tf.variable_scope('RecognitionModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.enc_arch)

            # add Dirichlet recognition layer
            alpha = dirichlet_encoder_layer(x=x, K=self.K)

            # add Gaussian recognition layer if needed
            if self.dim_z > 0:
                mu_z, sigma_z = gaussian_encoder_layer(x=x, dim_z=self.dim_z)
            else:
                mu_z = sigma_z = None

        return alpha, mu_z, sigma_z

    def __sample_posterior(self, alpha, mu_z, sigma_z):
        """
        :param alpha: MV-Kumaraswamy concentration parameter
        :param mu_z: Gaussian mean
        :param sigma_z: Gaussian covariance
        :return: o~Uniform(Permutations({1,...,K}), pi~StickBreak(Kumaraswamy, alpha, o), z~N(mu_z, sigma_z)
        """
        # sample o uniformly from all permutations and pi from the Kumaraswamy stick-breaking process using ordering o
        pi, o = self.sampler.sample(alpha, use_rand_perm=self.use_rand_perm)

        # sample z using the reparameterization trick
        z = reparameterization_trick(mu_z, sigma_z)

        return o, pi, z

    def __generative_network(self, y, z, training):
        """
        :param y: class label (can be observed or latent)
        :param z: latent Gaussian variable
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: likelihood model parameters
        """
        # if no input, return None
        if y is None:
            return [None] * 2

        with tf.variable_scope('GenerativeModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # network input
            if self.dim_z > 0:
                network_input = tf.concat((y, z), axis=1)
            else:
                network_input = y

            # build decoder network according to generative data distribution family
            if self.px_z == 'Gaussian':
                mu_x, sigma_x = gaussian_decoder_network(network_input,
                                                         dropout=dropout,
                                                         dim_x=self.dim_x,
                                                         dec_arch=self.dec_arch,
                                                         covariance_structure=self.covariance_structure,
                                                         k=1)
            elif self.px_z == 'Bernoulli':
                mu_x, sigma_x = bernoulli_decoder_network(network_input,
                                                          dropout=dropout,
                                                          dim_x=self.dim_x,
                                                          dec_arch=self.dec_arch,
                                                          k=1)
            else:
                mu_x = sigma_x = None

        return mu_x, sigma_x

    def __kl_divergences(self, o, alpha, mu_z, sigma_z):
        """
        :param o: a uniformly sampled ordering
        :param alpha: approximated posterior's MV-Kumaraswamy concentration parameters
        :param mu_z: approximated posterior's Normal mean
        :param sigma_z: approximated posterior's Normal covariance
        :return: the mean (over the batch) of the KL-Divergences associated with this model
        """
        # take the mean across the batch of the KL Divergence
        d_kl = tf.reduce_mean(self.sampler.kl_divergence(alpha, self.alpha_prior, wrt='Beta-Sticks', i_perm=o))
        if self.dim_z > 0:
            d_kl += tf.reduce_mean(kl_gaussian(q_mu=mu_z,
                                               q_sigma=sigma_z,
                                               p_mu=tf.constant(0, dtype=tf.float32),
                                               p_sigma=tf.constant(1, dtype=tf.float32)))

        return d_kl

    def __loss_labelled(self, x, y):
        """
        :param x: image data
        :param y: observable image label
        :return: -ELBO for labelled data
        """
        # if no input, return None
        if x is None or y is None:
            return [tf.constant(0, dtype=tf.float32)] * 3

        # run recognition network
        alpha, mu_z, sigma_z = self.__recognition_network(x, training=True)

        # sample from the posterior
        o, pi, z = self.__sample_posterior(alpha, mu_z, sigma_z)

        # run generative network
        mu_x, sigma_x = self.__generative_network(y, z, training=True)

        # compute the log likelihood
        ln_px = self.log_likelihood_decoder(x, mu_x, sigma_x)
        ln_py = tf.reduce_sum(y * tf.log(pi + LOG_EPSILON), axis=1)
        ll = ln_px + ln_py

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # get the KL divergences
        d_kl = self.__kl_divergences(o, alpha, mu_z, sigma_z)

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl

    def __loss_unlabelled(self, x):
        """
        :param x: image data
        :return: -ELBO for unlabelled data
        """
        # run recognition network
        alpha, mu_z, sigma_z = self.__recognition_network(x, training=True)

        # sample from the posterior
        o, pi, z = self.__sample_posterior(alpha, mu_z, sigma_z)

        # get all possible class labels that we must marginalize over
        y_all = tf.transpose(tf.eye(self.K, batch_shape=tf.shape(pi)[:1]), [1, 0, 2])

        # run generative network over all possible class labels
        if self.covariance_structure is None:
            mu_x_y = tf.map_fn(lambda y: self.__generative_network(y, z, training=True)[0], y_all)
            sigma_x_y = None
        else:
            mu_x_y, sigma_x_y = tf.map_fn(lambda y: self.__generative_network(y, z, training=True),
                                          y_all,
                                          dtype=(tf.float32, tf.float32))

        # compute the log likelihood
        if sigma_x_y is None:
            ln_px = tf.transpose(tf.map_fn(lambda p: self.log_likelihood_decoder(x, p, None),
                                           mu_x_y,
                                           dtype=tf.float32))
        else:
            ln_px = tf.transpose(tf.map_fn(lambda p: self.log_likelihood_decoder(x, p[0], p[1]),
                                           (mu_x_y, sigma_x_y),
                                           dtype=tf.float32))
        ln_py = tf.log(pi + LOG_EPSILON)
        ll = tf.reduce_logsumexp(ln_px + ln_py, axis=-1)

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # get the KL divergences
        d_kl = self.__kl_divergences(o, alpha, mu_z, sigma_z)

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl


class AutoEncodingDirichlet(VariationalAutoEncoder):

    def __init__(self, x_lat, x_obs=None, y_obs=None, **kwargs):
        """
        :param x_lat: image data with latent labels
        :param x_obs: image data with observed labels
        :param y_obs: labels for x_obs
        :param kwargs: configuration dictionary for the base VAE class
        """
        # initialize base class
        VariationalAutoEncoder.__init__(self, **kwargs)

        # set the training task type (i.e. unsupervised or semi-supervised)
        self.set_task_type(x_obs=x_obs, y_obs=y_obs)

        # duplicate data for MC sampling
        x_lat, x_obs, y_obs = self.duplicate_training_data(x_lat, x_obs, y_obs)

        # labelled loss
        self.loss_labelled, self.neg_ll_labelled, self.dkl_labelled = self.__loss_labelled(x_obs, y_obs)

        # unlabelled loss
        self.loss_unlabelled, self.neg_ll_unlabelled, self.dkl_unlabelled = self.__loss_unlabelled(x_lat)

        # compute re-weighted objective
        self.loss = self.loss_unlabelled + self.loss_labelled

        # configure training operation
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        global_step=self.global_step,
                                                        learning_rate=self.learning_rate,
                                                        optimizer=self.optimizer,
                                                        summaries=['loss', 'gradients'])

        # declare test operations (deactivates neural network regularization methods such as drop out)
        self.alpha_lat_test, self.mu_x_test, _ = self.__recognition_network(x_lat, training=False)
        self.alpha_obs_test, _, _ = self.__recognition_network(x_obs, training=False)
        self.x_recon, _ = self.__generative_network(tf.one_hot(tf.argmax(self.alpha_lat_test, axis=-1), self.K),
                                                    self.mu_x_test,
                                                    training=False)
        self.x_latent = self.latent_representation(self.__generative_network)

    def __recognition_network(self, x, training):
        """
        :param x: image data
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: posterior approximation parameters
        """
        # if no input, return None
        if x is None:
            return [None] * 3

        with tf.variable_scope('RecognitionModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.enc_arch)

            # add Dirichlet recognition layer
            alpha = dirichlet_encoder_layer(x=x, K=self.K)

            # add Gaussian recognition layer if needed
            if self.dim_z > 0:
                mu_z, sigma_z = gaussian_encoder_layer(x=x, dim_z=self.dim_z)
            else:
                mu_z = sigma_z = None

        return alpha, mu_z, sigma_z

    @staticmethod
    def __sample_guassian_posterior(mu_z, sigma_z):
        """
        :param mu_z: Gaussian mean
        :param sigma_z: Gaussian covariance
        :return: z~N(mu_z, sigma_z)
        """
        # sample z using the reparameterization trick
        z = reparameterization_trick(mu_z, sigma_z)

        return z

    def __generative_network(self, y, z, training):
        """
        :param y: class label (can be observed or latent)
        :param z: latent Gaussian variable
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: likelihood model parameters
        """
        # if no input, return None
        if y is None:
            return [None] * 2

        with tf.variable_scope('GenerativeModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # network input
            if self.dim_z > 0:
                network_input = tf.concat((y, z), axis=1)
            else:
                network_input = y

            # build decoder network according to generative data distribution family
            if self.px_z == 'Gaussian':
                mu_x, sigma_x = gaussian_decoder_network(network_input,
                                                         dropout=dropout,
                                                         dim_x=self.dim_x,
                                                         dec_arch=self.dec_arch,
                                                         covariance_structure=self.covariance_structure,
                                                         k=1)
            elif self.px_z == 'Bernoulli':
                mu_x, sigma_x = bernoulli_decoder_network(network_input,
                                                          dropout=dropout,
                                                          dim_x=self.dim_x,
                                                          dec_arch=self.dec_arch,
                                                          k=1)
            else:
                mu_x = sigma_x = None

        return mu_x, sigma_x

    def __loss_labelled(self, x, y):
        """
        :param x: image data
        :param y: observable image label
        :return: -ELBO for labelled data
        """
        # if no input, return None
        if x is None or y is None:
            return [tf.constant(0, dtype=tf.float32)] * 3

        # run recognition network
        alpha, mu_z, sigma_z = self.__recognition_network(x, training=True)

        # sample from the posterior
        dirichlet_posterior = tf.distributions.Dirichlet(alpha)
        pi = dirichlet_posterior.sample()
        z = self.__sample_guassian_posterior(mu_z, sigma_z)

        # run generative network
        mu_x, sigma_x = self.__generative_network(y, z, training=True)

        # compute the log likelihood
        ln_px = self.log_likelihood_decoder(x, mu_x, sigma_x)
        ln_py = tf.reduce_sum(y * tf.log(pi + LOG_EPSILON), axis=1)
        ll = ln_px + ln_py

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # get the KL divergences
        d_kl = tf.reduce_mean(dirichlet_posterior.kl_divergence(tf.distributions.Dirichlet(self.alpha_prior)))
        if self.dim_z > 0:
            d_kl += tf.reduce_mean(kl_gaussian(q_mu=mu_z,
                                               q_sigma=sigma_z,
                                               p_mu=tf.constant(0, dtype=tf.float32),
                                               p_sigma=tf.constant(1, dtype=tf.float32)))
        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl

    def __loss_unlabelled(self, x):
        """
        :param x: image data
        :return: -ELBO for unlabelled data
        """
        # run recognition network
        alpha, mu_z, sigma_z = self.__recognition_network(x, training=True)

        # sample from the posterior
        dirichlet_posterior = tf.distributions.Dirichlet(alpha)
        pi = dirichlet_posterior.sample()
        z = self.__sample_guassian_posterior(mu_z, sigma_z)

        # get all possible class labels that we must marginalize over
        y_all = tf.transpose(tf.eye(self.K, batch_shape=tf.shape(pi)[:1]), [1, 0, 2])

        # run generative network over all possible class labels
        if self.covariance_structure is None:
            mu_x_y = tf.map_fn(lambda y: self.__generative_network(y, z, training=True)[0], y_all)
            sigma_x_y = None
        else:
            mu_x_y, sigma_x_y = tf.map_fn(lambda y: self.__generative_network(y, z, training=True),
                                          y_all,
                                          dtype=(tf.float32, tf.float32))

        # compute the log likelihood
        if sigma_x_y is None:
            ln_px = tf.transpose(tf.map_fn(lambda p: self.log_likelihood_decoder(x, p, None),
                                           mu_x_y,
                                           dtype=tf.float32))
        else:
            ln_px = tf.transpose(tf.map_fn(lambda p: self.log_likelihood_decoder(x, p[0], p[1]),
                                           (mu_x_y, sigma_x_y),
                                           dtype=tf.float32))
        ln_py = tf.log(pi + LOG_EPSILON)
        ll = tf.reduce_logsumexp(ln_px + ln_py, axis=-1)

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # get the KL divergences
        d_kl = tf.reduce_mean(dirichlet_posterior.kl_divergence(tf.distributions.Dirichlet(self.alpha_prior)))
        if self.dim_z > 0:
            d_kl += tf.reduce_mean(kl_gaussian(q_mu=mu_z,
                                               q_sigma=sigma_z,
                                               p_mu=tf.constant(0, dtype=tf.float32),
                                               p_sigma=tf.constant(1, dtype=tf.float32)))

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl


class AutoEncodingSoftmax(VariationalAutoEncoder):

    def __init__(self, x_lat, x_obs=None, y_obs=None, **kwargs):
        """
        :param x_lat: image data with latent labels
        :param x_obs: image data with observed labels
        :param y_obs: labels for x_obs
        :param kwargs: configuration dictionary for the base VAE class
        """
        # initialize base class
        VariationalAutoEncoder.__init__(self, **kwargs)

        # set the training task type (i.e. unsupervised or semi-supervised)
        self.set_task_type(x_obs=x_obs, y_obs=y_obs)

        # duplicate data for MC sampling
        x_lat, x_obs, y_obs = self.duplicate_training_data(x_lat, x_obs, y_obs)

        # construct softmax approximation of Dirichlet priors
        self.mu_pi_prior, self.sigma_pi_prior = dirichlet_prior_laplace_approx(self.alpha_prior, self.K)

        # labelled loss
        self.loss_labelled, self.neg_ll_labelled, self.dkl_labelled = self.__loss_labelled(x_obs, y_obs)

        # unlabelled loss
        self.loss_unlabelled, self.neg_ll_unlabelled, self.dkl_unlabelled = self.__loss_unlabelled(x_lat)

        # compute re-weighted objective
        self.loss = self.loss_unlabelled + self.loss_labelled

        # configure training operation
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        global_step=self.global_step,
                                                        learning_rate=self.learning_rate,
                                                        optimizer=self.optimizer,
                                                        summaries=['loss', 'gradients'])

        # declare test operations (deactivates neural network regularization methods such as drop out)
        self.alpha_lat_test, _, self.mu_x_test, _ = self.__recognition_network(x_lat, training=False)
        self.alpha_obs_test, _, _, _ = self.__recognition_network(x_obs, training=False)
        self.x_recon, _ = self.__generative_network(tf.one_hot(tf.argmax(self.alpha_lat_test, axis=-1), self.K),
                                                    self.mu_x_test,
                                                    training=False)
        self.x_latent = self.latent_representation(self.__generative_network)

    def __recognition_network(self, x, training):
        """
        :param x: image data
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: posterior approximation parameters
        """
        # if no input, return None
        if x is None:
            return [None] * 4

        with tf.variable_scope('RecognitionModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.enc_arch)

            # add Gaussian recognition layer
            mu, sigma = gaussian_encoder_layer(x=x, dim_z=self.K + self.dim_z)

            # partition results
            mu_pi = mu[:, :self.K]
            sigma_pi = sigma[:, :self.K]
            if self.dim_z > 0:
                mu_z = mu[:, self.K:]
                sigma_z = sigma[:, self.K:]
            else:
                mu_z = sigma_z = None

        return mu_pi, sigma_pi, mu_z, sigma_z

    @staticmethod
    def __sample_posterior(mu_pi, sigma_pi, mu_z, sigma_z):
        """
        :param mu_pi: Gaussian logit mean
        :param sigma_pi: Gaussian logit covariance
        :param mu_z: Gaussian mean
        :param sigma_z: Gaussian covariance
        :return: pi~softmax(N(mu_pi, sigma_pi)), z~N(mu_z, sigma_z)
        """
        # sample on the simplex using the softmax approximation
        pi = reparameterization_trick(mu_pi, sigma_pi, softmax=True)

        # sample z using the reparameterization trick
        z = reparameterization_trick(mu_z, sigma_z)

        return pi, z

    def __generative_network(self, y, z, training):
        """
        :param y: class label (can be observed or latent)
        :param z: latent Gaussian variable
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: likelihood model parameters
        """
        # if no input, return None
        if y is None:
            return [None] * 2

        with tf.variable_scope('GenerativeModel', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # network input
            if self.dim_z > 0:
                network_input = tf.concat((y, z), axis=1)
            else:
                network_input = y

            # build decoder network according to generative data distribution family
            if self.px_z == 'Gaussian':
                mu_x, sigma_x = gaussian_decoder_network(network_input,
                                                         dropout=dropout,
                                                         dim_x=self.dim_x,
                                                         dec_arch=self.dec_arch,
                                                         covariance_structure=self.covariance_structure,
                                                         k=1)
            elif self.px_z == 'Bernoulli':
                mu_x, sigma_x = bernoulli_decoder_network(network_input,
                                                          dropout=dropout,
                                                          dim_x=self.dim_x,
                                                          dec_arch=self.dec_arch,
                                                          k=1)
            else:
                mu_x = sigma_x = None

        return mu_x, sigma_x

    def __kl_divergences(self, mu_pi, sigma_pi, mu_z, sigma_z):
        """
        :param mu_z: approximated posterior's Normal logits mean
        :param sigma_z: approximated posterior's Normal logits covariance
        :param mu_z: approximated posterior's Normal mean
        :param sigma_z: approximated posterior's Normal covariance
        :return: the mean (over the batch) of the KL-Divergences associated with this model
        """
        # take the mean across the batch of the KL Divergence
        d_kl = tf.reduce_mean(kl_gaussian(q_mu=mu_pi,
                                          q_sigma=sigma_pi,
                                          p_mu=self.mu_pi_prior,
                                          p_sigma=self.sigma_pi_prior))
        if self.dim_z > 0:
            d_kl += tf.reduce_mean(kl_gaussian(q_mu=mu_z,
                                               q_sigma=sigma_z,
                                               p_mu=tf.constant(0, dtype=tf.float32),
                                               p_sigma=tf.constant(1, dtype=tf.float32)))

        return d_kl

    def __loss_labelled(self, x, y):
        """
        :param x: image data
        :param y: observable image label
        :return: -ELBO for labelled data
        """
        # if no input, return None
        if x is None or y is None:
            return [tf.constant(0, dtype=tf.float32)] * 3

        # run recognition network
        mu_pi, sigma_pi, mu_z, sigma_z = self.__recognition_network(x, training=True)

        # sample from the posterior
        pi, z = self.__sample_posterior(mu_pi, sigma_pi, mu_z, sigma_z)

        # run generative network
        mu_x, sigma_x = self.__generative_network(y, z, training=True)

        # compute the log likelihood
        ln_px = self.log_likelihood_decoder(x, mu_x, sigma_x)
        ln_py = tf.reduce_sum(y * tf.log(pi + LOG_EPSILON), axis=1)
        ll = ln_px + ln_py

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # get the KL divergences
        d_kl = self.__kl_divergences(mu_pi, sigma_pi, mu_z, sigma_z)

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl

    def __loss_unlabelled(self, x):
        """
        :param x: image data
        :return: -ELBO for unlabelled data
        """
        # run recognition network
        mu_pi, sigma_pi, mu_z, sigma_z = self.__recognition_network(x, training=True)

        # sample from the posterior
        pi, z = self.__sample_posterior(mu_pi, sigma_pi, mu_z, sigma_z)

        # get all possible class labels that we must marginalize over
        y_all = tf.transpose(tf.eye(self.K, batch_shape=tf.shape(pi)[:1]), [1, 0, 2])

        # run generative network over all possible class labels
        if self.covariance_structure is None:
            mu_x_y = tf.map_fn(lambda y: self.__generative_network(y, z, training=True)[0], y_all)
            sigma_x_y = None
        else:
            mu_x_y, sigma_x_y = tf.map_fn(lambda y: self.__generative_network(y, z, training=True),
                                          y_all,
                                          dtype=(tf.float32, tf.float32))

        # compute the log likelihood
        if sigma_x_y is None:
            ln_px = tf.transpose(tf.map_fn(lambda p: self.log_likelihood_decoder(x, p, None),
                                           mu_x_y,
                                           dtype=tf.float32))
        else:
            ln_px = tf.transpose(tf.map_fn(lambda p: self.log_likelihood_decoder(x, p[0], p[1]),
                                           (mu_x_y, sigma_x_y),
                                           dtype=tf.float32))
        ln_py = tf.log(pi + LOG_EPSILON)
        ll = tf.reduce_logsumexp(ln_px + ln_py, axis=-1)

        # take the mean across the batch of the negative log likelihood
        neg_ll = tf.reduce_mean(-ll)

        # get the KL divergences
        d_kl = self.__kl_divergences(mu_pi, sigma_pi, mu_z, sigma_z)

        # compute total loss
        loss = neg_ll + d_kl

        return loss, neg_ll, d_kl


class KingmaM2(VariationalAutoEncoder):

    def __init__(self, x_lat, x_obs=None, y_obs=None, **kwargs):
        """
        :param x_lat: image data with latent labels
        :param x_obs: image data with observed labels
        :param y_obs: labels for x_obs
        :param kwargs: configuration dictionary for the base VAE class
        """
        # initialize base class (this model only supports dim(z) > 0)
        VariationalAutoEncoder.__init__(self, **kwargs)
        assert kwargs['dim_z'] > 0

        # set the training task type (this model only supports semi-supervised learning)
        self.set_task_type(x_obs=x_obs, y_obs=y_obs)
        assert self.task_type == 'semi-supervised'

        # duplicate data for MC sampling
        x_lat, x_obs, y_obs = self.duplicate_training_data(x_lat, x_obs, y_obs)

        # labelled loss
        self.loss_labelled, self.neg_ll_labelled, self.dkl_labelled = self.__loss_labelled(x_obs, y_obs)
        self.loss_labelled = tf.reduce_mean(self.loss_labelled)
        self.neg_ll_labelled = tf.reduce_mean(self.neg_ll_labelled)
        self.dkl_labelled = tf.reduce_mean(self.dkl_labelled)

        # unlabelled loss
        self.loss_unlabelled = self.__loss_unlabelled(x_lat)
        self.loss_unlabelled = tf.reduce_mean(self.loss_unlabelled)
        self.neg_ll_unlabelled = tf.constant(0, dtype=tf.float32)
        self.dkl_unlabelled = tf.constant(0, dtype=tf.float32)

        # bonus loss
        self.qy_x = tf.reduce_sum(y_obs * self.__recognition_network_pi(x_obs, training=True), axis=-1)
        self.loss_cross_entropy = tf.reduce_mean(-tf.log(self.qy_x + LOG_EPSILON))

        # compute objective
        self.loss = self.loss_labelled + self.loss_unlabelled + 0.1 * self.loss_cross_entropy

        # configure training operation
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        global_step=self.global_step,
                                                        learning_rate=self.learning_rate,
                                                        optimizer=self.optimizer,
                                                        summaries=['loss', 'gradients'])

        # declare test operations (deactivates neural network regularization methods such as drop out)
        self.alpha_lat_test = self.__recognition_network_pi(x_lat, training=False)
        self.mu_z_test, _ = self.__recognition_network_z(x_lat,
                                                         tf.one_hot(tf.argmax(self.alpha_lat_test, axis=-1), self.K),
                                                         training=False)
        self.alpha_obs_test = self.__recognition_network_pi(x_obs, training=False)
        self.x_recon, _ = self.__generative_network_x(self.mu_z_test, training=False)
        self.x_latent = None

    def __recognition_network_pi(self, x, training):
        """
        :param x: image data
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: posterior approximation parameter for pi
        """
        # if no input, return None
        if x is None:
            return [None] * 3

        with tf.variable_scope('qy_x', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.enc_arch)

            # compute pi
            pi = tf.layers.dense(inputs=x,
                                 units=self.K,
                                 activation=tf.nn.softmax,
                                 use_bias=True,
                                 kernel_initializer=WEIGHT_INIT,
                                 bias_initializer=BIAS_INIT,
                                 name='pi')

        return pi

    def __recognition_network_z(self, x, y, training):
        """
        :param x: image data
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: posterior approximation parameters for z
        """
        # if no input, return None
        if x is None:
            return [None] * 3

        with tf.variable_scope('qz_xy', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # construct base network
            x = base_encoder_network(x, training, dropout, self.enc_arch, y=y)

            # add Gaussian recognition layer
            mu_z, sigma_z = gaussian_encoder_layer(x=x, dim_z=self.dim_z)

        return mu_z, sigma_z

    def __generative_network_x(self, z, training):
        """
        :param z: latent Gaussian variable
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: likelihood model parameters for data
        """
        with tf.variable_scope('px_z', reuse=tf.AUTO_REUSE):

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # build decoder network according to generative data distribution family
            if self.px_z == 'Gaussian':
                mu_x, sigma_x = gaussian_decoder_network(z,
                                                         dropout=dropout,
                                                         dim_x=self.dim_x,
                                                         dec_arch=self.dec_arch,
                                                         covariance_structure=self.covariance_structure,
                                                         k=1)
            elif self.px_z == 'Bernoulli':
                mu_x, sigma_x = bernoulli_decoder_network(z,
                                                          dropout=dropout,
                                                          dim_x=self.dim_x,
                                                          dec_arch=self.dec_arch,
                                                          k=1)
            else:
                mu_x = sigma_x = None

        return mu_x, sigma_x

    def __generative_network_y(self, x, z, training):
        """
        :param x: image data
        :param z: latent Gaussian variable
        :param training: a boolean that will activate/deactivate drop out accordingly
        :return: likelihood model parameters for labels
        """
        with tf.variable_scope('py_xz', reuse=tf.AUTO_REUSE) as scope:

            # determine dropout usage
            dropout = self.dropout_prob * tf.constant(float(training), tf.float32)

            # network input
            x = tf.concat((tf.layers.flatten(x), z), axis=-1)

            # loop over fully connected layers
            for i in range(len(self.dec_arch['full'])):
                # run fully connected layer
                x = tf.layers.dense(inputs=x,
                                    units=self.dec_arch['full'][i],
                                    activation=STANDARD_ACTIVATION,
                                    use_bias=True,
                                    kernel_initializer=WEIGHT_INIT,
                                    bias_initializer=BIAS_INIT,
                                    name='full_layer{:d}'.format(i + 1))

                # apply drop out
                x = tf.layers.dropout(x, rate=dropout)

            # final layer
            y_hat = tf.layers.dense(inputs=x,
                                    units=self.K,
                                    activation=tf.nn.softmax,
                                    use_bias=True,
                                    kernel_initializer=WEIGHT_INIT,
                                    bias_initializer=BIAS_INIT,
                                    name='y_hat')

        return y_hat

    def __loss_labelled(self, x, y):
        """
        :param x: image data
        :param y: observable image label
        :return: -ELBO for labelled data
        """
        # run encoder
        mu_z, sigma_z = self.__recognition_network_z(x, y, training=True)

        # run sampler
        z = reparameterization_trick(mu_z, sigma_z)

        # run decoder
        mu_x, sigma_x = self.__generative_network_x(z, training=True)
        y_hat = self.__generative_network_y(x, z, training=True)

        # compute the negative log likelihood
        ln_px = self.log_likelihood_decoder(x, mu_x, sigma_x)
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
        """
        :param x: image data
        :return: -ELBO for unlabelled data
        """
        # run labelled loss for all possible labels
        y_all = tf.transpose(tf.eye(self.K, batch_shape=tf.shape(x)[:1]), [1, 0, 2])
        loss = tf.map_fn(lambda y: self.__loss_labelled(x, y)[0], y_all)

        # run encoder
        pi = self.__recognition_network_pi(x, training=True)

        # take the expectation
        loss = tf.einsum('ik,ki->i', pi, loss)

        # subtract the entropy
        loss += tf.reduce_sum(pi * tf.log(pi + LOG_EPSILON), axis=-1)

        return loss
