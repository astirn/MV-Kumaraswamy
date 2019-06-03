import numpy as np
import tensorflow as tf
from scipy.stats import beta
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class StickBreakingProcess(object):

    def __init__(self):

        # set number of test samples to be used after fitting the distribution
        self.num_test_samples = int(1e5)

    def plot_marginal_dirichlet_fit(self, pi, alpha_target, d_kl, dist_name):
        """
        :param pi: samples of simplex random variables
        :param alpha_target: parameters of target Dirichlet
        :param d_kl: KL-Divergence objective (for plotting purposes)
        :param dist_name: name of distribution used to generate pi (for plotting purposes)
        :return: None
        """
        # get number of classes
        K = len(alpha_target)

        # get number of plots and determine plot arrangement
        num_plots = 1 + K
        num_rows = 3
        num_cols = int(np.ceil(num_plots / num_rows))

        # declare plot
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(16, 10))
        ax = ax.reshape(-1)

        # plot learning curve
        ax[0].set_title('Training Objective: $D_{kl}(q || p)$')
        ax[0].plot(np.arange(1, 1 + len(d_kl)), d_kl, 'k-', lw=2, alpha=0.6, label='Dkl')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('$D_{kl}(q || p)$')

        # take some samples from a true Dirichlet
        pi_true = np.random.dirichlet(alpha_target, self.num_test_samples)

        # compute useful term
        alpha_target_0 = np.sum(alpha_target)

        # loop over the Beta marginals
        for k in range(K):

            # subplot title
            ax[1 + k].set_title('Beta Marginal for $\\pi_{' + str(k + 1) + '}$')
            ax[1 + k].set_xlabel('$\\pi_{' + str(k + 1) + '}$')
            ax[1 + k].set_ylabel('Density')

            # set true Beta marginal evaluation points
            x = np.linspace(beta.ppf(0.01, alpha_target[k], alpha_target_0 - alpha_target[k]),
                            beta.ppf(0.99, alpha_target[k], alpha_target_0 - alpha_target[k]),
                            100)

            # plot true Beta marginal pdf
            ax[1 + k].plot(x, beta.pdf(x, alpha_target[k], alpha_target_0 - alpha_target[k]), 'k-', lw=2, label='pdf')

            # plot histogram densities of generated samples (proposed and true Dirichlet)
            hist = ax[1 + k].hist([pi[:, k], pi_true[:, k]],
                                  density=True,
                                  histtype='stepfilled',
                                  alpha=0.2,
                                  bins=100,
                                  label=(dist_name, 'Dirichlet'))

            # enable the legend
            ax[1 + k].legend()

            # limit y-axis appropriately for non-saturating Beta distributions
            ax[1 + k].set_ylim(0, 1.25 * max([np.max(x) for x in hist[0]]))

        # make it tight
        plt.tight_layout()


class KumaraswamyStickBreakingProcess(StickBreakingProcess):

    def __init__(self, dkl_taylor_order=10):
        """
        :param K: number of classes
        :param dkl_taylor_order: Taylor expansion approximation order for the KL-divergence computation
        """
        # initialize base class
        StickBreakingProcess.__init__(self)

        # save Taylor approximation order
        assert isinstance(dkl_taylor_order, int) and dkl_taylor_order >= 1
        self.M = dkl_taylor_order

    @staticmethod
    def __parameter_rank_check(alpha):
        """
        This function ensures any parameter vector has rank 2 (i.e. at least [1 x K])
        :param alpha: Dirichlet parameters
        :return: alpha: Dirichlet parameters
        """
        # if alpha is not already [batch size x K], ensure alpha is then shape [1 x K]
        if len(alpha.get_shape().as_list()) == 1:
            alpha = tf.expand_dims(alpha, axis=0)

        return alpha

    def __stick_break_parameters(self, alpha):
        """
        This function converts a Dirichlet alpha parameter vector to the Beta marginal parameters required by the
        stick-breaking Kumaraswamy-Dirichlet sampling procedure.
        :param alpha: Dirichlet parameters
        :return: a, b: marginal Beta distribution parameters
        """
        # enforce rank
        alpha = self.__parameter_rank_check(alpha)

        # get number of clusters
        K = alpha.shape.as_list()[-1]

        # compute stick-break parameters
        a = []
        b = []
        for k in range(K - 1):
            a.append(alpha[:, k])
            b.append(tf.reduce_sum(alpha[:, k+1:], axis=1))
        a = tf.stack(a, axis=1)
        b = tf.stack(b, axis=1)

        return a, b

    def sample(self, alpha, use_rand_perm=True):
        """
        This method implements the Kumarswamy-Dirichlet sample approximation via a stick-breaking process.
        :param alpha: Dirichlet parameters [batch size x K]
        :param use_rand_perm: whether to perform randomized permutation
        :return:
            pi: a Dirichlet sample approximation generated by string cutting Kumaraswamy samples
            i_perm: the sampled permutation indices
        """
        # if no input, return None
        if alpha is None:
            return [None] * 2

        # get number of clusters
        K = alpha.shape.as_list()[-1]

        # using randomized permutation
        if use_rand_perm:

            # sample permutation indices
            i_perm = tf.contrib.framework.argsort(tf.random_uniform(tf.shape(alpha), minval=0, maxval=1), axis=-1)
            i_perm = tf.reshape(i_perm, [-1, K])  # arg sort returns [None, None] despite documentation

            # apply permutation since the pi[K] has bias for sparse sampling
            alpha = tf.batch_gather(alpha, i_perm)

        # no random permutation
        else:

            i_perm = None

        # get stick-break parameters
        a, b = self.__stick_break_parameters(alpha)

        # sample uniform noise (setting minimum value > 0 resolves numerical stability issues)
        u = tf.random_uniform(tf.shape(a), minval=1e-4, maxval=1)

        # sample Kumaraswamy (Beta approximations) via inverse CDF sampling
        x = (1 - (1 - u) ** (1 / b)) ** (1 / a)

        # convert to a Dirichlet sample approximation via Beta string cutting method
        pi = [x[:, 0]]
        for j in range(1, K - 1):
            pi.append((1 - tf.reduce_sum(tf.stack(pi, axis=-1), axis=-1)) * x[:, j])
        pi.append(1 - tf.reduce_sum(tf.stack(pi, axis=-1), axis=-1))
        pi = tf.stack(pi, axis=1)

        # using randomized permutation
        if use_rand_perm:

            # construct inverse permutation indices
            i_inv_perm = tf.contrib.framework.argsort(i_perm, axis=-1)
            i_inv_perm = tf.reshape(i_inv_perm, [-1, K])  # once again, arg sort returns [None, None]

            # invert permutation
            pi = tf.batch_gather(pi, i_inv_perm)

        return pi, i_perm

    def kl_divergence(self, alpha, alpha_prior, i_perm=None, wrt='Dirichlet-Marginals'):
        """
        Computes the KL divergence between the Kumaraswamy q distributions and the Dirichlet prior's Beta marginals.
        :param alpha: posterior approximation Dirichlet parameters
        :param alpha_prior: prior Dirichlet parameters
        :param i_perm: random permutation indices used during sampling procedure
        :param wrt: that which the KL divergence is with respect to, either Dirichlet marginal or Beta stick breaks
        :return: KL divergence of marginal Beta distributions of shape [batch size x K]
        """
        assert wrt in {'Dirichlet-Marginals', 'Beta-Sticks'}

        # apply permutation if one was provided
        if i_perm is not None:
            alpha_prior = self.__parameter_rank_check(alpha_prior)
            alpha_prior = tf.tile(alpha_prior, tf.stack((tf.shape(alpha)[0], 1)))
            alpha = tf.batch_gather(alpha, i_perm)
            alpha_prior = tf.batch_gather(alpha_prior, i_perm)

        # take KL divergence w.r.t. to the Dirichlet's marginal Betas
        if wrt == 'Dirichlet-Marginals':

            # compute marginal q(pi; a', b') approximation parameters
            a_prime = self.__parameter_rank_check(alpha)
            b_prime = tf.reduce_sum(a_prime, axis=1, keepdims=True) - a_prime

            # compute marginal p(pi; a, b) prior parameters
            a_prior = self.__parameter_rank_check(alpha_prior)
            b_prior = tf.reduce_sum(a_prior, axis=1, keepdims=True) - a_prior

        # take KL divergence w.r.t. to the stick-breaking marginal Betas
        else:

            # compute marginal q(pi; a', b') approximation parameters
            a_prime, b_prime = self.__stick_break_parameters(alpha)

            # compute marginal p(pi; a, b) prior parameters
            a_prior, b_prior = self.__stick_break_parameters(alpha_prior)

        # KL-Divergence
        kl = (a_prime - a_prior) / a_prime * (-np.euler_gamma - tf.digamma(b_prime) - 1 / b_prime) \
            + (tf.log(a_prime * b_prime)) \
            + (tf.lbeta(tf.stack((a_prior, b_prior), axis=-1))) \
            - (b_prime - 1) / b_prime
        for m in range(1, self.M + 1):
            B = tf.exp(tf.lbeta(tf.concat((tf.expand_dims(m / a_prime, axis=-1), tf.expand_dims(b_prime, axis=-1)), axis=-1)))
            kl += (b_prior - 1) * b_prime / (m + a_prime * b_prime) * B

        # sum over the dimensions
        kl = tf.reduce_sum(kl, axis=1)

        return kl

    def unit_tests(self, K=10):
        """
        Runs some unit tests on both the sampling procedure and the KL divergence computation.
        :param K: number of classes to run tests over
        :return: None
        """
        # begin test session
        tf.reset_default_graph()
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.Session(config=cfg) as sess:

            # declare parameter placeholder
            alpha_ph = tf.placeholder(dtype=tf.float32, shape=[self.num_test_samples, K])

            # sample some random values
            alpha = 10 * np.random.ranf([self.num_test_samples, K])

            # ensure probabilities sum to 1
            pi = sess.run(self.sample(alpha_ph)[0], feed_dict={alpha_ph: alpha})
            assert (np.abs(np.sum(pi, axis=1) - 1) <= 1e-6).all()

            # set an appropriate prior
            alpha_prior = tf.constant(np.ones(K) / K, dtype=tf.float32)

            # ensure KL divergences are non-negative
            kl = sess.run(self.kl_divergence(alpha_ph, alpha_prior), feed_dict={alpha_ph: alpha})
            assert (kl >= 0).all()

            # test batch gather functionality
            x = np.array([[0, 1, 2], [0, 2, 1],
                          [1, 0, 2], [1, 2, 0],
                          [2, 0, 1], [2, 1, 0]])
            x_sort = np.zeros(x.shape)
            for i in range(x.shape[0]):
                x_sort[i] = x[i, x[i]]
            x_test = sess.run(tf.batch_gather(tf.constant(x, dtype=tf.int32), tf.constant(x, dtype=tf.int32)))
            assert (x_sort == x_test).all()

        # print success
        print('Kumaraswamy-Dirichlet Sampler unit tests passed!')

    def fit(self, alpha_target, use_rand_perm, kl_wrt, learning_rate=5e-3, num_epochs=int(5e4)):
        """
        This is a test method that minimizes the KL divergence between the Kumarswamy approximations to the target
        Dirichlet's Beta marginals. It will plot the true Beta marginal PDF over histograms for exact marginalized Beta
        samples and samples generated by the Kumarswamy-Dirichlet sampler.
        :param alpha_target: target Dirichlet parameters
        :param learning_rate: learning rate applied to KL-divergence gradient
        :param num_epochs: number of training epochs
        :param use_rand_perm: whether to use random permutation matrices during sampling and KL divergence computation
        :param kl_wrt: that which the KL divergence is with respect to, either Dirichlet marginal or Beta stick breaks
        :return: samples in the simplex
        """
        # get number of classes
        K = len(alpha_target)

        # begin test session
        tf.reset_default_graph()
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.Session(config=cfg) as sess:

            # declare parameters we wish to fit
            alpha = tf.get_variable('alpha',
                                    shape=[1, K],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(minval=0, maxval=15))

            # define loss operation
            if use_rand_perm:
                kl = self.kl_divergence(alpha, tf.constant(alpha_target, dtype=tf.float32),
                                        i_perm=self.sample(alpha)[1],
                                        wrt=kl_wrt)
            else:
                kl = self.kl_divergence(alpha, tf.constant(alpha_target, dtype=tf.float32),
                                        i_perm=None,
                                        wrt=kl_wrt)
            loss_op = tf.reduce_sum(kl)

            # configure optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            # configure training operation
            train_op = tf.contrib.layers.optimize_loss(loss=loss_op,
                                                       global_step=tf.Variable(0, name='global_step', trainable=False),
                                                       learning_rate=learning_rate,
                                                       optimizer=optimizer)

            # run initialization
            sess.run(tf.global_variables_initializer())

            # train it
            loss = np.zeros(num_epochs)
            for t in range(num_epochs):
                _, loss[t] = sess.run([train_op, loss_op])
            print('Final DKL = {:.2f}'.format(loss[-1]))
            print('Alpha = ', sess.run(alpha))

            # take some samples
            pi = sess.run(self.sample(tf.tile(alpha, tf.constant([self.num_test_samples, 1])),
                                      use_rand_perm=use_rand_perm)[0])

            # plot results
            self.plot_marginal_dirichlet_fit(pi, alpha_target, loss, 'KDS')

        return pi


class BetaStickBreakingProcess(StickBreakingProcess):

    def __init__(self):
        # initialize base class
        StickBreakingProcess.__init__(self)

    @staticmethod
    def sample(alpha, n):
        """
        This method implements the Dirichlet samples via a stick-breaking process using Beta distributions.
        :param alpha: Dirichlet parameters
        :return: pi: a Dirichlet sample
        """
        # squeeze any extra alpha dimensions
        alpha = np.squeeze(alpha)

        # initialize x
        x = np.zeros([n, len(alpha)])
        pi = np.zeros([n, len(alpha)])

        # take initial sample
        x[:, 0] = np.random.beta(alpha[0], np.sum(alpha[1:]), n)

        # convert to a Dirichlet sample approximation via string cutting
        pi[:, 0] = x[:, 0]
        for j in range(1, len(alpha) - 1):
            x[:, j] = np.random.beta(alpha[j], np.sum(alpha[j + 1:]), n)
            pi[:, j] = ((1 - np.sum(pi[:, :j], axis=-1)) * x[:, j])
        pi[:, -1] = (1 - np.sum(pi[:, :-1], axis=-1))

        # make sure they sum to 1
        assert (np.sum(pi, axis=1) == 1).all()

        return pi

    @staticmethod
    def kl_divergence(alpha, alpha_prior):
        """
        Computes the KL divergence between two Dirichlet distributions
        :param alpha: posterior approximation Dirichlet parameters
        :param alpha_prior: prior Dirichlet parameters
        :return: KL divergence
        """
        # compute convenient terms
        alpha_0 = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        beta_0 = tf.reduce_sum(alpha_prior, axis=-1, keepdims=True)

        # compute KL(q || p)
        kl = \
            tf.lgamma(alpha_0) - \
            tf.reduce_sum(tf.lgamma(alpha), axis=-1) - \
            tf.lgamma(beta_0) + \
            tf.reduce_sum(tf.lgamma(alpha_prior)) + \
            tf.reduce_sum((alpha - alpha_prior) * (tf.digamma(alpha) - tf.digamma(alpha_0)), axis=-1)

        return kl

    def fit(self, alpha_target, learning_rate=1e-2, num_epochs=int(1e4)):
        """
        :param alpha_target: target Dirichlet parameters
        :param learning_rate: learning rate applied to KL-divergence gradient
        :param num_epochs: number of training epochs
        :return: None
        """
        # get number of classes
        K = len(alpha_target)

        # begin test session
        tf.reset_default_graph()
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.Session(config=cfg) as sess:

            # declare parameters we wish to fit
            alpha = tf.get_variable('alpha',
                                    shape=[1, K],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(minval=0, maxval=15))

            # define loss operation
            loss_op = tf.reduce_sum(self.kl_divergence(alpha, tf.constant(alpha_target, dtype=tf.float32)))

            # configure optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # configure training operation
            train_op = tf.contrib.layers.optimize_loss(loss=loss_op,
                                                       global_step=tf.Variable(0, name='global_step', trainable=False),
                                                       learning_rate=learning_rate,
                                                       optimizer=optimizer)

            # run initialization
            sess.run(tf.global_variables_initializer())

            # train it
            loss = np.zeros(num_epochs)
            for t in range(num_epochs):
                _, loss[t] = sess.run([train_op, loss_op])
            print('Final DKL = {:.2f}'.format(loss[-1]))

            # take some samples
            alpha_fit = sess.run(alpha)
            print(alpha_fit)
            pi = self.sample(alpha_fit, self.num_test_samples)

            # plot results
            self.plot_marginal_dirichlet_fit(pi, alpha_target, loss, 'KBS')


def mv_kumaraswamy_ordering_impact(pi_no_perm, pi_random_perm, alpha_target):
    """
    Plots MV Kumaraswamy's ordering's impact on bias (i.e. plots the marginals and shows bias on last dimension)
    :param pi_no_perm: samples that did not use random permutations
    :param pi_random_perm: samples that used random permutations
    :param alpha_target: Dirichlet target parameters
    :return: None
    """
    # get number of classes
    K = len(alpha_target)

    # declare plot
    fig, ax = plt.subplots(1, K, figsize=(8, 2))
    ax = ax.reshape(-1)

    # take some samples from a true Dirichlet
    pi_true = np.random.dirichlet(alpha_target, int(1e6))

    # compute useful term
    alpha_target_0 = np.sum(alpha_target)

    # loop over the Beta marginals
    for k in range(K):

        # enable y label for first axis only
        if K == 0:
            ax[k].set_ylabel('Density')

        # set true Beta marginal evaluation points
        x = np.linspace(beta.ppf(0.01, alpha_target[k], alpha_target_0 - alpha_target[k]),
                        beta.ppf(0.99, alpha_target[k], alpha_target_0 - alpha_target[k]),
                        int(1e5))

        # plot true Beta marginal pdf
        ax[k].plot(x, beta.pdf(x, alpha_target[k], alpha_target_0 - alpha_target[k]), 'k:', lw=2, label='PDF')

        # plot histogram densities of generated samples (proposed and true Dirichlet)
        hist = ax[k].hist([pi_true[:, k], pi_no_perm[:, k], pi_random_perm[:, k]],
                          density=True,
                          histtype='bar',
                          alpha=0.2,
                          bins=100,
                          label=('Dirichlet',
                                 '$f_{12345}$',
                                 '$E[f]$'))

        # limit axes appropriately
        ax[k].set_xlim(0, 0.05)
        ax[k].set_ylim(0, 55)

        # configure ticks
        ax[k].set_xticks([0, 0.03])
        if k > 0:
            ax[k].set_yticks([])
        ax[k].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[k].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        # enable the legend
        if k == int(K / 2):
            ax[k].legend(ncol=4, bbox_to_anchor=(2, -0.125))

    # make it tight
    plt.subplots_adjust(left=0.04, bottom=0.3, right=0.98, top=0.98, wspace=0.05, hspace=0)


if __name__ == '__main__':

    # declare Kumaraswamy stick-breaking sampler and run unit tests
    ksb = KumaraswamyStickBreakingProcess()
    ksb.unit_tests()

    # demonstrate effect ordering has on approximating sparsity-inducing Dirichlet
    alpha = np.ones(5) / 5
    pi_no_perm = ksb.fit(alpha, use_rand_perm=False, kl_wrt='Dirichlet-Marginals')
    pi_random_perm = ksb.fit(alpha, use_rand_perm=True, kl_wrt='Dirichlet-Marginals')
    mv_kumaraswamy_ordering_impact(pi_no_perm, pi_random_perm, alpha)
    plt.show()

    # declare Beta stick-breaking sampler (samples a Dirichlet random variable)
    bsb = BetaStickBreakingProcess()

    # declare some Dirichlet parameters to fit
    alphas = [np.ones(5),
              np.ones(5) / 5,
              1 + np.arange(5),
              np.flip(1 + np.arange(5)),
              (1 + np.arange(5)) * 2,
              np.flip((1 + np.arange(5)) * 2)]

    # loop over the parameters
    for a in alphas:

        # fit Kumaraswamy-Dirichlet sample under various methods
        ksb.fit(a, use_rand_perm=False, kl_wrt='Dirichlet-Marginals')  # not accurate w.r.t the model
        ksb.fit(a, use_rand_perm=True, kl_wrt='Dirichlet-Marginals')  # same as above--perm. does not affect KL wrt Dir.
        ksb.fit(a, use_rand_perm=False, kl_wrt='Beta-Sticks')  # without random permutation, we incur bias
        ksb.fit(a, use_rand_perm=True, kl_wrt='Beta-Sticks')   # truest to model--but stochastic (breaks DKL convexity)

        # fit Beta-Dirichlet sampler
        bsb.fit(a)

        # show results
        plt.show()
