import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mv_kumaraswamy_sampler import KumaraswamyStickBreakingProcess

# set random seeds
np.random.seed(123)
tf.set_random_seed(123)


def irg_variance_test(x, alphas, alpha_prior, N_trials):

    # get useful numbers
    K = x.shape[0]

    # compute optimal posterior parameters
    alpha_star = alpha_prior + x
    print('alpha max = {:.2f}'.format(np.max(alpha_star)))

    # initialize gradients
    gradients = np.zeros([len(alphas), N_trials, K])

    # reset graph with new session
    tf.reset_default_graph()
    with tf.Session() as sess:

        # declare training variable
        alpha_ph = tf.placeholder(tf.float32, [1, K])

        # set prior as a TF constant
        alpha_prior = tf.constant(alpha_prior, dtype=tf.float32)

        # declare sampler
        sampler = tf.distributions.Dirichlet(alpha_ph)
        pi = sampler.sample()

        # compute the expected log likelihood
        ll = tf.reduce_sum(x * tf.log(pi))

        # compute the ELBO
        elbo = ll - sampler.kl_divergence(tf.distributions.Dirichlet(alpha_prior))

        # compute gradient
        grad = tf.gradients(xs=[alpha_ph], ys=elbo)

        # loop over the alphas
        for i in range(len(alphas)):

            # set alpha for this test
            alpha = alpha_star * np.ones([1, K])
            alpha[0, 0] = alphas[i]

            # compute the gradient over the specified number of trials
            for j in range(N_trials):
                gradients[i, j] = sess.run(grad, feed_dict={alpha_ph: alpha})[0]

                # print update
                a_per = 100 * (i + 1) / len(alphas)
                n_per = 100 * (j + 1) / N_trials
                update_str = 'Alphas done: {:.2f}%, Trials done: {:.2f}%'.format(a_per, n_per)
                print('\r' + update_str, end='')

        print('')

    # return the gradients
    return gradients


def mv_kumaraswamy_variance_test(x, alphas, alpha_prior, N_trials):

    # get useful numbers
    K = x.shape[0]

    # compute optimal posterior parameters
    alpha_star = alpha_prior + x
    print('alpha max = {:.2f}'.format(np.max(alpha_star)))

    # initialize gradients
    gradients = np.zeros([len(alphas), N_trials, K])

    # reset graph with new session
    tf.reset_default_graph()
    with tf.Session() as sess:

        # declare training variable
        alpha_ph = tf.placeholder(tf.float32, [1, K])

        # set prior as a TF constant
        alpha_prior = tf.constant(alpha_prior, dtype=tf.float32)

        # declare sampler
        sampler = KumaraswamyStickBreakingProcess()
        pi, i_perm = sampler.sample(alpha_ph)

        # compute the expected log likelihood
        ll = tf.reduce_sum(x * tf.log(pi))

        # compute the ELBO
        elbo = ll - sampler.kl_divergence(alpha=alpha_ph, alpha_prior=alpha_prior, i_perm=i_perm)

        # compute gradient
        grad = tf.gradients(xs=[alpha_ph], ys=elbo)

        # loop over the alphas
        for i in range(len(alphas)):

            # set alpha for this test
            alpha = alpha_star * np.ones([1, K])
            alpha[0, 0] = alphas[i]

            # compute the gradient over the specified number of trials
            for j in range(N_trials):
                gradients[i, j] = sess.run(grad, feed_dict={alpha_ph: alpha})[0]

                # print update
                a_per = 100 * (i + 1) / len(alphas)
                n_per = 100 * (j + 1) / N_trials
                update_str = 'Alphas done: {:.2f}%, Trials done: {:.2f}%'.format(a_per, n_per)
                print('\r' + update_str, end='')

        print('')

    # return the gradients
    return gradients


if __name__ == '__main__':

    K = 100
    N = 100
    p_true = np.random.dirichlet(np.ones(K))
    x = np.random.multinomial(n=N, pvals=p_true)
    alphas = np.linspace(1.01, 3.0, 100)
    grads = irg_variance_test(x=x, alphas=alphas, alpha_prior=np.ones(K), N_trials=100)

    # take the variance across samples
    grad_var = np.var(grads, axis=1)

    plt.figure()
    plt.plot(alphas, grad_var[:, 0])
    plt.show()





