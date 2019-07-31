import os
import copy
import time
import pickle
import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import linear_sum_assignment

# import the models
from models_vae import AutoEncodingKumaraswamy, AutoEncodingDirichlet, AutoEncodingSoftmax, KingmaM2


def pre_process_data(ds, info, px_z):
    """
    :param ds: TensorFlow Dataset object
    :param info: TensorFlow DatasetInfo object
    :param px_z: Decoder modelling assumptions {Gaussian, Bernoulli} that impact pre-processing decisions
    :return: the passed in data set with map pre-processing applied
    """
    assert info.name in {'mnist', 'svhn_cropped'}
    assert px_z in {'Gaussian', 'Bernoulli'}

    def bernoulli_sample(x):
        orig_shape = tf.shape(x)
        p = tf.reshape(tf.cast(x, dtype=tf.float32) / 255.0, [-1, 1])
        logits = tf.log(tf.concat((1 - p, p), axis=1))
        return tf.cast(tf.reshape(tf.random.categorical(logits, 1), orig_shape), dtype=tf.float32)

    # apply pre-processing function for given data set and modelling assumptions
    if px_z == 'Gaussian':
        return ds.map(lambda d: {'image': tf.cast(d['image'], dtype=tf.float32) / d['image'].dtype.max,
                                 'label': d['label']},
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif px_z == 'Bernoulli':
        return ds.map(lambda d: {'image': bernoulli_sample(d['image']),
                                 'label': d['label']},
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        return ds


def configure_data_set(ds, info, px_z, batch_size, repeats):
    """
    :param ds: TensorFlow data set object
    :param info: TensorFlow DatasetInfo object
    :param px_z: data model assumption {Bernoulli, Gaussian}
    :param batch_size: batch size
    :param repeats: number of iterations through data before new epoch--a -1 indicates infinite repetition
    :return: a configured TensorFlow data set object
    """
    # enable cache if infinitely repeating (i.e. the labelled data)
    if repeats == -1:
        ds = ds.cache()

    # enable shuffling and repeats
    ds = ds.shuffle(10 * batch_size, reshuffle_each_iteration=True).repeat(repeats)

    # batch the data before pre-processing
    ds = ds.batch(batch_size)

    # pre-process the data set
    with tf.device('/cpu:0'):
        ds = pre_process_data(ds, info, px_z)

    # enable prefetch
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def generate_split(x, y, number_to_split, num_classes, balanced):
    """
    :param x: numpy array of data
    :param y: numpy array of labels
    :param number_to_split: number of examples to split away
    :param num_classes: number of classes
    :param balanced: boolean that determines whether to balance class labels during the split
    :return: x set-minus x_split, y set-minus y_split, x_split, y_split
    """
    # quick exit
    if number_to_split == 0:
        return x, y, None, None

    # balanced splitting
    if balanced:

        # make sure the split number is divisible by the number of classes
        assert np.mod(number_to_split, num_classes) == 0
        number_to_split = int(number_to_split / num_classes)

        # initialize the split
        x_split = np.zeros([0] + list(x.shape[1:]), dtype=x.dtype)
        y_split = np.zeros([0] + list(y.shape[1:]), dtype=y.dtype)

        # loop over the classes
        for k in range(num_classes):

            # randomly select points to split away corresponding to the label
            i_split = np.random.choice(np.where(y == k)[0], number_to_split, replace=False)

            # gather the indices that remain
            i_keep = list(set(np.arange(x.shape[0])) - set(i_split))

            # perform the split
            x_split = np.concatenate((x_split, x[i_split]))
            y_split = np.concatenate((y_split, y[i_split]))
            x = x[i_keep]
            y = y[i_keep]

    # splitting without balance enforcement
    else:

        # randomly select points to split away
        i_split = np.random.choice(np.arange(x.shape[0]), number_to_split, replace=False)

        # gather the indices that remain
        i_keep = list(set(np.arange(x.shape[0])) - set(i_split))

        # perform the split
        x_split = x[i_split]
        y_split = y[i_split]
        x = x[i_keep]
        y = y[i_keep]

    # shuffle the split
    i_shuffle = np.argsort(np.random.random(x_split.shape[0]))
    x_split = x_split[i_shuffle]
    y_split = y_split[i_shuffle]

    return x, y, x_split, y_split


def load_data_set(data_set_name, px_z, num_validation, num_labelled, balanced, batch_size=100):
    """
    :param data_set_name: data set name--call tfds.list_builders() for options
    :param px_z: Decoder modelling assumptions {Gaussian, Bernoulli} that also determine pre-processing
    :param batch_size: training/testing batch size
    :param num_validation: size of validation set
    :param num_labelled: size of training set with observable labels
    :param balanced: boolean that determines whether to balance class labels in the labelled training set
    :return:
        unlabelled_ds: TensorFlow Dataset object for the training data
        labelled_ds: TensorFlow Dataset object for the labelled data
        test_ds: TensorFlow Dataset object for the testing data
        info: data set info object
    """
    # load training and test sets
    train_ds, info = tfds.load(name=data_set_name, split=tfds.Split.TRAIN, with_info=True)
    test_ds = tfds.load(name=data_set_name, split=tfds.Split.TEST, with_info=False)

    # batch the data to the full set size
    train_ds = train_ds.batch(info.splits['train'].num_examples)
    test_ds = test_ds.batch(info.splits['test'].num_examples)

    # construct iterator
    iterator = train_ds.make_initializable_iterator()
    x, y = iterator.get_next().values()

    # construct initialization operations
    train_iter_init = iterator.make_initializer(train_ds)
    test_iter_init = iterator.make_initializer(test_ds)

    # start a monitored session
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:

        # retrieve training data
        sess.run(train_iter_init)
        x_train, y_train = sess.run([x, y])

        # retrieve test data
        sess.run(test_iter_init)
        x_test, y_test = sess.run([x, y])

    # make sure the sets are unique
    x_all = np.concatenate((x_train, x_test), axis=0)
    assert len(np.unique(x_all, axis=0)) >= x_all.shape[0] - 1  # apparently svhn has a duplicate item across sets

    # get number of classes
    K = len(np.unique(y_train))

    # shuffle the training data
    i_shuffle = np.argsort(np.random.random(x_train.shape[0]))
    x_train = x_train[i_shuffle]
    y_train = y_train[i_shuffle]

    # perform labelled split
    x_train, y_train, x_label, y_label = generate_split(x_train, y_train, num_labelled, K, balanced)

    # perform validation split
    x_train, y_train, x_valid, y_valid = generate_split(x_train, y_train, num_validation, K, balanced)

    # create and configure the data sets
    unlabelled_ds = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train})
    unlabelled_ds = configure_data_set(unlabelled_ds, info, px_z, batch_size, repeats=1)
    if x_label is not None and y_label is not None:
        labelled_ds = tf.data.Dataset.from_tensor_slices({'image': x_label, 'label': y_label})
        labelled_ds = configure_data_set(labelled_ds, info, px_z, batch_size, repeats=-1)
    else:
        labelled_ds = None
    if x_valid is not None and y_valid is not None:
        valid_ds = tf.data.Dataset.from_tensor_slices({'image': x_valid, 'label': y_valid})
        valid_ds = configure_data_set(valid_ds, info, px_z, batch_size, repeats=1)
    else:
        valid_ds = None
    test_ds = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test})
    test_ds = configure_data_set(test_ds, info, px_z, batch_size, repeats=1)

    return unlabelled_ds, labelled_ds, valid_ds, test_ds, info


def set_performance(mdl, sess, y_latent_ph, iter_init, perf, loss_type, i, y_observed_ph=None):
    """
    :param mdl: model class
    :param sess: TensorFlow session
    :param y_latent_ph: TensorFlow placeholder for unseen labels
    :param iter_init: TensorFlow data iterator initializer associated with the unsupervised data
    :param perf: performance dictionary
    :param loss_type: set name (e.g. unsupervised or test)
    :param i: insertion index (i.e. epoch - 1)
    :param y_observed_ph: TensorFlow placeholder for seen labels
    :return: updated performance dictionary
    """
    # initialize results
    loss = []
    loss_unlabelled = []
    neg_ll_unlabelled = []
    dkl_unlabelled = []
    loss_labelled = []
    neg_ll_labelled = []
    dkl_labelled = []
    alpha_unlabelled = np.zeros([0, mdl.K])
    y_latent = np.zeros([0, 1])
    alpha_labelled = np.zeros([0, mdl.K])
    y_observed = np.zeros([0, 1])

    # initialize unsupervised data iterator
    sess.run(iter_init)

    # loop over the batches within the unsupervised data iterator
    print('Evaluating ' + loss_type + ' set performance... ', end='')
    while True:
        try:
            # grab the results
            results = sess.run([mdl.loss,
                                mdl.loss_unlabelled,
                                mdl.neg_ll_unlabelled,
                                mdl.dkl_unlabelled,
                                mdl.loss_labelled,
                                mdl.neg_ll_labelled,
                                mdl.dkl_labelled,
                                mdl.alpha_lat_test,
                                y_latent_ph],
                               feed_dict={mdl.M: 1})

            # load metrics
            loss.append(results[0])
            loss_unlabelled.append(results[1])
            neg_ll_unlabelled.append(results[2])
            dkl_unlabelled.append(results[3])
            loss_labelled.append(results[4])
            neg_ll_labelled.append(results[5])
            dkl_labelled.append(results[6])
            alpha_unlabelled = np.concatenate((alpha_unlabelled, results[7]))
            y_latent = np.concatenate((y_latent, np.expand_dims(results[8], axis=1)))

            # if we are in semi-supervised mode, grab supervised classification results
            if y_observed_ph is not None:
                results = sess.run([mdl.alpha_obs_test, y_observed_ph], feed_dict={mdl.M: 1})
                alpha_labelled = np.concatenate((alpha_labelled, results[0]))
                y_observed = np.concatenate((y_observed, np.expand_dims(results[1], axis=1)))

        # iterator will throw this error when its out of data
        except tf.errors.OutOfRangeError:
            break

    # new line
    print('')

    # average the results
    perf['loss'][loss_type][i] = sum(loss) / len(loss)
    perf['elbo'][loss_type][i] = sum(loss_unlabelled) / len(loss_unlabelled)
    perf['neg_ll'][loss_type][i] = sum(neg_ll_unlabelled) / len(neg_ll_unlabelled)
    perf['dkl'][loss_type][i] = sum(dkl_unlabelled) / len(dkl_unlabelled)
    perf['elbo']['supervised'][i] = sum(loss_labelled) / len(loss_labelled)
    perf['neg_ll']['supervised'][i] = sum(neg_ll_labelled) / len(neg_ll_labelled)
    perf['dkl']['supervised'][i] = sum(dkl_labelled) / len(dkl_labelled)

    # compute classification accuracy
    perf['class_err'][loss_type][i] = unsupervised_labels(alpha_unlabelled, y_latent, mdl, loss_type)
    if y_observed_ph is not None:
        perf['class_err']['supervised'][i] =\
            unsupervised_labels(alpha_labelled, y_observed, mdl, 'supervised')

    return perf


def save_performance(perf, epoch, save_path):
    """
    :param perf: performance dictionary
    :param epoch: epoch number
    :param save_path: path to save plot to. if None, plot will be drawn
    :return: none
    """
    # return if save path is None
    if save_path is None:
        return

    # loop over the metrics
    for metric in perf.keys():

        # loop over the data splits
        for split in perf[metric].keys():

            # trim data to utilized epochs
            perf[metric][split] = perf[metric][split][:epoch]
            assert len(perf[metric][split]) == epoch

    # create the file name
    f_name = os.path.join(save_path, 'perf.pkl')

    # pickle it
    with open(f_name, 'wb') as f:
        pickle.dump(perf, f, pickle.HIGHEST_PROTOCOL)

    # make sure it worked
    with open(f_name, 'rb') as f:
        perf_load = pickle.load(f)
    assert str(perf) == str(perf_load), 'performance saving failed'


def random_concentration_and_reconstruction_per_class(mdl, sess, x_ph, y_ph, iter_init):
    """
    :param mdl: model class
    :param sess: TensorFlow session
    :param x_ph: TensorFlow placeholder
    :param y_ph: TensorFlow placeholder
    :param iter_init: TensorFlow data iterator initializer associated with the unsupervised data
    :return:
        x: original data
        alpha: concentration parameter
        x_recon: reconstruction
    """
    # initialize unsupervised data iterator
    sess.run(iter_init)

    # initialize data containers
    x_list = [None] * mdl.num_classes
    alpha_list = [None] * mdl.num_classes
    x_recon_list = [None] * mdl.num_classes

    # loop until we have the concentration and reconstruction for an element of each class
    while sum([x is None for x in x_list]) > 0:

        # grab the results
        x, y, alpha, x_recon = sess.run([x_ph, y_ph, mdl.alpha_lat_test, mdl.x_recon], feed_dict={mdl.M: 1})

        # loop over the classes
        for k in range(mdl.num_classes):

            # find matching classes
            i_match = np.where(y == k)[0]

            # if there are any
            if len(i_match):

                # pick a random one
                i_rand = np.random.choice(i_match)

                # save the result
                x_list[k] = x[i_rand]
                alpha_list[k] = alpha[i_rand]
                x_recon_list[k] = x_recon[i_rand]

    # convert to lists to arrays
    x = np.stack(x_list, axis=0)
    alpha = np.stack(alpha_list, axis=0)
    x_recon = np.stack(x_recon_list, axis=0)

    return x, alpha, x_recon


def unsupervised_labels(alpha, y, mdl, loss_type):
    """
    :param alpha: concentration parameter
    :param y: true label
    :param mdl: the model object
    :param loss_type: name used for printing updates
    :return: classification error rate
    """
    # same number of classes as labels?
    if mdl.K == mdl.num_classes:

        # construct y-hat
        y_hat = np.argmax(alpha, axis=1)

        # initialize count matrix
        cnt_mtx = np.zeros([mdl.K, mdl.K])

        # fill in matrix
        for i in range(len(y)):
            cnt_mtx[int(y_hat[i]), int(y[i])] += 1

        # find optimal permutation
        row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

        # compute error
        error = 1 - cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()

    # different number of classes than labels
    else:

        # initialize y-hat
        y_hat = -np.ones(y.shape)

        # loop over the number of latent clusters
        for i in range(mdl.K):

            # find the real label corresponding to the largest concentration for this cluster
            i_sort = np.argsort(alpha[:, i])[-100:]
            y_real = stats.mode(y[i_sort])[0]

            # assign that label to all points where its concentration is maximal
            y_hat[np.argmax(alpha, axis=1) == i] = y_real

        # make sure we handled everyone
        assert np.sum(y_hat < 0) == 0

        # compute the error
        error = np.mean(y != y_hat)

    # print results
    print('Classification error for ' + loss_type + ' data = {:.4f}'.format(error))

    return error


def plot_learning_curve(fig, performance, epoch, mode, save_path=None):
    """
    :param fig: figure handle
    :param performance: dictionary of various learning performance indicators
    :param epoch: epoch number
    :param mode: {unsupervised, semi-supervised}
    :param save_path: path to save plot to. if None, plot will be drawn
    :return: none
    """
    # are we skipping supervised plotting for all metrics
    skip_supervised_all = mode == 'unsupervised'

    # clear the figure
    fig.clear()

    # generate epoch numbers
    epochs = np.arange(1, epoch + 1)

    # common plotting function
    def plot_all(metric_name, skip_supervised=False):
        for key in performance[metric_name].keys():
            if skip_supervised and key == 'supervised':
                continue
            sp.plot(epochs, performance[metric_name][key][:epoch], label=key, linestyle=':')
            sp.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # plot the loss
    sp = fig.add_subplot(2, 2, 1)
    sp.set_title('Total Loss')
    plot_all('loss', skip_supervised=True)
    sp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plot the unsupervised label accuracy
    sp = fig.add_subplot(2, 2, 2)
    sp.set_title('Classification Error')
    plot_all('class_err', skip_supervised=skip_supervised_all)
    sp.set_ylim([0, None])
    sp.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # plot the loss components
    sp = fig.add_subplot(2, 3, 4)
    sp.set_title('-ELBO')
    plot_all('elbo', skip_supervised=skip_supervised_all)
    sp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    sp = fig.add_subplot(2, 3, 5)
    sp.set_title('Negative Log Likelihood')
    plot_all('neg_ll', skip_supervised=skip_supervised_all)
    sp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    sp.legend(ncol=len(performance['neg_ll'].keys()), bbox_to_anchor=(2, -0.125))

    sp = fig.add_subplot(2, 3, 6)
    sp.set_title('KL-Divergence')
    plot_all('dkl', skip_supervised=skip_supervised_all)
    sp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # eliminate those pesky margins
    plt.subplots_adjust(left=0.1, bottom=0.125, right=0.99, top=0.95, wspace=0.35, hspace=0.35)

    # draw or save
    if save_path is not None:
        fig.savefig(os.path.join(save_path, 'Learning_Curve.png'), dpi=600)


def train(method, config, unlabelled_set, labelled_set, valid_set, test_set, n_epochs, early_stop_buffer=15):
    """
    :param method: VAE method {Kumaraswamy, Softmax, KingmaM2}
    :param config: VAE configuration dictionary
    :param unlabelled_set: TensorFlow Dataset object that corresponds to unlabelled training data
    :param labelled_set: TensorFlow Dataset object that corresponds to labelled training data
    :param valid_set: TensorFlow Dataset object that corresponds to validation data
    :param test_set: TensorFlow Dataset object that corresponds to validation data
    :param n_epochs: number of epochs to run
    :param early_stop_buffer: early stop look-ahead distance (in epochs)
    :return: None
    """
    # make sure the method is supported
    assert method in {'Kumaraswamy', 'Nalisnick', 'Dirichlet', 'Softmax', 'KingmaM2'}

    # construct iterator
    iterator = unlabelled_set.make_initializable_iterator()
    x, y = iterator.get_next().values()

    # construct initialization operations
    unlabelled_iter_init = iterator.make_initializer(unlabelled_set)
    if valid_set is not None:
        valid_iter_init = iterator.make_initializer(valid_set)
    else:
        valid_iter_init = iterator.make_initializer(test_set)
    test_iter_init = iterator.make_initializer(test_set)

    # semi-supervised mode
    if labelled_set is not None:

        # construct one-shot iterator
        labelled_iter = labelled_set.make_one_shot_iterator()
        x_obs, y_obs = labelled_iter.get_next().values()

    # unsupervised mode
    else:
        x_obs = y_obs = None

    # construct the model according to method and dim(z)
    if method == 'Kumaraswamy':
        mdl = AutoEncodingKumaraswamy(x_lat=x, x_obs=x_obs, y_obs=y_obs, use_rand_perm=True, **config)
    elif method == 'Nalisnick':
        mdl = AutoEncodingKumaraswamy(x_lat=x, x_obs=x_obs, y_obs=y_obs, use_rand_perm=False, **config)
    elif method == 'Dirichlet':
        mdl = AutoEncodingDirichlet(x_lat=x, x_obs=x_obs, y_obs=y_obs, **config)
    elif method == 'Softmax':
        mdl = AutoEncodingSoftmax(x_lat=x, x_obs=x_obs, y_obs=y_obs, **config)
    elif method == 'KingmaM2' and config['dim_z'] > 0:
        mdl = KingmaM2(x_lat=x, x_obs=x_obs, y_obs=y_obs, **config)
    else:
        return

    # initialize performance dictionary
    init_dict = {'test': np.zeros(n_epochs),
                 'validation': np.zeros(n_epochs),
                 'unsupervised': np.zeros(n_epochs),
                 'supervised': np.zeros(n_epochs)}
    perf = {
        # full loss
        'loss': copy.deepcopy(init_dict),
        # classification error
        'class_err': copy.deepcopy(init_dict),
        # loss components
        'elbo': copy.deepcopy(init_dict),
        'neg_ll': copy.deepcopy(init_dict),
        'dkl': copy.deepcopy(init_dict),
    }

    # declare figure handles for plots updated during training
    fig_learn = plt.figure()

    # start a monitored session
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:

        # initialize model variables
        sess.run(tf.global_variables_initializer())

        # loop over the number of epochs
        for i in range(n_epochs):

            # get epoch number
            epoch = i + 1

            # start timer
            start = time.time()

            # initialize epoch iterator
            sess.run(unlabelled_iter_init)

            # loop over the batches
            while True:
                try:

                    # run just training and loss
                    _, loss = sess.run([mdl.train_op, mdl.loss],
                                       feed_dict={mdl.M: mdl.monte_carlo_samples})
                    if np.isnan(loss):
                        print('\n NaN whelp!')
                        return

                    # print update
                    print('\rEpoch {:d}, Loss = {:.2f}'.format(epoch, loss), end='')

                # iterator will throw this error when its out of data
                except tf.errors.OutOfRangeError:
                    break

            # new line
            print('')

            # get training set performance--this will compute supervised (data with observable labels) metrics as well
            # because supervised iterator repeats some data may be counted multiple times if percent_labelled < 50%
            # also some data might not be counted at all if percent_labelled > 50%
            perf = set_performance(mdl, sess, y, unlabelled_iter_init, perf, 'unsupervised', i, y_obs)

            # get validation and test set performances
            perf = set_performance(mdl, sess, y, valid_iter_init, perf, 'validation', i)
            perf = set_performance(mdl, sess, y, test_iter_init, perf, 'test', i)

            # TODO: uncomment this for plots!
            # # plot learning curve
            # plot_learning_curve(fig_learn, perf, epoch, mdl.task_type, save_path=mdl.save_dir)
            #
            # # plot reconstructed image(s)
            # x_orig, alpha, x_recon = random_concentration_and_reconstruction_per_class(mdl, sess, x, y, test_iter_init)
            # mdl.plot_random_reconstruction(x_orig, x_recon, alpha, epoch=epoch)
            #
            # # plot latent representation
            # mdl.plot_latent_representation(sess, epoch=epoch)

            # find the best evidence and classification performances
            i_best_elbo = np.argmin(perf['elbo']['validation'][:epoch])
            i_best_class = np.argmin(perf['class_err']['validation'][:epoch])

            # if the current validation ELBO is the best (and we are saving models), save it!
            if i == i_best_elbo and mdl.save_dir is not None:
                pass
                # save_path = mdl.saver.save(sess, os.path.join(mdl.save_dir, 'best_elbo', 'mdl.ckpt'))
                # print('New best ELBO model! Saving results to ' + save_path)

            # if the current validation classification error is the best (and we are saving models), save it!
            if i == i_best_class and mdl.save_dir is not None:
                pass
                # save_path = mdl.saver.save(sess, os.path.join(mdl.save_dir, 'best_class', 'mdl.ckpt'))
                # print('New best classification model! Saving results to ' + save_path)

            # pause for plot drawing if we aren't saving
            if mdl.save_dir is None:
                plt.pause(0.05)

            # print time for epoch
            stop = time.time()
            print('Time for Epoch = {:f}'.format(stop - start))

            # early stop check
            epochs_since_improvement = min(i - i_best_elbo, i - i_best_class)
            print('Early stop checks: {:d} / {:d}\n'.format(epochs_since_improvement, early_stop_buffer))
            if epochs_since_improvement >= early_stop_buffer:
                break

    # save the performance
    save_performance(perf, epoch, mdl.save_dir)


if __name__ == '__main__':

    # pick a data set and modelling assumption
    data_set = 'mnist'
    data_model = 'Gaussian'
    covariance_structure = 'diag'

    # training constants
    n_epochs = 750
    b_size = 250
    drop_out = 0.0

    # load the data set (custom split method)
    unlabelled_set, labelled_set, valid_set, test_set, set_info = load_data_set(data_set_name=data_set,
                                                                                px_z=data_model,
                                                                                num_validation=10000,
                                                                                num_labelled=600,
                                                                                balanced=True,
                                                                                batch_size=b_size)

    # get number of channels
    n_chans = set_info.features['image'].shape[-1]

    # define encoder architecture
    enc_arch = {'conv': [{'k_size': 5, 'out_chan': 5 * n_chans}, {'k_size': 3, 'out_chan': 10 * n_chans}],
                'full': [200, 200]}

    # define decoder architecture
    dec_arch = {'full': [200, 200],
                'conv_start_chans': 10 * n_chans,
                'conv': [{'k_size': 3, 'out_chan': 5 * n_chans}, {'k_size': 5, 'out_chan': n_chans}]}

    # configure the common VAE elements
    config = {
        'dim_x': list(set_info.features['image'].shape),
        'num_classes': set_info.features['label'].num_classes,
        'dim_z': 2,
        'K': 10,
        'enc_arch': enc_arch,
        'dec_arch': dec_arch,
        'learn_rate': 1e-3,
        'px_z': data_model,
        'covariance_structure': covariance_structure,
        'dropout_rate': drop_out,
        'save_dir': None}

    # run training
    train(method='Kumaraswamy',
          config=config,
          unlabelled_set=unlabelled_set,
          labelled_set=labelled_set,
          valid_set=valid_set,
          test_set=test_set,
          n_epochs=n_epochs)

    print('All done!')
    plt.show()
