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
from models_proposed import AutoEncodingCommon, AutoEncodingDirichlet, AutoEncodingDirichletNormal
from models_baseline import KingmaM2, AutoEncodingSoftmax, AutoEncodingSoftmaxNormal


def pre_process_data(ds, ds_name, px_z):
    """
    :param ds: TensorFlow Dataset object
    :param ds_name: name of the data set (unsupported names just return the original set)
    :param px_z: Decoder modelling assumptions {Gaussian, Bernoulli} that impact pre-processing decisions
    :return: the passed in data set with map pre-processing applied
    """
    def bernoulli_sample(x):
        orig_shape = tf.shape(x)
        p = tf.reshape(tf.cast(x, dtype=tf.float32) / 255.0, [-1, 1])
        logits = tf.log(tf.concat((1 - p, p), axis=1))
        return tf.cast(tf.reshape(tf.random.categorical(logits, 1), orig_shape), dtype=tf.float32)

    # apply pre-processing function for given data set and modelling assumptions
    if (ds_name == 'mnist' or ds_name == 'svhn_cropped') and px_z == 'Gaussian':
        return ds.map(lambda d: {'image': tf.cast(d['image'], dtype=tf.float32) / d['image'].dtype.max,
                                 'label': d['label']},
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif (ds_name == 'mnist' or ds_name == 'svhn_cropped') and px_z == 'Bernoulli':
        return ds.map(lambda d: {'image': bernoulli_sample(d['image']),
                                 'label': d['label']},
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        return ds


def configure_data_set(ds, ds_name, px_z, batch_size, repeats):
    """
    :param ds: TensorFlow data set object
    :param ds_name: name of the data set
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
        ds = pre_process_data(ds, ds_name, px_z)

    # enable prefetch
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load_data_set(ds_name='mnist', px_z='Gaussian', batch_size=100, percent_labelled=None):
    """
    :param ds_name: data set name--call tfds.list_builders() for options
    :param px_z: Decoder modelling assumptions {Gaussian, Bernoulli} that also determine pre-processing
    :param batch_size: training/testing batch size
    :param percent_labelled: percentage of labels to include (None for 0)--will int clamp to {1,2,...,99}
    :return:
        train_ds: TensorFlow Dataset object for the training data
        test_ds: TensorFlow Dataset object for the testing data
        super_ds: TensorFlow Dataset object for the labelled data
        info: data set info object
    """

    # load the test set with info and configure the set
    test_ds, info = tfds.load(name=ds_name, split=tfds.Split.TEST, with_info=True)
    test_ds = configure_data_set(test_ds, ds_name, px_z, batch_size, repeats=1)

    # including labelled data?
    if percent_labelled is not None:
        percent_labelled = int(percent_labelled)
        assert 0 < percent_labelled < 100

        # configure the split
        split = tfds.Split.TRAIN.subsplit([100 - percent_labelled, percent_labelled])

        # load the training set split into two sets (labelled and unlabelled)
        train_ds, label_ds = tfds.load(name=ds_name,
                                       split=split,
                                       as_dataset_kwargs={'shuffle_files': True})

        # configure the data sets
        train_ds = configure_data_set(train_ds, ds_name, px_z, batch_size, repeats=1)
        label_ds = configure_data_set(label_ds, ds_name, px_z, batch_size, repeats=-1)

    # no labelled data
    else:

        # load the test set with info and configure the set
        train_ds = tfds.load(name=ds_name, split=tfds.Split.TRAIN)
        train_ds = configure_data_set(train_ds, ds_name, px_z, batch_size, repeats=1)

        # no labelled training data
        label_ds = None

    # make sure we are all good
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(label_ds, tf.data.Dataset) or label_ds is None
    assert isinstance(test_ds, tf.data.Dataset)

    return train_ds, label_ds, None, test_ds, info


def generate_split(x, y, number_to_split, K, balanced):

    # quick exit
    if number_to_split == 0:
        return x, y, None, None

    # balanced splitting
    if balanced:

        # make sure the split number is divisible by the number of classes
        assert np.mod(number_to_split, K) == 0
        number_to_split = int(number_to_split / K)

        # initialize the split
        x_split = np.zeros([0] + list(x.shape[1:]), dtype=x.dtype)
        y_split = np.zeros([0] + list(y.shape[1:]), dtype=y.dtype)

        # loop over the classes
        for k in range(K):

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


def load_data_set_balanced(data_set_name, px_z, num_validation, num_labelled, balanced, batch_size=100):

    # load training set
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

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
    train_ds = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train})
    train_ds = configure_data_set(train_ds, data_set_name, px_z, batch_size, repeats=1)
    if x_label is not None and y_label is not None:
        label_ds = tf.data.Dataset.from_tensor_slices({'image': x_label, 'label': y_label})
        label_ds = configure_data_set(label_ds, data_set_name, px_z, batch_size, repeats=-1)
    else:
        label_ds = None
    if x_valid is not None and y_valid is not None:
        valid_ds = tf.data.Dataset.from_tensor_slices({'image': x_valid, 'label': y_valid})
        valid_ds = configure_data_set(valid_ds, data_set_name, px_z, batch_size, repeats=1)
    else:
        valid_ds = None
    test_ds = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test})
    test_ds = configure_data_set(test_ds, data_set_name, px_z, batch_size, repeats=1)

    return train_ds, label_ds, valid_ds, test_ds, info


def set_performance(mdl, sess, y_unsupervised_ph, iter_init, perf, loss_type, i, y_supervised_ph=None):
    """
    :param mdl: model class
    :param sess: TensorFlow session
    :param y_unsupervised_ph: TensorFlow placeholder for unseen labels
    :param iter_init: TensorFlow data iterator initializer associated with the unsupervised data
    :param perf: performance dictionary
    :param loss_type: set name (e.g. unsupervised or test)
    :param i: insertion index (i.e. epoch - 1)
    :param y_supervised_ph: TensorFlow placeholder for seen labels
    :return: updated performance dictionary
    """
    # initialize results
    loss = []
    loss_unsupervised = []
    neg_ll_unsupervised = []
    dkl_unsupervised = []
    loss_supervised = []
    neg_ll_supervised = []
    dkl_supervised = []
    alpha_unsupervised = np.zeros([0, mdl.common.K])
    y_unsupervised = np.zeros([0, 1])
    alpha_supervised = np.zeros([0, mdl.common.K])
    y_supervised = np.zeros([0, 1])

    # initialize unsupervised data iterator
    sess.run(iter_init)

    # loop over the batches within the unsupervised data iterator
    print('Evaluating ' + loss_type + ' set performance... ', end='')
    while True:
        try:
            # grab the results
            results = sess.run([mdl.loss,
                                mdl.loss_unsupervised,
                                mdl.neg_ll_unsupervised,
                                mdl.dkl_unsupervised,
                                mdl.loss_supervised,
                                mdl.neg_ll_supervised,
                                mdl.dkl_supervised,
                                mdl.alpha_test,
                                y_unsupervised_ph],
                               feed_dict={mdl.common.M: 1})

            # load metrics
            loss.append(results[0])
            loss_unsupervised.append(results[1])
            neg_ll_unsupervised.append(results[2])
            dkl_unsupervised.append(results[3])
            loss_supervised.append(results[4])
            neg_ll_supervised.append(results[5])
            dkl_supervised.append(results[6])
            alpha_unsupervised = np.concatenate((alpha_unsupervised, results[7]))
            y_unsupervised = np.concatenate((y_unsupervised, np.expand_dims(results[8], axis=1)))

            # if we are in semi-supervised mode, grab supervised classification results
            if y_supervised_ph is not None:
                results = sess.run([mdl.alpha_super_test, y_supervised_ph], feed_dict={mdl.common.M: 1})
                alpha_supervised = np.concatenate((alpha_supervised, results[0]))
                y_supervised = np.concatenate((y_supervised, np.expand_dims(results[1], axis=1)))

        # iterator will throw this error when its out of data
        except tf.errors.OutOfRangeError:
            break

    # new line
    print('')

    # average the results
    perf['loss'][loss_type][i] = sum(loss) / len(loss)
    perf['elbo'][loss_type][i] = sum(loss_unsupervised) / len(loss_unsupervised)
    perf['neg_ll'][loss_type][i] = sum(neg_ll_unsupervised) / len(neg_ll_unsupervised)
    perf['dkl'][loss_type][i] = sum(dkl_unsupervised) / len(dkl_unsupervised)
    perf['elbo']['supervised'][i] = sum(loss_supervised) / len(loss_supervised)
    perf['neg_ll']['supervised'][i] = sum(neg_ll_supervised) / len(neg_ll_supervised)
    perf['dkl']['supervised'][i] = sum(dkl_supervised) / len(dkl_supervised)

    # compute classification accuracy
    perf['class_err'][loss_type][i] = unsupervised_labels(alpha_unsupervised, y_unsupervised, mdl, loss_type)
    if y_supervised_ph is not None:
        perf['class_err']['supervised'][i] =\
            unsupervised_labels(alpha_supervised, y_supervised, mdl, 'supervised')

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


def random_sample_concentration_and_reconstruction_per_class(mdl, sess, y_ph, iter_init):
    """
    :param mdl: model class
    :param sess: TensorFlow session
    :param y_ph: TensorFlow placeholder
    :param iter_init: TensorFlow data iterator initializer associated with the unsupervised data
    :return:
        x: original data
        alpha: concentration parameter
        x_recon: reconstruction
    """
    # get concentration and reconstruction operations
    alpha_op, x_recon_op = mdl.concentration_and_reconstruction()

    # initialize unsupervised data iterator
    sess.run(iter_init)

    # initialize data containers
    x_list = [None] * mdl.common.num_classes
    alpha_list = [None] * mdl.common.num_classes
    x_recon_list = [None] * mdl.common.num_classes

    # loop until we have the concentration and reconstruction for an element of each class
    while sum([x is None for x in x_list]) > 0:

        # grab the results
        x, y, alpha, x_recon = sess.run([mdl.common.x, y_ph, alpha_op, x_recon_op], feed_dict={mdl.common.M: 1})

        # loop over the classes
        for k in range(mdl.common.num_classes):

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
    if mdl.common.K == mdl.common.num_classes:

        # construct y-hat
        y_hat = np.argmax(alpha, axis=1)

        # initialize count matrix
        cnt_mtx = np.zeros([mdl.common.K, mdl.common.K])

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
        for i in range(mdl.common.K):

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


def train(common, method, dim_z, train_set, label_set, valid_set, test_set, n_epochs, early_stop_buffer=15):

    # make sure the method is supported
    assert method in {'Kumaraswamy', 'Softmax', 'KingmaM2'}

    # construct iterator
    iterator = train_set.make_initializable_iterator()
    x, y = iterator.get_next().values()

    # construct initialization operations
    train_iter_init = iterator.make_initializer(train_set)
    if valid_set is not None:
        valid_iter_init = iterator.make_initializer(valid_set)
    else:
        valid_iter_init = iterator.make_initializer(test_set)
    test_iter_init = iterator.make_initializer(test_set)

    # semi-supervised mode
    if label_set is not None:

        # construct one-shot iterator
        super_iter = label_set.make_one_shot_iterator()
        x_super, y_super = super_iter.get_next().values()

    # unsupervised mode
    else:
        x_super = y_super = None

    # link the inputs
    common.link_inputs(x, x_super=x_super, y_super=y_super)

    # construct the model according to method and dim(z)
    if method == 'Kumaraswamy' and dim_z == 0:
        mdl = AutoEncodingDirichlet(common=common)
    elif method == 'Kumaraswamy' and dim_z > 0:
        mdl = AutoEncodingDirichletNormal(common=common, dim_z=dim_z)
    elif method == 'Softmax' and dim_z == 0:
        mdl = AutoEncodingSoftmax(common=common)
    elif method == 'Softmax' and dim_z > 0:
        mdl = AutoEncodingSoftmaxNormal(common=common, dim_z=dim_z)
    elif method == 'KingmaM2' and dim_z > 0:
        mdl = KingmaM2(common=common, dim_z=dim_z)
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # initialize model variables
        sess.run(tf.global_variables_initializer())

        # loop over the number of epochs
        for i in range(n_epochs):

            # get epoch number
            epoch = i + 1

            # start timer
            start = time.time()

            # initialize epoch iterator
            sess.run(train_iter_init)

            # loop over the batches
            while True:
                try:

                    # run just training and loss
                    _, loss = sess.run([mdl.train_op, mdl.loss],
                                       feed_dict={mdl.common.M: mdl.common.monte_carlo_samples})
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
            perf = set_performance(mdl, sess, y, train_iter_init, perf, 'unsupervised', i, y_super)

            # get validation and test set performances
            perf = set_performance(mdl, sess, y, valid_iter_init, perf, 'validation', i)
            perf = set_performance(mdl, sess, y, test_iter_init, perf, 'test', i)

            # plot learning curve
            plot_learning_curve(fig_learn, perf, epoch, mdl.common.mode, save_path=mdl.common.save_dir)

            # plot reconstructed image(s)
            x, alpha, x_recon = random_sample_concentration_and_reconstruction_per_class(mdl, sess, y, test_iter_init)
            mdl.common.plot_random_reconstruction(x, x_recon, alpha, epoch=epoch)

            # plot latent representation
            mdl.plot_latent_representation(sess, epoch=epoch)

            # find the best evidence and classification performances
            i_best_elbo = np.argmin(perf['elbo']['validation'][:epoch])
            i_best_class = np.argmin(perf['class_err']['validation'][:epoch])

            # if the current validation ELBO is the best (and we are saving models), save it!
            if i == i_best_elbo and mdl.common.save_dir is not None:
                pass
                # save_path = mdl.common.saver.save(sess, os.path.join(mdl.common.save_dir, 'best_elbo', 'mdl.ckpt'))
                # print('New best ELBO model! Saving results to ' + save_path)

            # if the current validation classification error is the best (and we are saving models), save it!
            if i == i_best_class and mdl.common.save_dir is not None:
                pass
                # save_path = mdl.common.saver.save(sess, os.path.join(mdl.common.save_dir, 'best_class', 'mdl.ckpt'))
                # print('New best classification model! Saving results to ' + save_path)

            # pause for plot drawing if we aren't saving
            if mdl.common.save_dir is None:
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
    save_performance(perf, epoch, mdl.common.save_dir)


if __name__ == '__main__':

    # pick a data set and modelling assumption
    data_set = 'mnist'
    data_model = 'Gaussian'
    covariance_structure = 'diag'

    # training constants
    n_epochs = 750
    b_size = 250
    drop_out = 0.0

    # load the data set (TensorFlow split method)
    # train_set, label_set, valid_set, test_set, set_info = load_data_set(ds_name=data_set,
    #                                                                     px_z=data_model,
    #                                                                     batch_size=b_size,
    #                                                                     percent_labelled=1)

    # load the data set (custom split method)
    train_set, label_set, valid_set, test_set, set_info = load_data_set_balanced(data_set_name=data_set,
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

    # make the common VAE assignments
    com = AutoEncodingCommon(
            dim_x=list(set_info.features['image'].shape),
            K=10,
            enc_arch=enc_arch,
            dec_arch=dec_arch,
            learn_rate=1e-3,
            px_z=data_model,
            covariance_structure=covariance_structure,
            dropout_rate=drop_out,
            num_classes=set_info.features['label'].num_classes,
            save_dir=None)

    # run training
    train(common=com,
          method='Kumaraswamy',
          dim_z=2,
          train_set=train_set,
          label_set=label_set,
          valid_set=valid_set,
          test_set=test_set,
          n_epochs=n_epochs)

    print('All done!')
    plt.show()
