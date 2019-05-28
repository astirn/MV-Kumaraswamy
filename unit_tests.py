import time
import numpy as np
import tensorflow as tf

from model_utils import load_data_set_balanced


def get_data_split(session, x_batch, y_batch, iterator_init):

    # initialize iterator
    session.run(iterator_init)

    # accumulate batch data
    x = np.zeros([0, 28, 28, 1])
    y = np.zeros([0])
    while True:
        try:
            data = session.run([x_batch, y_batch])
            x = np.concatenate((x, data[0]))
            y = np.concatenate((y, data[1]))
        except tf.errors.OutOfRangeError:
            break

    return x, y


def get_data(seed):

    # set random seed
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    # load the data set (custom split method)
    train_set, _, valid_set, test_set, _ = load_data_set_balanced(
        data_set_name='mnist',
        px_z='Gaussian',
        num_validation=10000,
        num_labelled=600,
        balanced=True,
        batch_size=250)

    # construct iterator
    iterator = train_set.make_initializable_iterator()
    x_batch, y_batch = iterator.get_next().values()

    # construct initialization operations
    train_iter_init = iterator.make_initializer(train_set)
    valid_iter_init = iterator.make_initializer(valid_set)
    test_iter_init = iterator.make_initializer(test_set)

    # start a monitored session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # get the data
        x_train, y_train = get_data_split(sess, x_batch, y_batch, train_iter_init)
        x_valid, y_valid = get_data_split(sess, x_batch, y_batch, valid_iter_init)
        x_test, y_test = get_data_split(sess, x_batch, y_batch, test_iter_init)

    # reset the graph
    tf.reset_default_graph()

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# set the seed
SEED = np.uint32(hash(time.time()))

# get the data
data = get_data(SEED)

# run the test several times
for i in range(5):

    # get the data again
    data_duplicate = get_data(SEED)

    # make sure its the same as the original
    for j in range(len(data)):
        assert (np.min(np.abs(data[j] - data_duplicate[j]))) == 0

print('Assumptions correct!')
