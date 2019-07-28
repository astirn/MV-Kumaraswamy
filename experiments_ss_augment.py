import os
import glob
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# import the training utilities
from model_utils import load_data_set, train

# import the architectures
from experiments_ss_run import architectures

if __name__ == '__main__':

    # model assumptions
    data_model = 'Gaussian'
    covariance_structure = 'diag'

    # training constants
    n_epochs = 750
    b_size = 250

    # add parser arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='Nalisnick', help='number of runs')
    parser.add_argument('--data_set', type=str, default='mnist', help='data set name = {mnist, svhn_cropped}')
    parser.add_argument('--num_labelled', type=int, default=600, help='number of labels')
    parser.add_argument('--dim_z', type=int, default=50, help='latent encoding dimensions')

    # parse the arguments
    args = parser.parse_args()
    print('Method = ', args.method)
    print('Data set = ', args.data_set)
    print('Num. labelled = {:d}'.format(args.num_labelled))
    print('Latent dims = {:d}'.format(args.dim_z))

    # get the architecture name
    labels = 'num_labelled_' + str(args.num_labelled)
    arch = list(architectures[args.data_set].keys())[0]

    # set up path components
    path_base = os.path.join(os.getcwd(), 'results_ss_' + args.data_set)
    path_search = os.path.join(labels, 'Kumaraswamy', arch, 'dim_z_' + str(args.dim_z))
    search = glob.glob(os.path.join(path_base, '*', path_search))

    # loop over the search results
    for path in search:

        # get/make directories for this run
        dir_run = path.replace(path_search, '')
        dir_labels = os.path.join(dir_run, 'num_labelled_' + str(args.num_labelled))
        assert os.path.exists(dir_labels)
        dir_method = os.path.join(dir_labels, args.method)
        if not os.path.exists(dir_method):
            os.mkdir(dir_method)
        dir_arch = os.path.join(dir_method, arch)
        if not os.path.exists(dir_arch):
            os.mkdir(dir_arch)
        dir_dim_z = os.path.join(dir_arch, 'dim_z_' + str(args.dim_z))
        if not os.path.exists(dir_dim_z):
            os.mkdir(dir_dim_z)

        # if we already have results, continue
        if os.path.exists(os.path.join(dir_dim_z, 'perf.pkl')):
            print('Skipping', dir_dim_z)
            continue

        # set random seed
        seed = np.uint32(hash(dir_run))
        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        # load the data set (custom split method)
        unlabelled_set, labelled_set, valid_set, test_set, set_info = load_data_set(
            data_set_name=args.data_set,
            px_z=data_model,
            num_validation=10000,
            num_labelled=args.num_labelled,
            balanced=True,
            batch_size=b_size)

        # configure the common VAE elements
        config = {
            'dim_x': list(set_info.features['image'].shape),
            'num_classes': set_info.features['label'].num_classes,
            'dim_z': args.dim_z,
            'K': set_info.features['label'].num_classes,
            'enc_arch': architectures[args.data_set][arch]['enc_arch'],
            'dec_arch': architectures[args.data_set][arch]['dec_arch'],
            'learn_rate': architectures[args.data_set][arch]['learn_rate'],
            'px_z': data_model,
            'covariance_structure': covariance_structure,
            'dropout_rate': 0.0,
            'save_dir': dir_dim_z}

        # run training
        train(method=args.method,
              config=config,
              unlabelled_set=unlabelled_set,
              labelled_set=labelled_set,
              valid_set=valid_set,
              test_set=test_set,
              n_epochs=n_epochs)

        # reset the graph
        tf.reset_default_graph()

        # close all plots
        plt.close('all')
