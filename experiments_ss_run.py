import os
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# import the common Auto-Encoder object
from models_proposed import AutoEncodingCommon

# import the training utilities
from model_utils import load_data_set_balanced, train

# define the methods
methods = {'Kumaraswamy', 'Softmax', 'KingmaM2'}

# define the architectures
architectures = {
    'mnist': {
        '2c_2d': {
            'enc_arch': {'conv': [{'k_size': 5, 'out_chan': 5}, {'k_size': 3, 'out_chan': 10}],
                         'full': [200, 200]},
            'dec_arch': {'full': [200, 200],
                         'conv_start_chans': 10,
                         'conv': [{'k_size': 3, 'out_chan': 5}, {'k_size': 5, 'out_chan': 1}]},
            'learn_rate': 1e-3
        },
    },
    'svhn_cropped': {
        '2c_2d': {
            'enc_arch': {'conv': [{'k_size': 5, 'out_chan': 15}, {'k_size': 3, 'out_chan': 30}],
                         'full': [200, 200]},
            'dec_arch': {'full': [200, 200],
                         'conv_start_chans': 30,
                         'conv': [{'k_size': 3, 'out_chan': 15}, {'k_size': 5, 'out_chan': 3}]},
            'learn_rate': 1e-4
        },
    },
}

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
    parser.add_argument('--num_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--data_set', type=str, default='mnist', help='data set name = {mnist, svhn_cropped}')
    parser.add_argument('--num_labelled', type=int, default=600, help='number of labels')
    parser.add_argument('--dim_z', type=int, default=50, help='data set name = {mnist, svnh}')

    # parse the arguments
    args = parser.parse_args()
    print('Num. runs = {:d}'.format(args.num_runs))
    print('Data set = ', args.data_set)
    print('Num. labelled = {:d}'.format(args.num_labelled))
    print('Latent dims = {:d}'.format(args.dim_z))

    # if results directory doesn't yet exist for this data set, make one
    dir_results = os.path.join(os.getcwd(), 'results_ss_' + args.data_set)
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)

    # loop over the runs
    cnt = 0
    for _ in range(args.num_runs):

        # make a fresh directory for this run
        dir_run = str(time.time())
        dir_run = os.path.join(dir_results, dir_run[:dir_run.index('.')])
        os.mkdir(dir_run)

        # make the method directory
        dir_labels = os.path.join(dir_run, 'num_labelled_' + str(args.num_labelled))
        os.mkdir(dir_labels)

        # loop over the methods
        for method in methods:

            # make the method directory
            dir_method = os.path.join(dir_labels, method)
            os.mkdir(dir_method)

            # loop over the architectures
            for arch in architectures[args.data_set]:

                # make the architecture directory
                dir_arch = os.path.join(dir_method, arch)
                os.mkdir(dir_arch)

                # skip Kingma M2 method if dim(z) == 0 since that model does not support this configuration
                if method == 'KingmaM2' and args.dim_z == 0:
                    continue

                # make the latent dimension directory
                dir_dim_z = os.path.join(dir_arch, 'dim_z_' + str(args.dim_z))
                os.mkdir(dir_dim_z)

                # print update
                cnt += 1
                print('\n' + '*' * 100)
                print('num labels =', args.num_labelled,
                      '| method =', method,
                      '| arch =', arch,
                      '| dim_z =', args.dim_z)

                # set random seed
                seed = np.uint32(hash(dir_run))
                np.random.seed(seed)
                tf.random.set_random_seed(seed)

                # load the data set (custom split method)
                train_set, label_set, valid_set, test_set, set_info = load_data_set_balanced(
                    data_set_name=args.data_set,
                    px_z=data_model,
                    num_validation=10000,
                    num_labelled=args.num_labelled,
                    balanced=True,
                    batch_size=b_size)

                # make the common VAE assignments
                com = AutoEncodingCommon(
                    dim_x=list(set_info.features['image'].shape),
                    K=set_info.features['label'].num_classes,
                    enc_arch=architectures[args.data_set][arch]['enc_arch'],
                    dec_arch=architectures[args.data_set][arch]['dec_arch'],
                    learn_rate=architectures[args.data_set][arch]['learn_rate'],
                    px_z=data_model,
                    covariance_structure=covariance_structure,
                    dropout_rate=0.0,
                    save_dir=dir_dim_z)

                # run training
                train(common=com,
                      method=method,
                      dim_z=args.dim_z,
                      train_set=train_set,
                      label_set=label_set,
                      valid_set=valid_set,
                      test_set=test_set,
                      n_epochs=n_epochs)

                # reset the graph
                tf.reset_default_graph()

                # close all plots
                plt.close('all')

    print('\n' + '*' * 100)
    print('Completed {:d} trainings!'.format(cnt))
