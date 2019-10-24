import os
import pickle
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind_from_stats

# import iterators from experiments run file
from experiments_ss_run import methods, architectures

# add parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, default='svhn_cropped', help='data set name = {mnist, svhn_cropped}')
parser.add_argument('--dir_prefix', type=str, default='new_results_ss_', help='results directory prefix')
args = parser.parse_args()

# default modeling assumptions
data_model = 'Gaussian'
covariance_structure = 'diag'

# base directory
base_dir = args.dir_prefix + args.data_set

# find all labellings
num_labelled = [str(x) for x in Path(os.path.join(os.getcwd(), base_dir)).glob('**/num_labelled_*/')]
num_labelled = np.unique([int(os.path.split(x)[1].replace('num_labelled_', '')) for x in num_labelled])

# find all latent dimensions
z_dims = [str(x) for x in Path(os.path.join(os.getcwd(), base_dir)).glob('**/dim_*/')]
z_dims = np.unique([int(os.path.split(x)[1].replace('dim_z_', '')) for x in z_dims])

# arrange methods
methods = list(methods)
pop = methods.pop(methods.index('Kumaraswamy'))
methods.sort()
methods.insert(0, pop)

# loop over the number of labels
for num_labels in num_labelled:

    # loop over the latent dimensions
    for dim_z in z_dims:

        # loop over the architectures
        for arch in architectures[args.data_set]:

            print('*' * 100)

            # loop over the methods
            t_test_dict = {}
            for method in methods:

                # skip Kingma M2 method if dim(z) == 0 since that model does not support this configuration
                if method == 'KingmaM2' and dim_z == 0:
                    continue

                # put together the folder string for results of this type
                folder_str = os.path.join('num_labelled_{:d}'.format(num_labels),
                                          method,
                                          arch,
                                          'dim_z_{:d}'.format(dim_z))

                # find all results for this configuration
                result_dirs = Path(os.path.join(os.getcwd(), base_dir)).glob(os.path.join('**', folder_str))
                result_dirs = [str(x) for x in result_dirs]

                # accumulate the results
                class_err = []
                log_ll = []
                for result_dir in result_dirs:
                    pkl_path = os.path.join(result_dir, 'perf.pkl')
                    if not os.path.exists(pkl_path):
                        continue
                    with open(pkl_path, 'rb') as f:
                        perf = pickle.load(f)
                    class_err.append(perf['class_err']['test'][np.argmin(perf['class_err']['validation'])])
                    log_ll.append(-perf['neg_ll']['test'][np.argmin(perf['neg_ll']['validation'])])

                # print update
                print('N = {:2d}'.format(len(class_err)),
                      'Class. Error = {:.3f} +- {:.3f}'.format(np.mean(class_err), np.std(class_err, ddof=1)),
                      'Log LL = {:9.2f} +- {:8.2f}'.format(np.mean(log_ll), np.std(log_ll, ddof=1)),
                      folder_str)

                # save results for t-test
                t_test_dict.update({method: {'err': {'mean': np.mean(class_err),
                                                     'std': np.std(class_err, ddof=1),
                                                     'N': len(class_err)},
                                             'll': {'mean': np.mean(log_ll),
                                                    'std': np.std(log_ll, ddof=1),
                                                    'N': len(log_ll)}}})

            # compute t-test results
            for method in t_test_dict.keys():
                if method == 'Kumaraswamy':
                    continue
                for metric in t_test_dict['Kumaraswamy'].keys():

                    t, p = ttest_ind_from_stats(mean1=t_test_dict['Kumaraswamy'][metric]['mean'],
                                                std1=t_test_dict['Kumaraswamy'][metric]['std'],
                                                nobs1=t_test_dict['Kumaraswamy'][metric]['N'],
                                                mean2=t_test_dict[method][metric]['mean'],
                                                std2=t_test_dict[method][metric]['std'],
                                                nobs2=t_test_dict[method][metric]['N'],
                                                equal_var=True)
                    if p < 0.01:
                        print(method, 'p-value for', metric, '= {:.2e}'.format(p).replace('e', '\\times 10^{') + '}')
                    else:
                        print(method, 'p-value for', metric, '= {:.2f}'.format(p))

