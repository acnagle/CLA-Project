import numpy as np
import os
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints full numpy array (no truncation)

    print('\n\t##### EXECUTING REMOVE_CATEGORICAL_FEATURES.PY #####\n')

    print('Reading in data ... ')
    # read in arguments from command line
    try:
        data_set_path = str(sys.argv[1])
    except (ValueError, IndexError):
        print('Arguments must be specified as follows:')
        print('python3 Remove_Categorical_Features.py <path-to-data-set>')
        sys.exit(0)

    data_set = np.load(data_set_path)

    print('Removing categorical features ... ')
    # remove second and third features.
    for i in range(data_set.shape[0]):
        instance = data_set[i, :]

        instance = np.delete(instance, 2)
        instance = np.delete(instance, 1)

        if i == 0:
            data_set_no_categorical = instance
        else:
            data_set_no_categorical = np.vstack((data_set_no_categorical, instance))

    print('Saving the data set ... ')
    path = data_set_path.split('/')
    dest_path = path[0] + '/' + path[1] + '/' + path[2] + '/'

    np.save(dest_path + path[3][:-4] + '_no_categorical.npy', data_set_no_categorical)


if __name__ == "__main__": main()