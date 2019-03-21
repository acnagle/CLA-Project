import numpy as np
import os
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints full numpy array (no truncation)

    print('\n\t##### EXECUTING ONE_HOT_ENCODING.PY #####\n')

    print('Reading in data ... ')
    # read in arguments from command line
    try:
        data_set_path = str(sys.argv[1])
    except (ValueError, IndexError):
        print('Arguments must be specified as follows:')
        print('python3 one_hot_encoding.py <path-to-data-set>')
        sys.exit(0)

    data_set = np.load(data_set_path)

    print('One-hot encoding categorical features ... ')
    # one-hot encode second and third features.
    for i in range(data_set.shape[0]):
        instance = data_set[i, :]

        feat2 = int(data_set[i, 1])
        feat3 = int(data_set[i, 2])

        feat2_one_hot = np.zeros(3)
        feat3_one_hot = np.zeros(3)

        if feat2 == 1:
            feat2_one_hot[0] = 1
        elif feat2 == 2:
            feat2_one_hot[1] = 1
        elif feat2 == 3:
            feat2_one_hot[2] = 1

        if feat3 == 1:
            feat3_one_hot[0] = 1
        elif feat3 == 2:
            feat3_one_hot[1] = 1
        elif feat3 == 3:
            feat3_one_hot[2] = 1

        instance = np.delete(instance, 2)
        instance = np.delete(instance, 1)

        instance = np.insert(instance, 1, feat3_one_hot)
        instance = np.insert(instance, 1, feat2_one_hot)

        if i == 0:
            data_set_one_hot = instance
        else:
            data_set_one_hot = np.vstack((data_set_one_hot, instance))

    print('Saving the data set ... ')
    path = data_set_path.split('/')
    dest_path = path[0] + '/' + path[1] + '/' + path[2] + '/'

    np.save(dest_path + path[3][:-4] + '_one_hot.npy', data_set_one_hot)


if __name__ == "__main__": main()