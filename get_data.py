import numpy as np


def get_data(fname):
    """
    This function reads a csv file and returns a list of lists representing
    the rows and the data in the rows
    """
    with open(fname) as f:
        data = f.read()

    # There are just a couple unknowns (5) in this dataset, they're replaced by 0s
    data = data.replace('?', '0')

    rows = [r.split(',') for r in data.split('\n')]
    return rows


def data_to_arrays(data):
    """
    This converts the list of lists from the csv file to 2 numpy arrays, one
    for benign and one for malicious
    """
    benign = [r[:-1] for r in data if r[-1] == 'B']  # [:-1] to cut off the class label
    malign = [r[:-1] for r in data if r[-1] == 'S']
    # convert the data to numeric representations
    benign = [[int(d) for d in r] for r in benign]
    malign = [[int(d) for d in r] for r in malign]
    return np.array(benign), np.array(malign)


def get_mwr(ben, mal, mwr, ttp):
    """
    :param ben: numpy array of benign samples
    :param mal: numpy array of malicious samples
    :param mwr: float ratio of malware samples to benign samples to train on
    :param ttp: float percent of data to be used for training
    """
    # shuffle the data
    np.random.shuffle(ben)
    np.random.shuffle(mal)
    # cut off the benign to get the ratio right (use all malware)
    mal_len = len(mal)
    ben_len = int((1/mwr) * mal_len - mal_len)
    new_ben = ben[:ben_len]
    # divide into train and test
    mal_idx = int(mal_len * ttp)
    ben_idx = int(ben_len * ttp)
    mal_train = mal[:mal_idx]
    mal_test = mal[mal_idx:]
    ben_train = new_ben[:ben_idx]
    ben_test = new_ben[ben_idx:]
    return ben_train, mal_train, ben_test, mal_test


if __name__ == '__main__':
    dat = get_data('/scratch/sutay/malware_data/drebin-215-dataset-5560malware-9476-benign.csv')
    ben, mal = data_to_arrays(dat)
    print(ben.shape, mal.shape)
    b_tr, m_tr, b_ts, m_ts = get_mwr(ben, mal, 0.4, 0.8)
    print(b_tr.shape, m_tr.shape, b_ts.shape, m_ts.shape)
    print(f'train malware : train total = {m_tr.shape[0]/(b_tr.shape[0]+m_tr.shape[0])}\n'
          f'test malware : test total = {m_ts.shape[0]/(b_ts.shape[0]+m_ts.shape[0])}\n'
          f'train total : total = {(m_tr.shape[0]+b_tr.shape[0])/(m_ts.shape[0]+b_ts.shape[0]+m_tr.shape[0]+b_tr.shape[0])}')

