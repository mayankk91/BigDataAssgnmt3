import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pickle

def get_preprocessed_train_val_test_data():
    df = pd.read_csv('train.csv')
    data = df.as_matrix()
    data = data[:,1:]
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_data = []
    label_data = []
    for rna_seq, label in data:
        rna_list = []
        for rna in rna_seq:
            rna_list.append(dict.get(rna))
        seq_data.append(rna_list)
        label_data.append(label)
    seq_data = np.array(seq_data)
    label_data = np.array(label_data)
    print(seq_data.shape)

    seq_data, label_data = shuffle(seq_data, label_data)

    train_seq_data = seq_data[0:1440]
    train_label = label_data[0:1440]

    val_seq_data = seq_data[1440:1600]
    val_label = label_data[1440:1600]

    test_seq_data = seq_data[1600:2000]
    test_label = label_data[1600:2000]

    pickle.dump([train_seq_data, train_label, val_seq_data, val_label, test_seq_data, test_label],
                open("preprocessed_train_val_test.pkl", "wb"), protocol=-1)

    print("Completed Preprocessing")
    # return seq_data, label_data

def get_unknown_test_data():
    test_df = pd.read_csv('test.csv')
    test_data = test_df.as_matrix()
    test_data = test_data[:,1:]
    test_seq_data = []
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for test_rna_seq in test_data:
        test_rna_list = []
        for test_rna in test_rna_seq[0]:
            test_rna_list.append(dict.get(test_rna))
        test_seq_data.append(test_rna_list)
    test_seq_data = np.array(test_seq_data)
    print(test_seq_data.shape)
    return test_seq_data

if __name__ == '__main__':
    get_preprocessed_train_val_test_data()