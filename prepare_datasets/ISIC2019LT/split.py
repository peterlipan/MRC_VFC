import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def check_leakage(train, test, val):
    """
    TO check if there exist data leakage between train and test/val sets.
    :param train: pandas.DataFrame
    :param test: pandas.DataFrame
    :param val: pandas.DataFrame
    :return:
    """
    assert all(item is False for item in train.duplicated()), \
        'Duplication in train set'
    assert all(item is False for item in test.duplicated()), \
        'Duplication in test set'
    assert all(item is False for item in val.duplicated()), \
        'Duplication in validation set'
    temp = pd.concat([train, test, val])
    assert all(item is False for item in temp.duplicated()), \
        'Data leakage!!!'


def construct_LT_dataset(df, imbalance_factor, random_seed):
    """
    Construct a long-tailed version of the input dataset,
    according to https://arxiv.org/abs/1901.05555
    :param df: pandas.DataFrame with columns = [ANY, class_1, ..., class_n]
    :return: train_df, test_df, val_df: long-tailed version of the input dataframe
    """
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    val_df = pd.DataFrame()
    names = df.columns[1:]
    # calculate the original number of each class
    counts = np.sum(df.iloc[:, 1:].values, axis=0)
    idx = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    counts_sorted = sorted(counts, reverse=True)
    counts_lt = np.zeros_like(counts_sorted, dtype=int)
    names_lt = names[idx]
    # imbalance factor to mu
    mu = (counts_sorted[0] / (counts_sorted[-1] * imbalance_factor)) ** (1 / (len(counts) - 1))
    # LT version
    for i in range(len(counts_sorted)):
        # n_LT = n_i * u ** i
        counts_lt[i] = np.ceil(counts_sorted[i] * mu ** i)
    name2count = {names_lt[i]: counts_lt[i] for i in range(len(names))}

    for item in name2count.keys():
        class_df = df[df[item] == 1]
        num = name2count[item]
        # random select num samples from class_df
        class_df = class_df.sample(n=num, random_state=random_seed)
        # split the dataset here
        # to make sure each class will appear in each subset
        temp_train_df, temp = train_test_split(class_df, test_size=0.3, random_state=random_seed)
        temp_test_df, temp_val_df = train_test_split(temp, test_size=1 / 3, random_state=random_seed)

        train_df = pd.concat([train_df, temp_train_df])
        test_df = pd.concat([test_df, temp_test_df])
        val_df = pd.concat([val_df, temp_val_df])

    # shuffle the rows
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return train_df, test_df, val_df


def construct_ISIC2019LT(imbalance_factor, data_root, csv_file_root, random_seed):
    # load ISIC2019 ground truth csv file
    csv = pd.read_csv(os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv"))
    # drop the last column (UNK)
    csv = csv.iloc[:, :-1]
    # construct the long-tailed version
    train_df, test_df, val_df = construct_LT_dataset(csv, imbalance_factor, random_seed)
    # check if exist data leakage
    check_leakage(train_df, test_df, val_df)

    if not os.path.exists(csv_file_root):
        os.makedirs(csv_file_root)
    train_df.to_csv(os.path.join(csv_file_root, "training.csv"), index=False)
    test_df.to_csv(os.path.join(csv_file_root, "testing.csv"), index=False)
    val_df.to_csv(os.path.join(csv_file_root, "validation.csv"), index=False)

