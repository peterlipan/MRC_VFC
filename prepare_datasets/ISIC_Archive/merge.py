import os
import pandas as pd
from sklearn.model_selection import train_test_split


root = "/mnt/ssd/li/ISIC_Archive"
split_root = "/home/panli/Ours/split/ISIC_Archive"
diagnosis = {'NV': 12875, 'MEL': 4522, 'BCC': 3393, 'SK': 1464,
             'AK': 869, 'SCC': 656, 'BKL': 384, 'SL': 270,
             'VASC': 253, 'DF': 246, 'LK': 32, 'LS': 27,
             'AN': 15, 'AMP': 14}
train_df = pd.DataFrame()
test_df = pd.DataFrame()
val_df = pd.DataFrame()
random_seed = 13


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


def check_labels(df, metadata, diagnosis):
    """
    This is to check whether the label of each image is correct.
    :param df: DataFrame including image ids and the label (one-hot)
    :param metadata: official metadata formate of ISIC of each class.
    :return:
    """
    df_imgs = set(df[df[diagnosis] == 1]['image'])
    metadata_imgs = set(metadata['isic_id'].values)
    assert df_imgs == metadata_imgs


print("merging metadata...")
# merge the metadata of all classes
for item in diagnosis.keys():
    csv_path = os.path.join(root, item + '.csv')
    assert os.path.exists(csv_path)

    metadata = pd.read_csv(csv_path)
    assert diagnosis[item] == metadata.shape[0]
    ids = metadata['isic_id'].values

    temp_df = pd.DataFrame({'image': ids, 'diagnosis': item})
    # split the dataset here
    # to make sure each class will appear in each subset
    temp_train_df, temp = train_test_split(temp_df, test_size=0.3, random_state=random_seed)
    temp_test_df, temp_val_df = train_test_split(temp, test_size=1 / 3, random_state=random_seed)

    train_df = pd.concat([train_df, temp_train_df])
    test_df = pd.concat([test_df, temp_test_df])
    val_df = pd.concat([val_df, temp_val_df])

# shuffle the rows
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)

# one-hot encoding
print("One-hot encoding...")
train_df = pd.get_dummies(train_df, columns=['diagnosis'], prefix='', prefix_sep='')
test_df = pd.get_dummies(test_df, columns=['diagnosis'], prefix='', prefix_sep='')
val_df = pd.get_dummies(val_df, columns=['diagnosis'], prefix='', prefix_sep='')

# check if there exist training sample leakage
check_leakage(train_df, test_df, val_df)
# check if the label of each image is correct
df = pd.concat([train_df, test_df, val_df])
for item in diagnosis.keys():
    csv_path = os.path.join(root, item + '.csv')
    metadata = pd.read_csv(csv_path)
    check_labels(df, metadata, item)

if not os.path.exists(split_root):
    os.makedirs(split_root)
train_df.to_csv(os.path.join(split_root, "training.csv"), index=False)
test_df.to_csv(os.path.join(split_root, "testing.csv"), index=False)
val_df.to_csv(os.path.join(split_root, "validation.csv"), index=False)
