import math
import os

import pandas as pd


class Config:
    def __init__(self):
        self.split_folder = "./splits/custom_100"
        self.classes = ['T0', 'T1', 'T2', 'T3', 'Tis']


if __name__ == '__main__':
    config = Config()

    splits_0 = pd.read_csv(os.path.join(config.split_folder, "splits_0.csv"), index_col=0)
    splits_0_bool = pd.read_csv(os.path.join(config.split_folder, 'splits_0_bool.csv'), index_col=0)
    splits_0_descriptor = pd.read_csv(os.path.join(config.split_folder, 'splits_0_descriptor.csv'), index_col=0)

    for val_sample in splits_0['val'].dropna():
        splits_0.loc[len(splits_0), 'train'] = val_sample
    for test_sample in splits_0['test'].dropna():
        splits_0.loc[len(splits_0), 'train'] = test_sample

    for index in splits_0['train']:
        if not splits_0_bool.loc[index, 'train']:
            splits_0_bool.loc[index, 'train'] = True

    for cls in config.classes:
        splits_0_descriptor.loc[cls, 'train'] = splits_0_descriptor.loc[cls, 'train'] + \
                                                splits_0_descriptor.loc[cls, 'val'] + \
                                                splits_0_descriptor.loc[cls, 'test']

    splits_0.to_csv(os.path.join(config.split_folder, "splits_0.csv"))
    splits_0_bool.to_csv(os.path.join(config.split_folder, 'splits_0_bool.csv'))
    splits_0_descriptor.to_csv(os.path.join(config.split_folder, 'splits_0_descriptor.csv'))
