import datetime
import math
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch


class Config:
    def __init__(self):
        self.seed = 42
        self.data_root = "./input/seed_patch/seed_patch_anno_4_256"
        self.data_csv = "seed.csv"
        self.pt_folder = "pt_files"
        self.sample_class = 'T0'
        self.bag_num = 58
        self.instance_sample_ratio = 0.5
        self.min_instances = 8

        self.output_data_root = "./input/seed_vdataset"


def seed_everything(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_none_exist_folder(path: str) -> None:
    """Create folders that don't exist.
    Args:
        path (str): Folder path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    print(" -> Creating virtual dataset\n")
    start_time = time.time()

    config = Config()
    seed_everything(seed=config.seed)

    create_none_exist_folder(os.path.join(config.output_data_root, config.pt_folder))

    data_df = pd.read_csv(os.path.join(config.data_root, config.data_csv))
    sample_df = data_df.loc[data_df['label'] == config.sample_class]
    keep_df = data_df.loc[data_df['label'] != config.sample_class]
    sample_df = sample_df.sample(n=config.bag_num, random_state=config.seed)
    keep_df = pd.concat([sample_df, keep_df], ignore_index=True)
    keep_df.drop_duplicates()

    for slide_id, label in zip(keep_df['slide_id'], keep_df['label']):
        print(f"    - Select {label} {slide_id}", end='')
        if label == config.sample_class:
            features = torch.load(os.path.join(config.data_root, config.pt_folder, f"{slide_id}.pt"))
            indices = torch.tensor(
                random.sample(range(len(features)), k=math.ceil(len(features) * config.instance_sample_ratio)))
            features = features[indices, :]
            torch.save(features, os.path.join(config.output_data_root, config.pt_folder, f"{slide_id}.pt"))
            print(f"    {math.ceil(len(features) * config.instance_sample_ratio)} samples saved")
        else:
            features = torch.load(os.path.join(config.data_root, config.pt_folder, f"{slide_id}.pt"))
            if features.shape[0] < config.min_instances:
                keep_df = keep_df.loc[keep_df['slide_id'] != slide_id]
                print(f"    {features.shape[0]} sample(s) dropped")
            else:
                shutil.copy(os.path.join(config.data_root, config.pt_folder, f"{slide_id}.pt"),
                            os.path.join(config.output_data_root, config.pt_folder, f"{slide_id}.pt"))
                print(f"    {features.shape[0]} samples saved")

    keep_df.to_csv(os.path.join(config.output_data_root, config.data_csv), index=False)

    print(f"\nComplete in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}.")
