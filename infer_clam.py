import datetime
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.augmentations.transforms import Normalize
from torch import nn

from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline


class Config:
    def __init__(self):
        self.log_level = 2

        self.model_type = 'CLAM_MB'
        self.drop_out = True
        self.n_classes = 5
        self.model_size = 'big'
        self.subtyping = True
        self.B = 8
        self.inst_loss = 'svm'
        self.weight_file = "./results/seed_clam_mb_big_no_es_no_fold_expand_dataset_epoch5000_s42/s_0_epoch4000_checkpoint.pt"

        self.submission_csv = "./input/seed_patch/seed_patch_anno_4_256_expand/Submission_Sample.csv"
        self.patch_folder = "./input/seed_patch/seed_patch_anno_4_256_expand/test"

        self.transform = Normalize(mean=(0.84823702, 0.77016022, 0.85043145), std=(0.13220921, 0.20896969, 0.10626152),
                                   always_apply=True)
        self.batch_size = 64

        self.result_folder = "./results/seed_clam_mb_big_no_es_no_fold_expand_dataset_epoch5000_s42"
        self.result_file_name = "result_0_epoch4000.csv"

        self.device = torch.device('cuda:0')
        self.feature_extraction_device = torch.device('cuda:3')


def create_model(config):
    if config.log_level >= 1:
        print(f"    - Create {config.model_type} model")

    model_dict = {"dropout": config.drop_out, 'n_classes': config.n_classes}
    if config.model_size is not None and config.model_type != 'mil':
        model_dict.update({"size_arg": config.model_size})
    if config.subtyping:
        model_dict.update({'subtyping': True})
    if config.B > 0:
        model_dict.update({'k_sample': config.B})
    if config.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM

        instance_loss_fn = SmoothTop1SVM(n_classes=2)
        if config.device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()
    if config.log_level == 2:
        for key, val in model_dict.items():
            print(f"        - {key}: {val}")

    if config.model_type == 'CLAM_SB':
        model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    elif config.model_type == 'CLAM_MB':
        model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)

    if config.log_level >= 1:
        print(f"    - Loading weights from {config.weight_file}")
    model.load_state_dict(torch.load(config.weight_file))

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(config.device)
    return model


class TestPatchDataset:
    def __init__(self, patch_folder, patch_index_df):
        self.patch_folder = patch_folder
        self.patch_index = patch_index_df
        self.patch_dict = self.gather_patches()

    def gather_patches(self):
        patch_dict = dict()
        for idx in self.patch_index['slide_id']:
            patch_dict[idx] = list()
            patch_list = os.listdir(os.path.join(self.patch_folder, idx))
            for patch in patch_list:
                patch = cv2.imread(os.path.join(self.patch_folder, idx, patch))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_dict[idx].append(patch)
        return patch_dict


def gen_features(config: Config, patches: list[np.ndarray]) -> np.ndarray:
    """Generate features from patches.

    Args:
        config (Config): Configurations.
        patches (list[np.ndarray]): Patches generated from wsi.

    Returns:
        np.ndarray: Features generated from patches.
    """
    device = config.feature_extraction_device
    model = resnet50_baseline(pretrained=True).to(device)
    model.eval()

    features = None
    batch = None
    with torch.no_grad():
        for idx, patch in enumerate(patches):
            patch = config.transform(image=patch)['image']
            patch = patch.transpose(2, 0, 1)
            patch = np.expand_dims(patch, axis=0)
            patch = torch.from_numpy(patch).float().to(device)
            if batch is None:
                batch = patch
            else:
                batch = torch.cat([batch, patch], dim=0)
            if batch.shape[0] == config.batch_size or (idx + 1) == len(patches):
                feature = model(batch)
                batch = None
                if features is None:
                    features = feature.cpu()
                else:
                    features = torch.cat([features, feature.cpu()], dim=0)
    return features


if __name__ == "__main__":
    config = Config()

    start_time = time.time()
    if config.log_level >= 1:
        print(" -> Initializing inferring protocol")
    mil_model = create_model(config)
    mil_model.eval()

    if config.log_level >= 1:
        print("    - Reading submission csv")
    submission_df = pd.read_csv(config.submission_csv, header=None, names=['slide_id'])
    result_df = pd.DataFrame({"result": [-1 for _ in range(len(submission_df['slide_id']))]})
    result_df = pd.concat([submission_df, result_df], axis=1)

    if config.log_level >= 1:
        print("    - Creating in-memory dataset")
    dataset_creation_start_time = time.time()
    dataset = TestPatchDataset(patch_folder=config.patch_folder, patch_index_df=result_df)
    if config.log_level == 2:
        print(f"        - Created in {time.time() - dataset_creation_start_time}s")

    if config.log_level >= 1:
        print("\n -> Executing inferring protocol")
    for count, idx in enumerate(result_df['slide_id']):
        if config.log_level >= 1:
            print(f"    - Inferring on {idx}    [{count + 1}/{len(result_df['slide_id'])}]", end='')
        patches = dataset.patch_dict[idx]
        features = gen_features(config, patches)
        with torch.no_grad():
            features = features.to(config.device)
            _, _, result, _, _ = mil_model(features)
            result = int(result.cpu().numpy().squeeze())
            if config.log_level >= 1:
                print(f" Class: {result}")
            result_df.loc[result_df['slide_id'] == idx, 'result'] = result

    if config.log_level >= 1:
        print(f"\n -> Saving results to {os.path.join(config.result_folder, config.result_file_name)}")
    result_df.to_csv(os.path.join(config.result_folder, config.result_file_name), index=False, encoding='utf-8',
                     header=False)

    if config.log_level >= 1:
        print(f"\n -> Inferring protocol completed in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
