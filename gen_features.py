import datetime
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose
from albumentations.augmentations.transforms import Normalize
from albumentations.augmentations.geometric.resize import Resize
from timm import create_model

from models.resnet_custom import resnet50_baseline


class Config:
    def __init__(self):
        self.patch_folder = "./input/seed_patch/seed_patch_anno_4_256_expand/train"
        self.class_without_anno = ['T0']

        self.model_type = 'beit_large_patch16_512'  # 'resnet50'
        self.device_type = 'cuda:1'
        self.batch_size = 2
        self.transform = Compose(
            [
                Resize(height=512, width=512, interpolation=cv2.INTER_CUBIC, always_apply=True),
                Normalize(mean=(0.84823702, 0.77016022, 0.85043145), std=(0.13220921, 0.20896969, 0.10626152),
                          always_apply=True),
            ]
        )

        self.csv_file_name = "./input/seed_patch/seed_patch_anno_4_256_expand/beit_seed.csv"
        self.feature_output_folder = "./input/seed_patch/seed_patch_anno_4_256_expand/beit_pt_files"


class PatchDataset:
    def __init__(self, patch_folder, class_without_anno):
        self.patch_folder = patch_folder
        self.class_without_anno = class_without_anno
        self.classes, self.patch_dict = self.gather_patches()

    def gather_patches(self):
        classes = [cls for cls in os.listdir(self.patch_folder) if os.path.isdir(os.path.join(self.patch_folder, cls))]
        patch_dict = dict()
        for cls in classes:
            patch_dict[cls] = dict()
            cls_bags = os.listdir(os.path.join(self.patch_folder, cls))
            for bag in cls_bags:
                if cls in self.class_without_anno or "Annotation" in bag:
                    bag_patches = os.listdir(os.path.join(self.patch_folder, cls, bag))
                    patch_dict[cls][bag] = list()
                    for patch_name in bag_patches:
                        patch = cv2.imread(os.path.join(self.patch_folder, cls, bag, patch_name))
                        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                        patch_dict[cls][bag].append(patch)
        return classes, patch_dict


def gen_features(config: Config, patches: list[np.ndarray]) -> np.ndarray:
    """Generate features from patches.

    Args:
        config (Config): Configurations.
        patches (list[np.ndarray]): Patches generated from wsi.

    Returns:
        np.ndarray: Features generated from patches.
    """
    device = torch.device(config.device_type)
    if config.model_type == 'resnet50':
        model = resnet50_baseline(pretrained=True).to(device)
    elif config.model_type == 'beit_large_patch16_512':
        model = create_model('beit_large_patch16_512', pretrained=True).to(device)

        outputs = list()

        def feature_hook(module, input, output):
            outputs.append(output.clone().detach())

        handle = model.fc_norm.register_forward_hook(feature_hook)
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
                if config.model_type == 'resnet50':
                    feature = model(batch)
                elif config.model_type == 'beit_large_patch16_512':
                    model(batch)
                    feature = outputs.pop()

                batch = None
                if features is None:
                    features = feature.cpu()
                else:
                    features = torch.cat([features, feature.cpu()], dim=0)
    return features


def create_none_exist_folder(path: str) -> None:
    """Create folders that don't exist.
    Args:
        path (str): Folder path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    config = Config()

    start_time = time.time()
    patch_dataset = PatchDataset(patch_folder=config.patch_folder, class_without_anno=config.class_without_anno)
    print(f"Created in-memory dataset in {time.time() - start_time}s.")

    create_none_exist_folder(config.feature_output_folder)
    print(f" -> Saving features to {config.feature_output_folder}\n")

    start_time = time.time()
    case_id_list = list()
    slide_id_list = list()
    label_list = list()
    count = 0
    for cls in patch_dataset.patch_dict.keys():
        cls_bags = patch_dataset.patch_dict[cls]
        for bag_id, bag in cls_bags.items():
            print(f"    - Processing on {cls} {bag_id}")
            case_id_list.append(f"patient_{count}")
            slide_id_list.append(bag_id)
            label_list.append(cls)
            features = gen_features(config, bag)
            torch.save(features, os.path.join(config.feature_output_folder, f"{bag_id}.pt"))
            count += 1
    dataset_df = pd.DataFrame({'case_id': case_id_list, 'slide_id': slide_id_list, 'label': label_list})
    dataset_df.to_csv(config.csv_file_name, index=False)

    print(f"\nComplete in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}.")
