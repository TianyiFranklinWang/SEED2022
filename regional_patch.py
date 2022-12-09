import datetime
import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import skimage
import tifffile as tiff


class Config:
    def __init__(self):
        self.image_folder = "./input/seed_processed_data"
        self.classes = ['T0', 'T1', 'T2', 'T3', 'Tis', 'test']
        self.annotated_classes = ['T1', 'T2', 'T3', 'Tis', 'test']
        self.no_thresh_classes = ['T1', 'T2', 'T3', 'Tis', 'test']  # Some WSIs has black marks which jam the threshold method
        self.annotation_folder_postfix = '_json'
        self.sample_dict = dict()
        self.annotation_dict = dict()

        self.max_workers = 2
        self.work_group_size = 2

        self.scale = 4
        self.patch_size = 256
        self.thresh_method = 'otsu'  # or 'adaptive'
        self.area_threshold = 16384
        self.min_size = 16384
        self.connectivity = 8

        self.output_folder = f"./input/seed_patch/seed_patch_anno_{self.scale}_{self.patch_size}_expand"

        self.gather_sample_and_annotation()

    def gather_sample_and_annotation(self):
        for cls in self.classes:
            self.sample_dict[cls] = os.listdir(os.path.join(self.image_folder, cls))
            if cls in self.annotated_classes:
                self.annotation_dict[cls] = os.listdir(
                    os.path.join(self.image_folder, cls + self.annotation_folder_postfix))
            else:
                self.annotation_dict[cls] = dict()


def create_none_exist_folder(path: str) -> None:
    """Create folders that don't exist.
    Args:
        path (str): Folder path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def resize_wsi(wsi: np.ndarray, scale: int) -> np.ndarray:
    """Resizes the wsi to the given size.
    Args:
        wsi (np.ndarray): wsi.
        scale (int): Relevant scale factor.
    Returns:
        np.ndarray: Resized wsi.
    """
    dsize = (wsi.shape[1] // scale, wsi.shape[0] // scale)
    wsi = cv2.resize(wsi, dsize=dsize, fx=0, fy=0, interpolation=cv2.INTER_AREA)
    return wsi


def pad_wsi(config: Config, wsi: np.ndarray, pad_value: int) -> np.ndarray:
    """Pad the wsi in order to be dividable.
    Args:
        config (Config): Configurations.
        wsi (np.ndarray): Wsi to be pad.
        pad_value (int): Padding value.
    Returns:
        np.ndarray: Padded wsi.
    """
    scaled_shape = wsi.shape
    pad_size = config.patch_size
    pad0, pad1 = (int(pad_size - (scaled_shape[0] % pad_size)),
                  int(pad_size - (scaled_shape[1] % pad_size)))
    if len(scaled_shape) == 3:
        wsi = np.pad(wsi, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                     constant_values=pad_value)
    elif len(scaled_shape) == 2:
        wsi = np.pad(wsi, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2]],
                     constant_values=pad_value)
    return wsi


def extract_contours(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    contours = list()
    for context in data['contexts']:
        contour = list()
        points = context['points']
        for point in points:
            contour.append([int(point['x']), int(point['y'])])
        contour = np.asarray(contour, dtype=np.int64)
        contours.append(contour)
    return contours


def gen_contour_mask(mask_shape, contours, thickness=-1):
    if contours is None:
        mask = np.ones(shape=mask_shape, dtype=np.uint8)
    else:
        mask = np.zeros(shape=mask_shape, dtype=np.uint8)
        for cnt in contours:
            mask = cv2.drawContours(mask, [cnt], 0, (1, 1, 1), thickness)
    return mask[:, :, 0]


def pad_to_contour(wsi, contours, pad_value=255):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for cnt in contours:
        for point in cnt:
            x, y = point
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    pad_x1 = 0
    pad_x2 = 0
    pad_y1 = 0
    pad_y2 = 0
    if min_x < 0:
        pad_x1 = -min_x
    if wsi.shape[1] < max_x:
        pad_x2 = max_x - wsi.shape[1]
    if min_y < 0:
        pad_y1 = -min_y
    if wsi.shape[0] < max_y:
        pad_y2 = max_y - wsi.shape[0]
    wsi = np.pad(wsi, [[pad_y1, pad_y2], [pad_x1, pad_x2], [0, 0]], constant_values=pad_value)

    return wsi


def thresh_wsi(config: Config, wsi: np.ndarray) -> np.ndarray:
    """Apply thresholding to the wsi.
    Args:
        config (Config): Configurations.
        wsi (np.ndarray): Wsi to be threshed.
    Returns:
        np.ndarray: Threshed wsi.
    """
    gray_scaled_wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2GRAY)
    blured_scaled_wsi = cv2.medianBlur(gray_scaled_wsi, 3)
    if config.thresh_method == "adaptive":
        threshed_wsi = cv2.adaptiveThreshold(blured_scaled_wsi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 21, 8)
    elif config.thresh_method == "otsu":
        _, threshed_wsi = cv2.threshold(blured_scaled_wsi, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    else:
        raise AttributeError(f"No thresh method named {config.thresh_method}")
    threshed_wsi = skimage.morphology.remove_small_holes(threshed_wsi > 0, area_threshold=config.area_threshold,
                                                         connectivity=config.connectivity)
    threshed_wsi = skimage.morphology.remove_small_objects(threshed_wsi, min_size=config.min_size,
                                                           connectivity=config.connectivity)
    return threshed_wsi.astype(np.uint8)


def gen_patch(wsi: np.ndarray, patch_size: int) -> np.ndarray:
    """Generate Patches from wsi of given size.
    Args:
        wsi (np.ndarray): Wsi to be processed.
        patch_size (int): Size of a patch.
    Returns:
        np.ndarray: Patches of given size.
    """
    shape = wsi.shape
    if len(shape) == 2:
        patches = wsi.reshape(shape[0] // patch_size, patch_size,
                              shape[1] // patch_size, patch_size)
        patches = patches.transpose(0, 2, 1, 3)
        patches = patches.reshape(-1, patch_size, patch_size)
    elif len(shape) == 3:
        patches = wsi.reshape(shape[0] // patch_size, patch_size,
                              shape[1] // patch_size, patch_size, 3)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, patch_size, patch_size, 3)
    return patches


def select_patch(config: Config, mask_patches: np.ndarray, thresh_patches: np.ndarray) -> list[int]:
    """Select out patches that contain information.
    Args:
        config (Config): Configurations.
        mask_patches (np.ndarray): Patches from mask.
        thresh_patches (np.ndarray): Patches from threshed wsi.
    Returns:
        list[int]: List of selected indices.
    """
    selected_ids = list()
    for idx, thresh_patch in enumerate(thresh_patches):
        if thresh_patch.sum() > 0:
            if mask_patches is not None:
                mask_patch = mask_patches[idx]
                if mask_patch[config.patch_size // 2, config.patch_size // 2]:
                    selected_ids.append(idx)
            else:
                selected_ids.append(idx)
    return selected_ids


def save_patches(config, cls, sample_name, wsi_patches, select_ids, anno_id=None):
    if len(select_ids) > 0:
        if anno_id is None:
            sample_folder = os.path.join(config.output_folder, cls, sample_name.split('.')[0])
        else:
            sample_folder = os.path.join(config.output_folder, cls, sample_name.split('.')[0] + f'_Annotation{anno_id}')
        create_none_exist_folder(sample_folder)
        for idx in select_ids:
            wsi_patches[idx] = cv2.cvtColor(wsi_patches[idx], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(sample_folder, f"{sample_name.split('.')[0]}_{idx}.png"), wsi_patches[idx])


def patch_main(config, cls, sample_name):
    sample_path = os.path.join(config.image_folder, cls, sample_name)
    wsi = tiff.imread(sample_path)

    contours = None
    if sample_name.split('.')[0] + '.json' in config.annotation_dict[cls]:
        json_file_path = os.path.join(os.path.join(config.image_folder, cls + config.annotation_folder_postfix,
                                                   sample_name.split('.')[0] + '.json'))
        contours = extract_contours(json_file_path)

    if contours is not None:
        wsi = pad_to_contour(wsi, contours, pad_value=255)
        orig_wsi_shape = wsi.shape

    wsi = resize_wsi(wsi, config.scale)
    wsi = pad_wsi(config, wsi, pad_value=255)
    if cls in config.no_thresh_classes:
        threshed_wsi = np.ones_like(wsi, dtype=np.uint8)
    else:
        threshed_wsi = thresh_wsi(config, wsi)
    wsi_patches = gen_patch(wsi, config.patch_size)
    threshed_wsi_patches = gen_patch(threshed_wsi, config.patch_size)

    del wsi
    gc.collect()

    if contours is None:
        mask_patches = None
        select_ids = select_patch(config, mask_patches, threshed_wsi_patches)
        save_patches(config, cls, sample_name, wsi_patches, select_ids, anno_id=None)
    else:
        for idx, cnt in enumerate(contours):
            mask = gen_contour_mask(orig_wsi_shape, [cnt])
            mask = resize_wsi(mask, config.scale)
            mask = pad_wsi(config, mask, pad_value=0)
            mask_patches = gen_patch(mask, config.patch_size)
            select_ids = select_patch(config, mask_patches, threshed_wsi_patches)
            save_patches(config, cls, sample_name, wsi_patches, select_ids, anno_id=idx)
    return True


if __name__ == "__main__":
    config = Config()
    start_time = time.time()

    full_sample_list = list()
    for val in config.sample_dict.values():
        full_sample_list += val

    print(f" -> Processing on {len(full_sample_list)} samples")
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        count = 0
        for cls in config.classes:
            work_queue = list()
            patch_fn = partial(patch_main, config, cls)
            for idx, sample in enumerate(config.sample_dict[cls]):
                work_queue.append(sample)
                if len(work_queue) == config.work_group_size or (idx + 1) == len(config.sample_dict[cls]):
                    count += len(work_queue)
                    print(f"\n    - Processing on {cls} {work_queue}    [{count}/{len(full_sample_list)}]", end='')
                    futures = executor.map(patch_fn, work_queue)
                    for future in futures:
                        try:
                            print(f" Success", end='')
                        except Exception as exec:
                            raise exec
                    work_queue.clear()
                    gc.collect()

    print(f"\nComplete in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}.")
