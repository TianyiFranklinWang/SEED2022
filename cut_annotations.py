import datetime
import json
import math
import os
import time

import cv2
import numpy as np
import tifffile as tiff


class Config:
    def __init__(self):
        self.image_folder = "./input/seed_processed_data"
        self.classes = ['T0', 'T1', 'T2', 'T3', 'Tis', 'test']
        self.annotated_classes = ['T1', 'T2', 'T3', 'Tis', 'test']
        self.annotation_folder_postfix = '_json'
        self.sample_dict = dict()
        self.annotation_dict = dict()

        self.scale = 12

        self.output_folder = f"./input/seed_patch/seed_cut_anno"

        self.gather_sample_and_annotation()

    def gather_sample_and_annotation(self):
        for cls in self.annotated_classes:
            self.sample_dict[cls] = os.listdir(os.path.join(self.image_folder, cls))
            self.annotation_dict[cls] = os.listdir(
                os.path.join(self.image_folder, cls + self.annotation_folder_postfix))


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


def getSubImage(src, rect):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(round, center)), tuple(map(round, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D(center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(255, 255, 255))
    out = cv2.getRectSubPix(dst, size, center)
    return out


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


def gen_contour_mask(mask_shape, contours, thickness=-1):
    mask = np.zeros(shape=mask_shape, dtype=np.uint8)
    for cnt in contours:
        mask = cv2.drawContours(mask, [cnt], 0, (255, 255, 255), thickness)
    return mask[:, :, 0]


def pad_to_rotate(wsi, pad_value=255):
    width, height = wsi.shape[1], wsi.shape[0]
    pad = math.ceil(math.sqrt(width ** 2 + height ** 2)) // 2
    if len(wsi.shape) == 3:
        pad = [[pad, pad], [pad, pad], [0, 0]]
    elif len(wsi.shape) == 2:
        pad = [[pad, pad], [pad, pad]]
    wsi = np.pad(wsi, pad, constant_values=pad_value)
    return wsi


def create_none_exist_folder(path: str) -> None:
    """Create folders that don't exist.
    Args:
        path (str): Folder path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    config = Config()
    start_time = time.time()

    full_annotations_list = list()
    for val in config.annotation_dict.values():
        full_annotations_list += val

    print(f" -> Processing on {len(full_annotations_list)} samples")
    count = 0
    for cls, annotations in config.annotation_dict.items():
        create_none_exist_folder(os.path.join(config.output_folder, cls))
        for annotation in annotations:
            print(f"    - Processing on {cls} {annotation}    [{count + 1}/{len(full_annotations_list)}]")
            wsi = tiff.imread(os.path.join(config.image_folder, cls, annotation.split(".")[0] + '.tif'))
            contours = extract_contours(
                os.path.join(config.image_folder, cls + config.annotation_folder_postfix, annotation))
            wsi = pad_to_contour(wsi, contours)
            orig_wsi_shape = wsi.shape
            wsi = resize_wsi(wsi, config.scale)
            wsi = pad_to_rotate(wsi)

            for idx, cnt in enumerate(contours):
                mask = gen_contour_mask(orig_wsi_shape, [cnt])
                mask = resize_wsi(mask, config.scale)
                mask = pad_to_rotate(mask, pad_value=0)
                cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                rect = cv2.minAreaRect(cnt[0])
                out = getSubImage(wsi, rect)
                cv2.imwrite(os.path.join(config.output_folder, cls,
                                         annotation.split(".")[0] + f'_Annotation{idx}' + '.png'), out)
            count += 1

    print(f"\nComplete in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}.")
