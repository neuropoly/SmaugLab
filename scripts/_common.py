"""Helpers shared by the standalone scripts in this directory.

These used to live in `smauglab/utils/utils.py` and so shipped in the wheel, but
nothing under `smauglab/` ever imported them once the `__main__` demo blocks were
removed -- they are MONAI-training and data-loading support for `train_monai.py` and
`generate_augmentations.py`, not part of the augmentation library. `config2parser`
and `sig_fn` came along too and had no callers at all; they are gone.

`smauglab/utils/image.py` deliberately did NOT move: five modules in the sibling
segtransferaug repository import `smauglab.utils.image.Image`, so it is a real part
of the public API despite having no in-package consumer.

Scripts here are run as `python scripts/<name>.py`, which puts this directory on
sys.path, so `from _common import ...` resolves.
"""

from __future__ import annotations

import json
import os

import numpy as np
from progress.bar import Bar


def fetch_image_config(config_data: dict, split: str = "TRAINING") -> tuple[list[dict], list]:
    """Resolve a data config's image/label pairs for one split.

    :param config_data: Config dict where every label used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param split: Split of the data needed in the config file ('TRAINING', 'VALIDATION', 'TESTING').
    :return: out_list: list of dictionary with image and label paths (like monai load_decathlon_datalist)
        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': '/workspace/data/chest_19_label.nii.gz'},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': '/workspace/data/chest_31_label.nii.gz'}
        ]
    """
    # Check config type to ensure that labels paths are specified and not images
    if config_data["TYPE"] != "LABEL":
        raise ValueError("TYPE error: Type LABEL not detected")

    # Get file paths based on split
    dict_list = config_data[split]

    # Init progression bar
    bar = Bar(f"Load {split} data", max=len(dict_list))

    err = []
    out_list = []
    for i, di in enumerate(dict_list):
        input_img_path = os.path.join(config_data["DATASETS_PATH"], di["IMAGE"])
        input_seg_path = os.path.join(config_data["DATASETS_PATH"], di["LABEL"])
        if not os.path.exists(input_img_path):
            err.append([input_img_path, "path error"])
        else:
            out_list.append({"image": os.path.abspath(input_img_path), "segmentation": os.path.abspath(input_seg_path)})

        # Plot progress. Indexing by enumerate, not dict_list.index(di): the latter is
        # a linear scan per item (quadratic overall) and reports the wrong number
        # whenever two entries are equal.
        bar.suffix = f"{i + 1}/{len(dict_list)}"
        bar.next()
    bar.finish()
    return out_list, err


def parser2config(args, path_out: str) -> None:
    """Extract the parameters from an input parser to create a config json file.

    :param args: parser arguments
    :param path_out: path out of the config file
    """
    # Check if path_out exists or create it
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))

    # Serializing json
    json_object = json.dumps(vars(args), indent=4)

    # Inform user
    if os.path.exists(path_out):
        print(f"The config file {path_out} with all the training parameters was updated")
    else:
        print(f"The config file {path_out} with all the training parameters was created")

    # Write json file
    with open(path_out, "w") as outfile:
        outfile.write(json_object)


def tuple_type_int(strings: str) -> tuple[int, ...]:
    """Copied from https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument"""
    strings = strings.replace("(", "").replace(")", "")
    return tuple(map(int, strings.split(",")))


def tuple_type_float(strings: str) -> tuple[float, ...]:
    """Copied from https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument"""
    strings = strings.replace("(", "").replace(")", "")
    return tuple(map(float, strings.split(",")))


def tuple2string(t) -> str:
    return str(t).replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")


def adjust_learning_rate(optimizer, lr: float, gamma: float) -> float:
    """Set the learning rate to the initial LR decayed by schedule.

    Copied from https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    """
    lr *= gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def compute_dsc(gt_mask, pred_mask, sigmoid: bool = False):
    """Dice similarity coefficient.

    :param gt_mask: Ground truth mask used as the reference
    :param pred_mask: Prediction mask
    :param sigmoid: Apply sigmoid on prediction if True (default=False)

    :return: dsc=2*intersection/(number of non zero pixels)
    """
    if sigmoid:
        pred_mask = 1 / (1 + np.exp(-pred_mask))
    numerator = 2 * (gt_mask * pred_mask).sum()
    denominator = gt_mask.sum() + pred_mask.sum()
    if denominator == 0:
        # Both ground truth and prediction are empty
        return 0
    return numerator / denominator


def get_validation_image(in_img, target_img, pred_img, sigmoid: bool = False):
    """Stack input / target / prediction mid-slices into one image for logging."""
    in_img = in_img.data.cpu().numpy()
    target_img = target_img.data.cpu().numpy()
    pred_img = pred_img.data.cpu().numpy()
    if sigmoid:
        pred_img = 1 / (1 + np.exp(-pred_img))
    in_all = []
    target_all = []
    pred_all = []
    for num_batch in range(in_img.shape[0]):
        # Load 3D numpy array
        x = in_img[num_batch, 0]
        y = target_img[num_batch, 0]
        y_pred = pred_img[num_batch, 0]
        shape = x.shape

        # Extract middle slice
        x = x[shape[0] // 2, :, :]
        y = y[shape[0] // 2, :, :]
        y_pred = y_pred[shape[0] // 2, :, :]

        # Normalize intensity
        x = normalize_percentile(x) * 255
        y = normalize_percentile(y) * 255
        y_pred = normalize_percentile(y_pred) * 255

        # Regroup batch
        in_all.append(x)
        target_all.append(y)
        pred_all.append(y_pred)

    # Regroup batch into 1 array
    in_line_arr = np.concatenate(np.array(in_all), axis=1)
    target_line_arr = np.concatenate(np.array(target_all), axis=1)
    pred_line_arr = np.concatenate(np.array(pred_all), axis=1)

    # Regroup image/target/pred into 1 array
    img_result = np.concatenate((in_line_arr, target_line_arr, pred_line_arr), axis=0)

    return img_result, target_line_arr, pred_line_arr


def normalize_percentile(arr: np.ndarray) -> np.ndarray:
    """Rescale using the 10th/90th percentiles.

    Renamed from `normalize`: three functions in this repository shared that name and
    two of them computed something else (min-max, in the GPU demo blocks). The name now
    says which one this is. See `normalize_minmax` in demo_augmentations.py.
    """
    p10 = np.percentile(arr, 10)
    p90 = np.percentile(arr, 90)
    return (arr - p10) / (p90 - p10 + 0.00001)
