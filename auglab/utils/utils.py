import argparse
import json
import os

import numpy as np
from progress.bar import Bar


def fetch_image_config(config_data, split="TRAINING"):
    """
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
    for di in dict_list:
        input_img_path = os.path.join(config_data["DATASETS_PATH"], di["IMAGE"])
        input_seg_path = os.path.join(config_data["DATASETS_PATH"], di["LABEL"])
        if not os.path.exists(input_img_path):
            err.append([input_img_path, "path error"])
        else:
            out_list.append({"image": os.path.abspath(input_img_path), "segmentation": os.path.abspath(input_seg_path)})

        # Plot progress
        bar.suffix = f"{dict_list.index(di) + 1}/{len(dict_list)}"
        bar.next()
    bar.finish()
    return out_list, err


def config2parser(config_path):
    """
    Create a parser object from a json file
    """
    # Read json file and create a dictionary
    with open(config_path) as file:
        config_dict = json.load(file)

    return argparse.Namespace(**config_dict)


def parser2config(args, path_out):
    """
    Extract the parameters from an input parser to create a config json file
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


def tuple_type_int(strings):
    """
    Copied from https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
    """
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def tuple_type_float(strings):
    """
    Copied from https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
    """
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)


def tuple2string(t):
    return str(t).replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")


def adjust_learning_rate(optimizer, lr, gamma):
    """
    Sets the learning rate to the initial LR decayed by schedule
    Copied from https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    """
    lr *= gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def compute_dsc(gt_mask, pred_mask, sigmoid=False):
    """
    :param gt_mask: Ground truth mask used as the reference
    :param pred_mask: Prediction mask
    :param sigmoid: Apply sigmoid on prediction if True (default=False)

    :return: dsc=2*intersection/(number of non zero pixels)
    """
    if sigmoid:
        pred_mask = sig_fn(pred_mask)
    numerator = 2 * (gt_mask * pred_mask).sum()
    denominator = gt_mask.sum() + pred_mask.sum()
    if denominator == 0:
        # Both ground truth and prediction are empty
        return 0
    else:
        return numerator / denominator


def sig_fn(z):
    return 1 / (1 + np.exp(-z))


def get_validation_image(in_img, target_img, pred_img, sigmoid=False):
    in_img = in_img.data.cpu().numpy()
    target_img = target_img.data.cpu().numpy()
    pred_img = pred_img.data.cpu().numpy()
    if sigmoid:
        pred_img = sig_fn(pred_img)
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
        x = normalize(x) * 255
        y = normalize(y) * 255
        y_pred = normalize(y_pred) * 255

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


def normalize(arr):
    """
    Normalize image using percentiles
    """
    # Use 10th percentile
    p10 = np.percentile(arr, 10)
    p90 = np.percentile(arr, 90)
    return (arr - p10) / (p90 - p10 + 0.00001)
