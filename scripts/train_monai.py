"""
This script trains a segmentation network using MONAI with augmentations done on the fly during training.
"""

import argparse
import copy
import importlib
import json
import os
import random

import numpy as np
import torch
import wandb
from monai.data import DataLoader, Dataset
from monai.losses import DiceFocalLoss
from monai.networks.nets import UNETR, AttentionUnet, SwinUNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
    Spacingd,
)
from torch import optim
from tqdm import tqdm

from auglab import configs

# Import AugLab GPU transforms 🐞
from auglab.transforms.gpu.transforms import AugTransformsGPU

# Import AugLab custom transforms
from auglab.utils.utils import (
    adjust_learning_rate,
    compute_dsc,
    fetch_image_config,
    get_validation_image,
    parser2config,
    tuple2string,
    tuple_type_float,
    tuple_type_int,
)


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train monai network")
    parser.add_argument(
        "--config",
        required=True,
        help="Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)",
    )
    parser.add_argument(
        "--transforms", default=None, help='Transforms JSON with GPU parameters default="auglab/configs/transform_params_gpu.json"'
    )
    parser.add_argument(
        "--model",
        type=str,
        default="attunet",
        choices=["attunet", "unetr", "swinunetr"],
        help='Model used for training. Options:["attunet", "unetr", "swinunetr"] (default="attunet")',
    )
    parser.add_argument("--batch-size", type=int, default=3, help="Training batch size (default=3).")
    parser.add_argument("--nb-epochs", type=int, default=300, help="Number of training epochs (default=300).")
    parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch (default=0).")
    parser.add_argument(
        "--schedule",
        type=tuple_type_float,
        default=(0.3, 0.6, 0.9),
        help="Fraction of the max epoch where the learning rate will be reduced of a factor gamma (default=(0.3, 0.6, 0.9)).",
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="Factor used to reduce the learning rate (default=0.1)")
    parser.add_argument(
        "--channels", type=tuple_type_int, default=(32, 64, 128, 256), help="Channels if attunet selected (default=16,32,64,128,256)"
    )
    parser.add_argument("--patch-size", type=tuple_type_int, default=(64, 64, 64), help="Training patch size (default=(64, 64, 64)).")
    parser.add_argument(
        "--pixdim", type=tuple_type_float, default=(1, 1, 1), help="Training resolution in RSP orientation (default=(1, 1, 1))."
    )
    parser.add_argument("--lr", default=1e-4, type=float, metavar="LR", help="Initial learning rate (default=1e-4)")
    parser.add_argument(
        "--weight-folder",
        type=str,
        default=os.path.abspath("weights/"),
        help='Folder where the weights will be stored and loaded. Will be created if does not exist. (default="src/ply/weights/3DGAN")',
    )
    parser.add_argument("--start-weights", type=str, default="", help="Path to the model weights used to start the training.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Set seed
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    # Load config data
    # Read json file and create a dictionary
    with open(args.config) as file:
        config_data = json.load(file)

    # Load variables
    weight_folder = args.weight_folder

    # Save training config
    model = args.model if args.model != "attunet" else f"{args.model}{args.channels[-1]!s}"
    json_name = f"config_{model}_pixdimRSP_{tuple2string(args.pixdim)}.json"
    saved_args = copy.copy(args)
    parser2config(saved_args, path_out=os.path.join(weight_folder, json_name))  # Create json file with training parameters

    # Create weights folder to store training weights
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    # Load images for training and validation
    print("loading images...")
    train_list, err_train = fetch_image_config(
        config_data=config_data,
        split="TRAINING",
    )

    val_list, err_val = fetch_image_config(
        config_data=config_data,
        split="VALIDATION",
    )

    # Load AugLab transform parameters 🐞
    configs_path = importlib.resources.files(configs)
    gpu_transforms_path = args.transforms if args.transforms is not None else configs_path / "transform_params_gpu.json"

    # Compose MONAI and AugLab transforms
    pixdim = args.pixdim
    patch_size = args.patch_size
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "segmentation"]),
            EnsureChannelFirstd(keys=["image", "segmentation"]),
            Orientationd(keys=["image", "segmentation"], axcodes="LAS"),
            Spacingd(
                keys=["image", "segmentation"],
                pixdim=pixdim,
                mode=(2, "nearest"),  # 2 for spline interpolation
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            RandCropByPosNegLabeld(
                keys=["image", "segmentation"],
                label_key="segmentation",
                spatial_size=patch_size,
                pos=3,
                neg=1,
                num_samples=3,
                allow_smaller=True,
            ),
            ResizeWithPadOrCropd(keys=["image", "segmentation"], spatial_size=patch_size),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "segmentation"]),
            EnsureChannelFirstd(keys=["image", "segmentation"]),
            Orientationd(keys=["image", "segmentation"], axcodes="LAS"),
            Spacingd(
                keys=["image", "segmentation"],
                pixdim=pixdim,
                mode=(2, "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            RandCropByPosNegLabeld(
                keys=["image", "segmentation"],
                label_key="segmentation",
                spatial_size=patch_size,
                pos=3,
                neg=1,
                num_samples=3,
                allow_smaller=True,
            ),
            ResizeWithPadOrCropd(keys=["image", "segmentation"], spatial_size=patch_size),
        ]
    )

    # Define train and val dataset
    train_ds = Dataset(
        data=train_list,
        transform=train_transforms,
    )
    val_ds = Dataset(
        data=val_list,
        transform=val_transforms,
    )

    # Define train and val DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=False, persistent_workers=False)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=5, pin_memory=False, persistent_workers=False)

    # Load AugLab GPU transforms and set on device 🐞
    gpu_transforms = AugTransformsGPU(json_path=gpu_transforms_path).to(device)

    # Create model
    channels = args.channels
    if args.model == "attunet":
        model = AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=1, channels=channels, strides=[2] * (len(channels) - 1), kernel_size=3
        ).to(device)
    elif args.model == "swinunetr":
        model = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=1, img_size=patch_size, feature_size=24).to(device)
    elif args.model == "unetr":
        model = UNETR(
            in_channels=1,
            out_channels=1,
            img_size=patch_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)
    else:
        raise ValueError(f"Specified model {args.model} is unknown")

    # Init weights if weights are specified
    if args.start_weights:
        # Check if weights path exists
        if not os.path.exists(args.start_weights):
            raise ValueError(f"Weights path {args.start_weights} does not exist")
        else:
            # Load model weights
            model.load_state_dict(torch.load(args.start_weights, map_location=torch.device(device))["weights"])

    # Path to the saved weights
    weights_path = f"{weight_folder}/{json_name.replace('config_SegVert_', '').replace('.json', '.pth')}"

    # Init criterion
    loss_func = DiceFocalLoss(sigmoid=True, smooth_dr=1e-4)
    torch.backends.cudnn.benchmark = True

    # Add optimizer
    lr = args.lr  # learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler()

    # 🐝 Initialize wandb run
    wandb.init(project="MonaiSeg", config=vars(args))

    # 🐝 Log gen gradients of the models to wandb
    wandb.watch(model, log_freq=100)

    # 🐝 Add training script as an artifact
    artifact_script = wandb.Artifact(name="training", type="file")
    artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
    wandb.log_artifact(artifact_script)

    # start a typical PyTorch training
    val_dsc_best = 0
    for epoch in range(args.start_epoch, args.nb_epochs):
        # Adjust learning rate
        if epoch in [int(sch * args.nb_epochs) for sch in args.schedule]:
            lr = adjust_learning_rate(optimizer, lr, gamma=args.gamma)

        print(f"\nEpoch: {epoch + 1:d} | LR: {lr:.8f}")

        # train for one epoch
        train_loss, train_dsc = train(train_loader, gpu_transforms, model, loss_func, optimizer, scaler, device)

        # 🐝 Plot loss and dice similarity coefficient
        wandb.log({"Loss_train/epoch": train_loss})
        wandb.log({"DSC_train/epoch": train_dsc})
        wandb.log({"training_lr/epoch": lr})

        # evaluate on validation set
        val_loss, val_dsc = validate(val_loader, model, loss_func, epoch, device)

        # 🐝 Plot loss and dice similarity coefficient
        wandb.log({"Loss_val/epoch": val_loss})
        wandb.log({"DSC_val/epoch": val_dsc})

        # remember best acc and save checkpoint
        if val_dsc > val_dsc_best:
            val_dsc_best = val_dsc
            state = copy.deepcopy({"weights": model.state_dict()})
            torch.save(state, weights_path)

    # 🐝 close wandb run
    wandb.finish()


def validate(data_loader, model, loss_func, epoch, device):
    model.eval()
    dsc_list = [0]
    epoch_iterator = tqdm(data_loader, desc="Validation (loss=X.X) (DSC=X.X)", dynamic_ncols=True)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            # Load input and target
            x, y = (batch["image"].to(device), batch["segmentation"].to(device))

            # Get output from model
            if x.isnan().any():
                print("found a nan in data.")
            y_pred = model(x)
            if y_pred.isnan().any():
                print("found a nan in output.")

            # Compute loss for each element in the batch size
            loss = loss_func(y_pred, y)

            # Calculate DSC
            dsc = compute_dsc(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), sigmoid=True)
            if dsc > 0:
                dsc_list.append(dsc)

            epoch_iterator.set_description(f"Validation (loss={loss.mean().item():2.5f}) (DSC={np.mean(dsc_list):2.5f})")

            # Display first image
            if step == 0:
                res_img, target_img, pred_img = get_validation_image(x, y, y_pred, sigmoid=True)

                # 🐝 log visuals for the first validation batch only in wandb
                wandb.log({"validation_img/batch_1": wandb.Image(res_img, caption=f"res_{epoch}")})
                wandb.log({"validation_img/groud_truth": wandb.Image(target_img, caption=f"ground_truth_{epoch}")})
                wandb.log({"validation_img/prediction": wandb.Image(pred_img, caption=f"prediction_{epoch}")})

    return loss.mean().item(), np.mean(dsc_list)


def train(data_loader, gpu_transforms, model, loss_func, optimizer, scaler, device):
    model.train()
    dsc_list = [0]
    epoch_iterator = tqdm(data_loader, desc="Training (loss=X.X) (DSC=X.X)", dynamic_ncols=True)
    for _step, batch in enumerate(epoch_iterator):
        # Load input and target
        x, y = batch["image"].to(device), batch["segmentation"].to(device)

        with torch.amp.autocast("cuda"):
            # Apply GPU transforms 🐞
            x_aug, y_aug = gpu_transforms(x, y)

            # Get output from model
            y_pred = model(x_aug)

            # Compute loss for each element in the batch size
            loss = 0
            loss = loss_func(y_pred, y_aug)

            # Calculate DSC
            dsc = compute_dsc(y_aug.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), sigmoid=True)
            if dsc > 0:
                dsc_list.append(dsc)

        # Train model
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_iterator.set_description(f"Training (loss={loss.mean().item():2.5f}) (DSC={np.mean(dsc_list):2.5f})")
    return loss.mean().item(), np.mean(dsc_list)


if __name__ == "__main__":
    main()
