import sys
import argparse
import yaml
import pathlib

import torch
# from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
import torch.nn as nn
from monai.networks.nets.vnet import VNet
from monai.networks.nets.segresnet import SegResNet
from monai.networks.nets.ahnet import Ahnet
from monai.networks.nets.senet import SENet
# from monai.networks.nets.vit import ViT
from monai.networks.nets.regunet import RegUNet
import os



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True

sys.path.append('../hecktor/src/')
sys.path.append('../hecktor/src/data/')
from hecktor.src import dataset, transforms, losses, metrics, trainer, models, models_ori
from hecktor.src.model_VIT import SEUnet_VIT
from hecktor.src.model_VIT import TASE_Unet,TransUNet
from hecktor.src.ResSimAM_UNet import ResSimAMmodel

# import transforms
# import losses
# import metrics
# import trainer
# import models
from hecktor.src.data.utils import get_paths_to_patient_files, get_train_val_paths
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    path_to_data = pathlib.Path(config['path_to_data'])
    path_to_pkl = pathlib.Path(config['path_to_pkl'])
    path_to_save_dir = pathlib.Path(config['path_to_save_dir'])

    train_batch_size = int(config['train_batch_size'])
    val_batch_size = int(config['val_batch_size'])
    num_workers = int(config['num_workers'])
    lr = float(config['lr'])
    n_epochs = int(config['n_epochs'])
    n_cls = int(config['n_cls'])
    in_channels = int(config['in_channels'])
    n_filters = int(config['n_filters'])
    reduction = int(config['reduction'])
    T_0 = int(config['T_0'])
    eta_min = float(config['eta_min'])
    baseline = config['baseline']

    # train and val data paths:
    all_paths = get_paths_to_patient_files(path_to_imgs=path_to_data, append_mask=True)
    # print(all_paths)
    train_paths, val_paths = get_train_val_paths(all_paths=all_paths, path_to_train_val_pkl=path_to_pkl)
    train_paths = train_paths[:180]
    val_paths = val_paths[:35]

    # train and val data transforms:
    train_transforms = transforms.Compose([
        transforms.RandomRotation(p=0.5, angle_range=[0, 45]),
        transforms.Mirroring(p=0.5),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.RandomRotation(p=0.5, angle_range=[0, 45]),
        transforms.Mirroring(p=0.5),
        transforms.NormalizeIntensity(),
        transforms.ToTensor(),

    ])

    # datasets:
    train_set = dataset.HecktorDataset(train_paths, transforms=train_transforms)
    val_set = dataset.HecktorDataset(val_paths, transforms=val_transforms)

    # dataloaders:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    if baseline:
        # model = models.BaselineUNet(in_channels, n_cls, n_filters)
        model = ResSimAMmodel.ResSimAM_Unet(in_channels, n_cls, n_filters, e_lambda=1e-4)
        # model = TransUNet.TransUnet(args)
        # model = models.build_UNETR()
        # model = models.UNETR(in_channels=2,
        #                      out_channels=1,
        #                      img_size=(128, 128, 128),
        #                      feature_size=16,
        #                      hidden_size=768,
        #                      mlp_dim=3072,
        #                      num_heads=12,
        #                      pos_embed='perceptron',
        #                      norm_name='instance',
        #                      conv_block=True,
        #                      res_block=True,
        #                      dropout_rate=0.0
        #                      )
        # model = swin_unetr.SwinUNETR(
        #     img_size=(128, 128, 128),
        #     in_channels=2,
        #     out_channels=1,
        #     feature_size=16,
        #     drop_rate=0.0,
        #     attn_drop_rate=0.0,
        #     dropout_path_rate=0.0
        # )
        # model = SENet(spatial_dims=3,in_channels=2,block=2,layers=[2,2,2],groups=8,reduction=16)
        # model = ViT(in_channels=2, spatial_dims=3, img_size=(128, 128, 128), patch_size=16)
        # model = Ahnet(spatial_dims=3,in_channels=2,out_channels=1)
        # model = VNet(spatial_dims=3,in_channels=2, out_channels=1)
        # model = SegResNet(spatial_dims=3,in_channels=2,out_channels=1)
        # model = SegResNet(spatial_dims=3,in_channels=2,out_channels=1)
    else:
        model = TASE_Unet.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels, n_cls, n_filters, reduction)

    criterion = losses.Dice_and_FocalLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    metric = metrics.dice
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # dice_loss = DiceCELoss(
    #     to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6
    # )
    # dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    # scheduler = LinearWarmupCosineAnnealingLR(
    #     optimizer, warmup_epochs=50, max_epochs=5000
    # )
    ######原UNETR


    trainer_ = trainer.ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        metric=metric,
        scheduler=scheduler,
        num_epochs=n_epochs,
        parallel=False,
        cuda_device="cuda:0"
        # cuda_device="cpu"
    )

    trainer_.train_model()
    trainer_.save_results(path_to_dir=path_to_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
