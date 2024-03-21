import sys
import argparse
import yaml
import pathlib

from monai.networks.nets import VNet, SegResNet

from src.data import utils
from src import dataset, transforms, models, predictor, models_ori
import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
from src.model_VIT import SEUnet_VIT,SEUnet_VIT_lastfusion,SEUnet_VIT_earlyfusion,SEUnet_UNETRPP,TASE_Unet

from monai.networks.nets.ahnet import Ahnet
from monai.networks.nets.regunet import RegUNet

sys.path.append('../src/')
sys.path.append('../src/data/')

# import transforms

# import models
# import predictor


def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    path_to_data = pathlib.Path(config['path_to_data'])
    path_to_save_dir = pathlib.Path(config['path_to_save_dir'])
    path_to_weights = config['path_to_weights']
    probs = config['probs']
    num_workers = int(config['num_workers'])
    n_cls = int(config['n_cls'])
    in_channels = int(config['in_channels'])
    n_filters = int(config['n_filters'])
    reduction = int(config['reduction'])

    # test data paths:
    all_paths = utils.get_paths_to_patient_files(path_to_imgs=path_to_data, append_mask=False)

    # input transforms:
    input_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor(mode='test')
    ])

    # ensemble output transforms:
    output_transforms = [
        transforms.InverseToTensor(),
        transforms.CheckOutputShape(shape=(128, 128, 128))
    ]
    if not probs:
        output_transforms.append(transforms.ProbsToLabels())

    output_transforms = transforms.Compose(output_transforms)

    # dataset and dataloader:
    data_set = dataset.HecktorDataset(all_paths, transforms=input_transforms, mode='test')
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=num_workers)

    # model:
    # model = Ahnet(in_channels=2,out_channels=1,spatial_dims=3)
    # model = RegUNet(in_channels=2,out_channels=1,spatial_dims=3,num_channel_initial=2,depth=3)
    # model = SegResNet(spatial_dims=3, in_channels=2, out_channels=1)
    # model = models_ori.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels, n_cls, n_filters, reduction)
    # model = models.BaselineUNet(in_channels, n_cls, n_filters)
    # model = VNet(spatial_dims=3,in_channels=2,out_channels=1)
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
    # model = models.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels, n_cls, n_filters, reduction)
    model = TASE_Unet.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels, n_cls, n_filters, reduction)
    # init predictor:
    predictor_ = predictor.Predictor(
        model=model,
        path_to_model_weights=path_to_weights,
        dataloader=data_loader,
        output_transforms=output_transforms,
        path_to_save_dir=path_to_save_dir
    )

    # check if multiple paths were provided to run an ensemble:
    if isinstance(path_to_weights, list):
        predictor_.ensemble_predict()

    elif isinstance(path_to_weights, str):
        predictor_.predict()

    else:
        raise ValueError(f"Argument 'path_to_weights' must be str or list of str, provided {type(path_to_weights)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Inference Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
