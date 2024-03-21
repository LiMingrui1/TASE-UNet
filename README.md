# TASE-UNet: A dual-branch encoder network based on SE-UNet and Transformer for 3D PET-CT images tumor segmentation


> Thank iantsen for sharing the code. The structure of this code is similar to that of iantsen. 
> Some methods are derived from https://github.com/iantsen/hecktor.


### Main requirements
- PyTorch 1.6.0 (cuda 10.2)
- SimpleITK 1.2.4 (ITK 4.13)
- nibabel 3.1.1
- skimage 0.17.2

### Dataset
Train and test images are available through the competition [website]( https://hecktor.grand-challenge.org/). 
     

### Data preprocessing
The data preprocessing consists of:
- Resampling the pair of PET & CT images for each patient to a common reference space.
- Extracting the region of interest (bounding box) of the size of 128x128x128 voxels. 
- Saving the transformed images in NIfTI format.
- The bounding box extraction is performed using the boundings.py file, in which the threshold and the size of the bounding box can be adjusted. Use resample.py to cut the image according to the bounding box.
```sh
python boundings.py
python resample.py
```
### Training
For training the model from scratch, one can use `notebooks/model_train.ipynb` setting all parameters right in the notebook. Otherwise, with all parameters written in the config file, one needs to run `hecktor/model/train.py` from its current directory:
```sh
python train.py -p hecktor/config/model_train.yaml
```
All parameters are described in `hecktor/config/model_train.yaml` that should be used as a template to build your own config file.

### Inference
For inference, run the script `hecktor/model/predict.py` with parameters defined in the config file `hecktor/config/model_predict.yaml`:
```sh
python predict.py -p hecktor/config/model_predict.yaml
```


### Reference
We used iantsen's image segmentation framework, so we cite the following paper ([arXiv](https://arxiv.org/abs/2102.10446)):
> Iantsen A., Visvikis D., Hatt M. (2021) Squeeze-and-Excitation Normalization for Automated Delineation of Head and Neck Primary Tumors in Combined PET and CT Images. In: Andrearczyk V., Oreiller V., Depeursinge A. (eds) Head and Neck Tumor Segmentation. HECKTOR 2020. Lecture Notes in Computer Science, vol 12603. Springer, Cham. https://doi.org/10.1007/978-3-030-67194-5_4
