# IrisFormer

This is the official Pytorch implementation of IrisFormer. We have submitted the paper to a journal for possible publication.

<br>

## Environment
Packages needed by this project:
```
pytorch
torchvision
PIL
scikit-learn
pyeer
numpy
pandas
wandb
```

<br>

## Data Preparation
IrisFormer takes normalized iris images as inputs, and all normalized images should be resized to 64*512. We deployed the Hough circle detection<sup>1</sup> to locate iris regions, and utilized the rubber-sheet model<sup>2</sup> to transform the ring-like regions into rectangles. We also applied the same contrast enhancement process as UniNet<sup>3</sup> to the normalized iris images.

Paths to the datasets are stored in the config files in the ```data_config``` folder, and you may modify them.

Train/test splitting protocols should go to the ```Protocal``` folder. We have placed example files there.

<br>

## Training
The training codes will be publicly available upon acceptance.

<br>

## Testing
Download model parameters from [Google Drive](https://drive.google.com/drive/folders/1p7yqLePpVfuf4n-PFMxnbmRwCqjx6GQB?usp=drive_link) and save them in the ```checkpoint``` folder.

Modify the ```run_name``` in the ```main``` function in ```test.py``` to the name of the model you want to test.

Run the following command for testing with IrisFormer:
```
python test.py --position_embedding rope2d --ft_pool map --save_report
```
or run the following command for testing with the original ViT:
```
python test.py --position_embedding learnable --ft_pool cls --save_report
```
Results will be saved in a ```eval``` folder under the same directory with the model parameters.

Please refer to the ```./args_config/test_config.py``` for more detailed settings.

<br>

## References
1: Wildes, Richard P. "Iris recognition: an emerging biometric technology." Proceedings of the IEEE 85.9 (1997): 1348-1363.

2: Daugman, John G. "High confidence visual recognition of persons by a test of statistical independence." IEEE transactions on pattern analysis and machine intelligence 15.11 (1993): 1148-1161.

3: Zhao, Zijing, and Ajay Kumar. "Towards more accurate iris recognition using deeply learned spatially corresponding features." Proceedings of the IEEE international conference on computer vision. 2017.
