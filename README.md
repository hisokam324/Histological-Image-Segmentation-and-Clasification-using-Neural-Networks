# Histological Image Segmentation and Clasification using Neural Networks

This repository has some examples on how to create basic segmentation and clasification neural networks models using Pytorch

<p align="center">
    <img src=Readme_images/example.png width = "100%">
</p>

## Index

### Readme

[Installation](#installation)

[Segmentation](#segmentation)

[Clasification](#clasification)

[Clasification Lynfoma](#clasification-lynfoma)

[Credits](#credits)


### Wiki

[Tutorials](https://github.com/hisokam324/Histological-Image-Segmentation-and-Clasification-using-Neural-Networks/wiki/Tutorials) 

[src](https://github.com/hisokam324/Histological-Image-Segmentation-and-Clasification-using-Neural-Networks/wiki/src)

## Installation

In order to use this code is necessary to install the modules in requirements.txt.

    pip install -r requirements.txt

## Tutorials

This folder has tutorials in order to undertand how to implement basic neural networks in Pytorch.

### Pytorch

Collection of Pytorch tutorials extracted from [pytorch.org](https://pytorch.org/) and [youtube.com/pytorch](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN). It has 5 ipynb notbooks about Tensors, Autograd, building models, tensorborad and model trainning. 

### Neural Networks

Collection of basic Neural Networks tutorials extracted from [OpenFing/Aprendizaje Profundo para el Análisis de Imágenes Biomédicas](https://open.fing.edu.uy/courses/dlbioim/4/) It has 5 ipynb notbooks about Introduction, Autograd, MLP, CNN and RNR. Additionaly, following the preexising notbooks, 2 new notbooks were created, named Auto (implementing an Autoencoder) and UNet. This last notbooks were tested using the [data-science-bowl-2018](https://www.kaggle.com/competitions/data-science-bowl-2018) dataset consisting in fluorecent images for segmentation.

## src

This folder has the results of implemnting some basic Neural Networks in pytorch for Histological analisys.

### utils

Axiluary module to run main code.

### models

Pytorch models implementation

### Segmentation

Segmentation trainning over [NuInsSeg](https://www.kaggle.com/datasets/ipateam/nuinsseg)

### Clasification

Clasification trainning over [PathMNIST](https://medmnist.com/)

### Clasification Lynfoma

Segmentation and Clasification trainning over [fe-extern/Lymphoma Dataset](https://git.fh-ooe.at/fe-extern/Lymphoma-Dataset)

## Credits

Llenar creditos
