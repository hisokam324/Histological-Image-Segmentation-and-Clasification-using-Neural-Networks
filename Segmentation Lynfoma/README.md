# MastCell Data

Dataset of histological images of canine mastcell tumor used for the publication: "Nuclear Pleomorphism in Canine Cutaneous Mast Cell Tumors – Comparison of Reproducibility and Prognostic Relevance between Estimates, Manual Morphometry and Algorithmic Morphometry"



### Modelling_Dataset:

This dataset provides RGB slide images along with their respective label masks, where each individual nucleus is identified by a unique identification number.
The ".png" files are just for the visualization of the datasamples as well as example input for the workflow example. Please use the numpy version for modelling and testing purposes.
(As indicaded by the naming scheme of the individual files, these images are sourced from tiff files.)


### mct_application:

Example workflow implemented in Python for the application of the used segmentation neural network.
This workflow example should be considered as code basis for processing either individual image samples per case or multiple samples per case, but does not represent a fully functional workflow for every type of data and available hardware.

The used Unet++(1) with RegNetY120 backbone(2) of the segmentation models package(3) was trained using the the PyTorch lightning framework(4).



### Please cite our related publication if you use this dataset:
https://doi.org/10.48550/arXiv.2309.15031


(1) Zhou Z, Siddiquee MMR, Tajbakhsh N, Liang J. UNet++: A Nested U-Net Architecture for Medical Image Segmentation. Deep Learn Med Image Anal Multimodal Learn Clin Decis Support (2018). 2018;11045: 3-11. 10.1007/978-3-030-00889-5_1

(2) Radosavovic I, Kosaraju RP, Girshick R, He K, Dollár P: Designing network design spaces. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10428-10436. 2020

(3) Iakubovskii P. Segmentation Models Pytorch. Note: https://github.com/qubvel/segmentation_models.pytorch. 2019

(4) Falcon W: PyTorch Lightning. GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning,. 2019
