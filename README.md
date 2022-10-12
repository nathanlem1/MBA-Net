### Multi-Branch with Attention Network for Hand-Based Person Recognition

Code for the paper [Multi-Branch with Attention Network for Hand-Based Person Recognition](https://arxiv.org/abs/2108.02234) which has been published on [ICPR 2022](https://www.icpr2022.com/).

## Overview
In this paper, we propose a novel hand-based person recognition method for the purpose of criminal investigations since the hand image is often the only available information in cases of serious crime such as sexual abuse. Our proposed method, Multi-Branch with Attention Network (MBA-Net), incorporates both channel and spatial attention modules in branches in addition to a global (without attention) branch to capture global structural information for discriminative feature learning. The attention modules focus on the relevant features of the hand image while suppressing the irrelevant backgrounds. In order to overcome the weakness of the attention mechanisms, equivariant to pixel shuffling, we integrate relative positional encodings into the spatial attention module to capture the spatial positions of pixels. Extensive evaluations on two large multi-ethnic and publicly available hand datasets demonstrate that our proposed method achieves state-of-the-art performance, surpassing the existing hand-based identification methods. 


The proposed attention modules and the overall MBA-Net architecture are shown below.

a) Channel Attention Module (CAM):

![](./doc_images/CAM.png)

b) Spatial Attention Module with Relative Positional Encodings (SAM-RPE):

![](./doc_images/SAM.png)

c) MBA-Net:

![](./doc_images/MBA_Net.png)



The qualitative result of our proposed method is shown below. 

![](./doc_images/results_demo.png)
Some qualitative results of our method using query vs
ranked results retrieved from gallery. From top to bottom row
are HD, left palmar of 11k, right palmar of 11k, left dorsal
of 11k and right dorsal of 11k datasets. The green and red
bounding boxes denote the correct and the wrong matches,
respectively.

## Installation

## Data Preparation

## Train

## Evaluate

## Reference

