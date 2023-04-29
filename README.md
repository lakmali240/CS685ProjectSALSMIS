# SAL-SMIS:Self-Supervised Active Learning for Semi-Supervised Medical Image Segmentation

Implemented using Pytorch

## Content
+ Abstract
+ Dataset
+ Installation
+ Training

## Abstract
Deep learning-based medical image segmentation models to be effective, require a large pool
of annotated data. However, curating such large-scale medical image datasets is challenging
due to expensive, time-consuming, and error-prone annotation procedures. As an alternative
to fully supervised models, semi-supervised learning has been gaining a lot of attention for
annotation-efficient medical image segmentation. While semi-supervised learning has shown
promise in reducing annotation requirements, it still requires a considerable number of
annotations to bootstrap the learning process. To address this issue, deep active learning can
be incorporated as a cost-effective solution. However, most existing active learning methods
randomly select initial samples, leading to redundant annotations and unnecessary costs.
This study proposes a warm-start active learning method that utilizes pre-trained weights
from self-supervised learning to build an initial representative set for annotation, resulting
in promising results with significant enhancements in semi-supervised image segmentation
task compared to current baseline methods on the ISIC 2017 skin lesion dataset.
Keywords: Semi-supervised learning, Self-supervised Learning, Active learning, image
segmentation, skin lesion


![flow_chart](https://github.com/lakmali240/CS685_Project/blob/main/Image/Overall_flow_with_UNET.jpg)


## Dataset
Download the ISIC 2017: International Skin Imaging Collaboration (ISIC) dataset from ISIC 2017 which composes of 2000 RGB dermoscopy images with binary masks of lesions.

**Preprocess**: refer to the image pre-processing method in [CEAL ](https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation).

+ Unlabeled Pool: 1600 samples

+ Labeled Pool: 400 samples

## Installation
```
Install PyTorch 1.10.2 + CUDA 11.2 
```

## STEPS

### 1. Create the dataset
```
cd data
run data.py
```
### 2. Self-Supervised Learning Model

The U-Net framework for self-supervised learning (SSL) was established to extract features from an unlabeled dataset and store the weights of the SSL model for the purpose of pre-training the segmentation model.

```
cd Self-supervised
run Self_supervised.ipynb
```

### 3. Active-Learning

Subsequently, using the clustering outcomes obtained from the potential features extracted through SSL, a standard is formulated to choose significant samples from the unlabeled data. These selected samples constitute the starting dataset for the warm-up training process.

```
cd Active-Learning
run Active_Learning.ipynb
```

### 4. Training and Testing

```
cd Semi-supervised-segmentation
bash run.sh
```





