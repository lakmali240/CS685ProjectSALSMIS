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


![flow_chart](./image/Overall_flow_with_UNET.jpg)
