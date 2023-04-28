#!/usr/bin/bash

gpu=1
num_classes=2
# weight_path='/content/drive/MyDrive/Spring_research_2023/SSLModel/Reuslts/pretrained_weights/2023-04-02_02-19-33/ISIC_Unsup.pt'
weight_path='/content/drive/MyDrive/Spring_research_2023/SSLModel/Reuslts/pretrained_weights/2023-04-22_20-58-04/ISIC_Unsup.pt'
num_epochs=10
batch_size=10
pre_train_type=None
# pre_train_type=continue
lr=0.001
loss=ce_dice
select_type=select
select_num=300
train_num=1600
iteration_num=9
active_epochs=2
aug_samples=35
# load_idx='../overall_result/random_select_index/2022-01-17_20-04-08/index.npy'
round=10
# data_path_train="/content/drive/MyDrive/Spring_research_2023/data/GrayData"


cluster_idx_path_512_10='/content/drive/MyDrive/Spring_research_2023/Cluster_Results/Hidden_features/euclidean_pooling_512_dim511/2023-04-23_13-15-12_cluster_ids_x.pt'
cluster_npy_path_512_10='/content/drive/MyDrive/Spring_research_2023/Cluster_Results/Hidden_features/euclidean_pooling_512_dim511/2023-04-23_13-15-12_cluster.npy'

# cluster_idx_path_512_5
# cluster_npy_path_512_5

# cluster_idx_path_2048_10
# cluster_npy_path_2048_10

# cluster_idx_path_2048_5
# cluster_npy_path_2048_5


python /content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/train.py --Adam --num_classes $num_classes --num-epochs $num_epochs --batch-size $batch_size --pre_train_type None --lr $lr --loss $loss --select_type $select_type --select_num $select_num --train_num $train_num --iteration_num $iteration_num --active_epochs $active_epochs --aug_samples $aug_samples --round $round --weight_path $weight_path --cluster_idx_path $cluster_idx_path_512_10 --cluster_npy_path $cluster_npy_path_512_10 

