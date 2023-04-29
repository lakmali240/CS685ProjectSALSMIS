import torch
from torch.utils.data import DataLoader
from model import UNet
import numpy as np
from tqdm import tqdm
from utils import losses
import binary as mmb
import pdb
import pytz

def test(test_set,checkpoint_path):
    net = 'UNet'
    path =checkpoint_path
    
    num_classes = 2
    img_size = [256,256]
    testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    model = UNet(n_channels=1, n_classes=num_classes).cuda()  # old version
    model = model.cuda()
    print('load model from', path)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    dice_list = []
    dice_test = []
    
    pred_vol_lst  = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
    label_vol_lst = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
    
    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cpu().numpy()

        outputs = model(volume_batch)
        outputs = torch.softmax(outputs, dim = 1)
        outputs = torch.argmax(outputs, dim = 1).cpu().numpy()
        
        pred_vol_lst[i_batch] = outputs[0,...]
        label_vol_lst[i_batch] = label_batch[0,...]
        
        # print("outputs:",outputs)

    for i in range(len(testloader)):
        dice_test.append(mmb.dc(pred_vol_lst[i], label_vol_lst[i]))
        for c in range(1, num_classes):   
            pred_test_data_tr = pred_vol_lst[i].copy()              
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label_vol_lst[i].copy() 
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))

    dice_arr = 100 * np.reshape(dice_list, [-1, 2]).transpose()
    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    print('Dice0:',dice_mean[0])
    print('Dice1:',dice_mean[1])
    print('Dice test:',dice_arr.mean())
    
    # i = 2
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(pred_vol_lst[i], cmap='gray')
    # ax[0].set_title('Predicted Mask')
    # ax[1].imshow(label_vol_lst[i], cmap='gray')
    # ax[1].set_title('Ground Truth Mask')
    # fig.suptitle(f'Test {i}, Dice: {dice_test[i]:.2f}', fontsize=16)
    # fig.savefig(f'/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/Predicted_masks/test_{i}_dice_{dice_test[i]:.2f}.png')
    # plt.close(fig)
    
    # select a specific index to visualize
    idx = 18
    # get the image and label for the selected index
    # image = volume_batch[idx]
    # Label_img = masks[idx]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(volume_batch[idx], cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(label_vol_lst[idx], cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(fpred_vol_lst[idx], cmap='gray')
    axs[2].set_title('Predicted output')
    fig.savefig(f'/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/Predicted_masks/test_{i}_dice_{dice_test[i]:.2f}.png')
    
    
    return dice_arr.mean(),outputs





# import torch
# from torch.utils.data import DataLoader
# from model import UNet
# import numpy as np
# from tqdm import tqdm
# from utils import losses
# import binary as mmb
# import pdb
# import pytz
# import matplotlib.pyplot as plt
# import os
# import matplotlib.pyplot as plt

# #===================================== test code-------------

# def test(test_set,checkpoint_path):
#     net = 'UNet'
#     path =checkpoint_path
    
#     num_classes = 2
#     img_size = [256,256]
#     testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

#     model = UNet(n_channels=1, n_classes=num_classes)  # old version
#     model = model
#     print('load model from', path)
#     model.load_state_dict(torch.load(path))
#     model.eval()
    
#     dice_list = []
#     dice_test = []

#     for i_batch, sampled_batch in enumerate(tqdm(testloader)):
#         volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#         volume_batch, label_batch = volume_batch.type(torch.FloatTensor), label_batch.type(torch.LongTensor).cpu().numpy()

#         outputs = model(volume_batch)
#         outputs = torch.argmax(outputs, dim = 1)
#         outputs = outputs.detach().cpu().numpy()
        
#         print(outputs, outputs.shape)
        
#         pred_vol_lst = []
#         label_vol_lst = []
#         dice_per_batch = []

#         for c in range(num_classes):
#             # Compute dice score per class
#             pred_test_data_tr = outputs[0, ...].copy()              
#             pred_test_data_tr[pred_test_data_tr != c] = 0

#             pred_gt_data_tr = label_batch[0, ...].copy() 
#             pred_gt_data_tr[pred_gt_data_tr != c] = 0
            
#             # print('pred_test_data_tr:',pred_test_data_tr)
#             # print('pred_gt_data_tr:',pred_gt_data_tr)
#             dice_per_class = mmb.dc(pred_test_data_tr, pred_gt_data_tr)
#             dice_list.append(dice_per_class)
#             dice_per_batch.append(dice_per_class)

#             # Append predicted and ground truth masks to lists
#             pred_vol_lst.append(pred_test_data_tr)
#             label_vol_lst.append(pred_gt_data_tr)

#         # Compute dice score per batch
#         dice_test.append(np.mean(dice_per_batch))

#         i = i_batch
#         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#         ax[0].imshow(pred_vol_lst[1], cmap='binary')
#         ax[0].set_title('Predicted Mask')
#         ax[1].imshow(label_vol_lst[1], cmap='binary')
#         ax[1].set_title('Ground Truth Mask')
#         fig.suptitle(f'Test {i}, Dice: {dice_test[i]:.2f}', fontsize=16)
#         fig.savefig(f'/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/Predicted_masks/test_{i}_dice_{dice_test[i]:.2f}.png')
#         plt.close(fig)

#     dice_arr = 100 * np.reshape(dice_list, [-1, num_classes]).transpose()
#     dice_mean = np.mean(dice_arr, axis=1)
#     dice_std = np.std(dice_arr, axis=1)
    
#     print('Dice0:',dice_mean[0])
#     print('Dice1:',dice_mean[1])
#     print('Dice test:',dice_arr.mean())
    
#     # i = 2
#     # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     # ax[0].imshow(pred_vol_lst[i], cmap='gray')
#     # ax[0].set_title('Predicted Mask')
#     # ax[1].imshow(label_vol_lst[i], cmap='gray')
#     # ax[1].set_title('Ground Truth Mask')
#     # fig.suptitle(f'Test {i}, Dice: {dice_test[i]:.2f}', fontsize=16)
#     # fig.savefig(f'/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/Predicted_masks/test_{i}_dice_{dice_test[i]:.2f}.png')
#     # plt.close(fig)
    
#     return dice_arr.mean(),outputs


# #=====================================Lakmali2===============

# # def test(test_set, checkpoint_path):
# #     net = 'UNet'
# #     path = checkpoint_path

# #     num_classes = 2
# #     img_size = [256,256]
# #     testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

# #     model = UNet(n_channels=1, n_classes=num_classes)  # old version
# #     model = model
# #     print('load model from', path)
# #     model.load_state_dict(torch.load(path))
# #     model.eval()

# #     dice_list = []
# #     dice_test = []

# #     pred_vol_lst  = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
# #     label_vol_lst = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
    
# #     for i_batch, sampled_batch in enumerate(tqdm(testloader)):
# #         volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
# #         volume_batch, label_batch = volume_batch.type(torch.FloatTensor), label_batch.type(torch.LongTensor).cpu().numpy()

# #         outputs = model(volume_batch)
# #         outputs = torch.argmax(outputs, dim = 1)
# #         outputs = outputs.detach().cpu().numpy()
        
# #         # outputs = torch.softmax(outputs, dim = 1)
# #         # outputs = torch.argmax(outputs, dim = 1).cpu().numpy()

# #         pred_vol_lst[i_batch] = outputs[0,...]
# #         label_vol_lst[i_batch] = label_batch[0,...]
        
# #     for i in range(len(testloader)):
# #         dice_test.append(mmb.dc(pred_vol_lst[i], label_vol_lst[i]))
# #         for c in range(1, num_classes):
# #             pred_test_data_tr = pred_vol_lst[i].copy()
# #             pred_test_data_tr[pred_test_data_tr != c] = 0

# #             pred_gt_data_tr = label_vol_lst[i].copy()
# #             pred_gt_data_tr[pred_gt_data_tr != c] = 0

# #             dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))

# #     dice_arr = 100 * np.reshape(dice_list, [-1, 2]).transpose()
# #     dice_mean = np.mean(dice_arr, axis=1)
# #     dice_std = np.std(dice_arr, axis=1)

# #     print('Dice0:',dice_mean[0])
# #     print('Dice1:',dice_mean[1])
# #     print('Dice test:',dice_arr.mean())

# #     # Print out the individual Dice coefficients for each class
# #     print('Dice coefficients for class 0:', dice_arr[0])
# #     print('Dice coefficients for class 1:', dice_arr[1])

# #     if not os.path.exists('/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/Predicted_masks'):
# #         os.makedirs('/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/Predicted_masks')
        
# #     i = 2
# #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# #     ax[0].imshow(pred_vol_lst[i], cmap='gray')
# #     ax[0].set_title('Predicted Mask')
# #     ax[1].imshow(label_vol_lst[i], cmap='gray')
# #     ax[1].set_title('Ground Truth Mask')
# #     fig.suptitle(f'Test {i}, Dice: {dice_test[i]:.2f}', fontsize=16)
# #     fig.savefig(f'/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/Predicted_masks/test_{i}_dice_{dice_test[i]:.2f}.png')
# #     plt.close(fig)
        
# #     return dice_arr.mean(), outputs



    
# #============================================================

# # import matplotlib.pyplot as plt

# # def test(test_set, checkpoint_path):
# #     net = 'UNet'
# #     path = checkpoint_path

# #     num_classes = 2
# #     img_size = [256,256]
# #     testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

# #     model = UNet(n_channels=1, n_classes=num_classes)  # old version
# #     model = model
# #     print('load model from', path)
# #     model.load_state_dict(torch.load(path))
# #     model.eval()

# #     dice_list = []
# #     dice_test = []

# #     pred_vol_lst  = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
# #     label_vol_lst = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
    
# #     for i_batch, sampled_batch in enumerate(tqdm(testloader)):
# #         volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
# #         volume_batch, label_batch = volume_batch.type(torch.FloatTensor), label_batch.type(torch.LongTensor).cpu().numpy()

# #         outputs = model(volume_batch)
# #         outputs = torch.softmax(outputs, dim = 1)
# #         outputs = torch.argmax(outputs, dim = 1).cpu().numpy()

# #         pred_vol_lst[i_batch] = outputs[0,...]
# #         label_vol_lst[i_batch] = label_batch[0,...]
        
# #     for i in range(len(testloader)):
# #         dice_test.append(mmb.dc(pred_vol_lst[i], label_vol_lst[i]))
# #         for c in range(1, num_classes):
# #             pred_test_data_tr = pred_vol_lst[i].copy()
# #             pred_test_data_tr[pred_test_data_tr != c] = 0

# #             pred_gt_data_tr = label_vol_lst[i].copy()
# #             pred_gt_data_tr[pred_gt_data_tr != c] = 0

# #             dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))

# #     dice_arr = 100 * np.reshape(dice_list, [-1, 2]).transpose()
# #     dice_mean = np.mean(dice_arr, axis=1)
# #     dice_std = np.std(dice_arr, axis=1)

# #     print('Dice0:',dice_mean[0])
# #     print('Dice1:',dice_mean[1])
# #     print('Dice test:',dice_arr.mean())

# #     # Print out the individual Dice coefficients for each class
# #     # print('Dice coefficients for class 0:', dice_arr[0])
# #     # print('Dice coefficients for class 1:', dice_arr[1])

# #     if not os.path.exists('/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/predicted_masks'):
# #         os.makedirs('/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/predicted_masks')
        
# #     i = 2
# #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# #     ax[0].imshow(pred_vol_lst[i], cmap='gray')
# #     ax[0].set_title('Predicted Mask')
# #     ax[1].imshow(label_vol_lst[i], cmap='gray')
# #     ax[1].set_title('Ground Truth Mask')
# #     fig.suptitle(f'Test {i}, Dice: {dice_test[i]:.2f}', fontsize=16)
# #     fig.savefig(f'/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/predicted_masks/test_{i}_dice_{dice_test[i]:.2f}.png')
# #     plt.close(fig)
        
# #     # # Save predicted masks and ground truth masks side-by-side
# #     # for i in range(len(testloader)):
# #     #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# #     #     ax[0].imshow(pred_vol_lst[i], cmap='gray')
# #     #     ax[0].set_title('Predicted Mask')
# #     #     ax[1].imshow(label_vol_lst[i], cmap='gray')
# #     #     ax[1].set_title('Ground Truth Mask')
# #     #     fig.suptitle(f'Test {i}, Dice: {dice_test[i]:.2f}', fontsize=16)
# #     #     fig.savefig(f'/content/drive/MyDrive/Spring_research_2023/python_scripts/TEST_4_23/Segmentation_TEST/Images_Plots/predicted_masks/test_{i}_dice_{dice_test[i]:.2f}.png')
# #     #     plt.close(fig)
        
# #     return dice_arr.mean(), outputs

#