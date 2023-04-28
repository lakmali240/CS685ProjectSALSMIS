import os
import torch
from data import load_dataset, random_select_data, repre_select_data, repre_select_labeled_and_unlabeled_data
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from model import UNet
import numpy as np
import torch.optim as optim
import logging
from tqdm import tqdm
import sys
import random
from configure import get_arguments
import time
from utils import losses
from utils.logger import get_cur_time,checkpoint_save
from utils.lr import adjust_learning_rate,cosine_rampdown
import binary as mmb
import pdb
from datetime import datetime
import pytz
from test import test
import torch.nn.functional as F

# =================================================
#         load data path and get arguments
# =================================================
timenow = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')),'%Y-%m-%d_%H-%M-%S')
basedir = '/content/drive/MyDrive/Spring_research_2023/overall_result' # save path
if not os.path.exists(basedir):
    os.makedirs(basedir)

args = get_arguments()

args.loss = args.loss.strip()
batch_size = args.batch_size
base_lr = args.lr
num_classes = args.num_classes
img_size = [256,256]
pre_train_type = args.pre_train_type.strip()
load_idx = args.load_idx

# make dir
logdir = os.path.join(basedir, 'logs', str(args.select_type)+'_'+str(args.select_num), timenow)
print(logdir)
savedir = os.path.join(basedir, 'checkpoints', str(args.select_type)+'_'+str(args.select_num), timenow)
print(savedir)
shotdir = os.path.join(basedir, 'snapshot',str(args.select_type)+'_'+str(args.select_num), timenow)
print(shotdir)

os.makedirs(logdir, exist_ok=False)
os.makedirs(savedir, exist_ok=False)
os.makedirs(shotdir, exist_ok=False)

writer = SummaryWriter(logdir)

logging.basicConfig(filename=shotdir+"/"+"snapshot.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.info(str(args))

# =================================================
#                 data loader
# =================================================

imgspath_train = '/content/drive/MyDrive/Spring_research_2023/data/GrayData/imgs_train.npy'
maskspath_train = '/content/drive/MyDrive/Spring_research_2023/data/GrayData/imgs_mask_train.npy'
imgspath_test = '/content/drive/MyDrive/Spring_research_2023/data/GrayData/imgs_test.npy'
maskspath_test = '/content/drive/MyDrive/Spring_research_2023/data/GrayData/imgs_mask_test.npy'

# imgspath_train = args.data_path_train + '/imgs_train.npy'
# maskspath_train = args.data_path_train + '/imgs_mask_train.npy'
# imgspath_test = args.data_path_train + '/imgs_test.npy'
# maskspath_test = args.data_path_train+ '/imgs_mask_test.npy'

dataset = load_dataset(imgspath_train,maskspath_train,[256,256])
total_num = len(dataset)
train_dataset = dataset[0:args.train_num] # 1600 refer to CEAL as unlabeled pool
test_set = dataset[args.train_num:total_num]

test_set2 = load_dataset(imgspath_test,maskspath_test,[256,256])

    
print('\nSize of training set: {}'.format(len(train_dataset)))
print('Size of test set: {}\n'.format(len(test_set)))

def kl_divergence_seg(outSeg, outSegUnlabeled):
  p = F.softmax(outSeg, dim = 1)
  log_p = F.log_softmax(outSeg, dim = 1)
  log_q = F.log_softmax(outSegUnlabeled, dim = 1)
  kl = p * (log_p - log_q)
  
  return kl.mean()

def train(labeled_train_set,unlabeled_train_set,weight_path,num_epochs,pre_train):
    labeled_trainloader  = DataLoader(labeled_train_set, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True)
    unlabeled_trainloader   = DataLoader(unlabeled_train_set, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True)

    # Combine labeled and unlabeled data loaders into a single data loader
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    # if labeled_iter is None:
    #     labeled_iter = iter(train_set_labeled)
    # if unlabeled_iter is None:
    #     unlabeled_iter = iter(unlabeled_trainloader)
    
    def combine_labeled_unlabeled_data(labeled_iter,unlabeled_iter):
        labeled_batch = next(labeled_iter, None)
        unlabeled_batch = next(unlabeled_iter, None)
        if labeled_batch is None:
            labeled_iter = iter(labeled_trainloader)
            labeled_batch = next(labeled_iter)
        if unlabeled_batch is None:
            unlabeled_iter = iter(unlabeled_trainloader)
            unlabeled_batch = next(unlabeled_iter)
        return labeled_batch, unlabeled_batch
    
    model = UNet(n_channels=1, n_classes=num_classes)
# =================================================
#           load pre-trained weights
# =================================================
    # load pre-trained weights from self-supervised learning
    if pre_train == 'init': # load all weights except the last output layer
        pretext_model = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in pretext_model.items() if 'outc' not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)   
        print('\nload init weight from: ',weight_path)
    elif pre_train == 'encoder':    
        pretext_model = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in pretext_model.items() if 'outc' not in k if 'up' not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)   
        print('\nload encoder weight from: ',weight_path)
    elif pre_train == 'all':
        model.load_state_dict(torch.load(weight_path))
        print('\nload all weight from: ',weight_path)
        
    model.train()
    
    if args.SGD:
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              momentum=0.9, weight_decay=0.0001)
    if args.Adam:
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        
    # ------Define the loss functions-------
    #For labeled Data
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    #For Unlabeled Data
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    # kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    #---------------------------------------------
    
    logging.info("{} iterations per epoch".format(len(labeled_trainloader)))
    print('{} iterations per expoch'.format(len(labeled_trainloader)))
    
# =================================================
#                 training
# =================================================
    iter_num = 0
    best_performance = 0.0
    performance = 0.0


    for epoch_num in range(num_epochs):
        print("Epoch %d / %d : " %(epoch_num+1,num_epochs))

        loss_epoch_labeled  = 0
        dice_epoch_labeled  = 0
        loss_epoch_unlabeled = 0
        loss_epoch_total = 0
        
        supervised_loss = 0
        unsupervised_loss = 0
        total_loss = 0
        
        
        # Iterate over the labeled training data and unlabeled data
        for i_batch in tqdm(range(len(labeled_trainloader))):
            labeled_batch, unlabeled_batch = combine_labeled_unlabeled_data(labeled_iter,unlabeled_iter)
            
            #For labeled data
            volume_batch, label_batch = labeled_batch['image'], labeled_batch['label']
            # volume_batch, label_batch = volume_batch.type(torch.FloatTensor), labeled_batch.type(torch.LongTensor)
            volume_batch = volume_batch.type(torch.FloatTensor)
            label_batch = label_batch.type(torch.LongTensor)

            #For unlabeled data
            unlabeled_volume_batch = unlabeled_batch['image']
            unlabeled_volume_batch = unlabeled_volume_batch.type(torch.FloatTensor)
            
            optimizer.zero_grad()
            
            #Supervised loss
            outputs_labeled = model(volume_batch) 
            loss_ce = ce_loss(outputs_labeled, label_batch)
            loss_dice, _ = dice_loss(outputs_labeled, label_batch)

            if args.loss == 'ce':
                loss = loss_ce
            elif args.loss == 'dice':
                loss = loss_dice
            elif args.loss == 'ce_dice':
                supervised_loss = 5 * (loss_dice + loss_ce)

            dice_epoch_labeled  += 1 - loss_dice.item()
            loss_epoch_labeled  += supervised_loss.item()
            
            #Unsupervised loss
            pseudo_threshold = 0.8
            with torch.no_grad():
                outputs_unlabeled = model(unlabeled_volume_batch)
                pseudo_labels = torch.softmax(outputs_unlabeled, dim=1)
                max_values, max_indices = torch.max(pseudo_labels, dim=1)
                mask = max_values.ge(pseudo_threshold).float()
            # Calculate KL divergence loss using pseudo-labels and mask
            KL_div_loss = kl_divergence_seg(outputs_unlabeled, pseudo_labels)
            # KL_div_loss = kl_loss(outputs_unlabeled, pseudo_labels.detach())
            # KL_div_loss = (KL_div_loss.sum(dim=0) * mask).mean()
            unsupervised_loss = KL_div_loss
            loss_epoch_unlabeled += KL_div_loss.sum().item()

            # total loss
            total_loss = supervised_loss + 0.05 * unsupervised_loss
            loss_epoch_total  += total_loss.item()
            
            # optimizer.zero_grad()
            
            total_loss.backward()
            optimizer.step()
            
            iter_num = iter_num + 1
            logging.info(
                'iteration %d : total_loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, total_loss.item(), loss_ce.item(), loss_dice.item()))

        # print("total_loss:",total_loss)
        
        total_length = len(labeled_trainloader)+len(unlabeled_trainloader)
        #losses for one epoch
        epoch_loss_total = loss_epoch_total/total_length
        print("epoch_loss_total",epoch_loss_total)
        writer.add_scalar('info/total_loss', epoch_loss_total,epoch_num +1)
        
        epoch_dice_labeled = dice_epoch_labeled / len(labeled_trainloader)
        epoch_loss_labeled = loss_epoch_labeled / len(labeled_trainloader)
        epoch_loss_unlabeled = loss_epoch_unlabeled / len(unlabeled_trainloader)
        
        # print("epoch_loss_labeled",epoch_loss_labeled)
        # writer.add_scalar('info/dice', epoch_dice_labeled, epoch_num +1)
        
        # print("epoch_loss_unlabeled",epoch_loss_unlabeled)
        # epoch_loss = epoch_loss_labeled + 0.05*epoch_loss_unlabeled
        # print("epoch_loss",epoch_loss)
        # writer.add_scalar('info/loss', epoch_loss, epoch_num + 1)
        

        logging.info('epoch %d : total_loss : %f dice: %f' % (epoch_num+1, epoch_loss_total, epoch_dice_labeled))
        print('epoch {} : total_loss : {} dice: {}'.format(epoch_num+1, epoch_loss_total, epoch_dice_labeled))

        checkpoint_path = checkpoint_save(model, True, savedir)
        logging.info("save model to {}".format(savedir))


    if args.Adam:
        writer.add_hparams({'log_dir':logdir, 'loss_func': args.loss,'optimizer': 'Adam', 'lr': args.lr, 'batch_size': args.batch_size, 'img_size':args.img_size,  'num_epoch':num_epochs}, {'val_dice': best_performance })
    elif args.SGD:
        writer.add_hparams({'log_dir':logdir, 'loss_func': args.loss,'optimizer': 'SGD', 'lr': args.lr, 'batch_size': args.batch_size, 'img_size':args.img_size,  'num_epoch':num_epochs}, {'val_dice': best_performance })
    print('========================================')
    print(pre_train_type) 
    dice_mean,outputs_test = test(test_set, checkpoint_path)
    print('Test epoch {} : dice: {}'.format(epoch_num+1, dice_mean))
    return checkpoint_path, dice_mean,outputs_test
    # return checkpoint_path

def main():
    for round_idx in range(args.round):
        print('\n\n====================== round {} ==========================\n\n'.format(round_idx))
        dice_mean_list = []
        
        # criteria of inital sample selection
        if args.select_type == 'random':
            if args.load_idx == 'create': # random select
                random_index = random.sample(range(0,len(train_dataset)),len(train_dataset))
                idx_path = os.path.join(basedir,'random_select_index', timenow)
                os.makedirs(idx_path, exist_ok=False)
                load_idx = idx_path+'/index.npy'
                np.save(load_idx,random_index)
                print('Create random load index at:',load_idx)
            else: # use a determined list
                load_idx = args.load_idx
                print('random load index from: ',args.load_idx)
                
            train_set = random_select_data(train_dataset,args.select_num,load_idx)

        # select by clustering results
        elif args.select_type == 'select':
            # train_set = repre_select_data(train_dataset,args.select_num,args.cluster_npy_path,args.cluster_idx_path)
            train_set_labeled, train_set_unlabeled = repre_select_labeled_and_unlabeled_data(train_dataset,args.select_num,args.cluster_npy_path,args.cluster_idx_path)

        print('----------Labeled images---------')     
        print('train_labeled size: {} '.format(len(train_set_labeled)))
        print((train_set_labeled[0]['image']).shape,(train_set_labeled[0]['label']).shape)
        
        print('----------Unlabeled images---------')
        print('train_unlabeled size: {} '.format(len(train_set_unlabeled)))
        shape1 = (train_set_unlabeled[0]['image']).shape
        shape2 = (train_set_unlabeled[0]['label']).shape
        print(f'train unlabeled image size: {shape1}')


        print('\n\n-------------- initialization training -----------------\n\n')
        select_num = args.select_num
        if pre_train_type == 'load':
            weight,dice_mean,outputs = train(train_set_labeled,train_set_unlabeled,args.weight_path,args.num_epochs,'init')
        elif pre_train_type == 'continue':
            weight,dice_mean,outputs = train(train_set_labeled,train_set_unlabeled,args.weight_path,args.num_epochs,'all')
        elif pre_train_type == 'encoder':
            weight,dice_mean,outputs = train(train_set_labeled,train_set_unlabeled,args.weight_path,args.num_epochs,'encoder')
        else:
            weight,dice_mean,outputs = train(train_set_labeled,train_set_unlabeled,args.weight_path,args.num_epochs,None)
        dice_mean_list.append(dice_mean)
        for i in range(args.iteration_num):
            print('\n-------------- training iteration {} -----------------\n\n'.format(i+1))
            select_num = select_num + args.aug_samples
            if args.select_type == 'random':
                train_set = random_select_data(train_dataset,select_num,load_idx)
            elif args.select_type == 'select':
                 train_set = repre_select_data(train_dataset,select_num,args.cluster_npy_path,args.cluster_idx_path)
            print('Train_set size: {} '.format(len(train_set)))
            print((train_set[0]['image']).shape,(train_set[0]['label']).shape)

            weight,dice_mean,outputs = train(train_set_labeled,train_set_unlabeled,weight,args.active_epochs,'all')
            dice_mean_list.append(dice_mean)
        
        cluster_path = args.cluster_npy_path
        cluster_path = cluster_path.split('/')[-2]
        list_path = os.path.join(args.result_path,str(args.select_type)+'_'+str(args.select_num)+'_'+pre_train_type,cluster_path)
        os.makedirs(list_path, exist_ok=True)
        np.save(os.path.join(list_path,str(round_idx)+'_'+timenow+'.npy'), dice_mean_list)
        print('save dice result path: ',os.path.join(list_path,str(round_idx)+'_'+timenow+'.npy'))
        
    dice_mean,outputs_test = test(test_set2, weight)
    # print('Test epoch {} : dice: {}'.format(epoch_num+1, dice_mean)
    
    writer.close()
    

    

if __name__ == "__main__":
    main()

    


