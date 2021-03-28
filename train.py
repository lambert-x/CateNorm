import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torch import optim
from tqdm import tqdm
import pandas as pd
from eval import eval_net
from unet import DualNorm_Unet
from dice_loss import DiceLoss
from weighted_ce_loss import Weighted_Cross_Entropy_Loss
from torch.utils.tensorboard import SummaryWriter

from unet.unet_parts import *
from torch.utils.data import DataLoader, random_split
from data_loader import SiteSet
import numpy as np
from sklearn.model_selection import KFold
import datetime


# dir_img = 'data/imgs/'
# dir_mask = 'data/masks/'


def readlist(datalist):
    with open(datalist, 'r') as fp:
        rows = fp.readlines()
    image_list = np.array([row.strip() for row in rows])
    return image_list


def weights_init(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def train_net(net,
              device,
              output_dir,
              train_date='',
              epochs=20,
              iters=900,
              bs=4,
              lr=0.01,
              save_cp=True,
              only_lastandbest=False,
              eval_freq=5,
              fold_idx=None,
              site='A',
              eval_site=None,
              gpus=None,
              save_folder='',
              aug=False,
              zoom=False,
              whitening=True,
              nonlinear='relu',
              norm_type='BN',
              pretrained=False,
              loaded_model_file_name='model_best',
              spade_seg_mode='soft',
              spade_inferred_mode='mask',
              spade_aux_blocks='',
              freeze_except=None,
              ce_weighted=False,
              spade_reduction=2,
              excluded_classes=None,
              dataset_name=None
              ):
    net.apply(weights_init)
    logging.info('Model Parameters Reset!')
    net.to(device=device)

    if pretrained:
        pretrained_model_dir = pretrained + f'/Fold_{fold_idx}/{loaded_model_file_name}.pth'
        pretrained_dict_load = torch.load(pretrained_model_dir)
        model_dict = net.state_dict()



        pretrained_dict = {}
        for k, v in pretrained_dict_load.items():
            if (k in model_dict) and (model_dict[k].shape == pretrained_dict_load[k].shape):
                pretrained_dict[k] = v

        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)


        print('Freeze Excluded Keywords:', freeze_except)
        if freeze_except is not None:
            for k, v in net.named_parameters():
                v.requires_grad = False
                for except_key in freeze_except:
                    if except_key in k:
                        v.requires_grad = True
                        print(k, ' requires grad')
                        break
        logging.info(f'Model loaded from {pretrained_model_dir}')

    pretrain_suffix = ''
    if pretrained:
        pretrain_suffix = f'_Pretrained'

    block_names = ['inc', 'down1', 'down2', 'down3', 'down4', 'mid', 'up1', 'up2', 'up3', 'up4']

    spade_blocks_suffix = ''
    if spade_aux_blocks != '':
        if spade_inferred_mode == 'mask':
            spade_blocks_suffix += f'_SPADE_R{spade_reduction}_{spade_seg_mode}_Aux_'
        else:
            spade_blocks_suffix += f'_SPADE_R{spade_reduction}_{spade_inferred_mode}_Aux_'

        for blockname in spade_aux_blocks:
            block_idx = block_names.index(blockname)
            spade_blocks_suffix += str(block_idx)





    excluded_classes_suffix = ''

    if excluded_classes is not None:
        excluded_classes_string = [str(c) for c in excluded_classes]
        excluded_classes_suffix = '_exclude' + ''.join(excluded_classes_string)

    dir_results = output_dir

    tensorboard_logdir = dir_results + 'logs/' + f'{save_folder}/' + f'{train_date}_Site_{site}_GPUs_{gpus}/' + \
                         f'BS_{bs}_Epochs_{epochs}_Aug_{aug}_Zoom_{zoom}_Nonlinear_{nonlinear}_Norm_{norm_type}' + \
                         excluded_classes_suffix + pretrain_suffix + spade_blocks_suffix + f'/Fold_{fold_idx}'
    writer = SummaryWriter(log_dir=tensorboard_logdir)

    dir_checkpoint = dir_results + 'checkpoints/' + f'{save_folder}/' + \
                     f'{train_date}_Site_{site}_GPUs_{gpus}/' + f'Epochs_{epochs}_Aug_{aug}_Zoom_{zoom}_Nonlinear_{nonlinear}_Norm_{norm_type}' + \
                       excluded_classes_suffix + pretrain_suffix + spade_blocks_suffix + f'/Fold_{fold_idx}' + '/'
    print(tensorboard_logdir)
    dir_eval_csv = dir_results + 'eval_csv/' + f'{save_folder}/' + \
                   f'{train_date}_Site_{site}_GPUs_{gpus}/' + f'Epochs_{epochs}_Aug_{aug}_Zoom_{zoom}_Nonlinear_{nonlinear}_Norm_{norm_type}' + \
                   excluded_classes_suffix + pretrain_suffix + spade_blocks_suffix + '/'
    csv_files_prefix = f'{train_date}_Site_{site}_GPUs_{gpus}_'
    print(dir_eval_csv)
    if not os.path.exists(dir_eval_csv):
        os.makedirs(dir_eval_csv)
    train_list = {}
    val_list = {}
    test_list = {}

    train_list['Overall'] = []
    val_list['Overall'] = []
    test_list['Overall'] = []

    if eval_site is None:
        sites_inferred = list(site)
    else:
        sites_inferred = list(set(site + eval_site))
    sites_inferred.sort()
    print(sites_inferred)
    for site_idx in sites_inferred:
        train_list[site_idx] = all_list[site_idx][split_list[site_idx][fold_idx][0]].tolist()
        val_list[site_idx] = all_list[site_idx][split_list[site_idx][fold_idx][1]].tolist()
        test_list[site_idx] = all_list[site_idx][split_list[site_idx][fold_idx][1]].tolist()
        if site_idx in site:
            train_list['Overall'].append(train_list[site_idx])
            val_list['Overall'].append(val_list[site_idx])
        test_list['Overall'].append(test_list[site_idx])

    print('-----------------------------------------')
    print('Dataset Info:')
    for site_key in train_list.keys():
        if site_key in ['Overall', 'ABC_mixed']:
            case_total_train = 0
            case_total_test = 0
            for site_list_train, site_list_test in zip(train_list[site_key], test_list[site_key]):
                case_total_train += len(site_list_train)
                case_total_test += len(site_list_test)
            print(f'{site_key}: {len(train_list[site_key])} sites  '
                  f'Train: {case_total_train} cases, Test: {case_total_test} cases')
        else:
            print(f'Site {site_key} Train:  {len(train_list[site_key])} cases, Test:  {len(test_list[site_key])} cases')
    print('-----------------------------------------')
    n_train = iters * bs
    if len(site) > 1:
        train_set = SiteSet(train_list['Overall'], iters=n_train, training=True, augmentation=aug, source="Overall",
                            zoom_crop=zoom, whitening=whitening, batchsize=bs // len(site), site_num=len(site),
                            n_classes=net.n_classes, excluded_classes=excluded_classes)
        val_set = SiteSet(val_list['Overall'], iters=n_train, training=True, augmentation=False, source="Overall",
                          zoom_crop=False, whitening=whitening, batchsize=bs // len(site), site_num=len(site),
                          n_classes=net.n_classes, excluded_classes=excluded_classes)
    else:
        train_set = SiteSet(train_list[site], iters=n_train, training=True, augmentation=aug, source=site,
                            zoom_crop=zoom, whitening=whitening, batchsize=bs // len(site), site_num=len(site),
                            n_classes=net.n_classes, excluded_classes=excluded_classes)
        val_set = SiteSet(val_list[site], iters=n_train, training=True, augmentation=False, source=site,
                          zoom_crop=False, whitening=whitening, batchsize=bs // len(site), site_num=len(site),
                          n_classes=net.n_classes, excluded_classes=excluded_classes)


    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    global_step = 0

    logging.info(f'''Starting training:
        Output Path:     {output_dir}
        Epochs:          {epochs}
        Iterations:      {iters}
        Batch size:      {bs}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Only_Last_Best:  {only_lastandbest}
        Eval_Frequency:  {eval_freq}
        Device:          {device.type}
        GPU ids:         {gpus}
        Site:            {site}
        Shift+Rotation:  {aug}
        Zoom+Crop:       {zoom}
        Whitening:       {whitening}
        Pretrain:        {pretrained}
        Classes:         {net.n_classes}
        Excluded Clases: {excluded_classes}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                     patience=5)

    criterion = DiceLoss()

    best_score = 0
    csv_header = True
    for epoch in range(epochs):

        losses = []
        losses_first_forward = []
        epoch_loss = 0
        with tqdm(total=iters, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            # for batch in train_loader:
            time.sleep(2)
            for (train_batch, val_batch) in zip(train_loader, val_loader):
                net.train()
                imgs = train_batch[0]
                all_masks = train_batch[1]
                site_label = train_batch[2]

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                all_masks = all_masks.to(device=device, dtype=mask_type)


                masks_pred = net(imgs)
                if spade_aux_blocks == '':
                    loss, loss_hard = criterion(masks_pred, all_masks, num_classes=net.n_classes,
                                                return_hard_dice=True, softmax=True)
                    writer.add_scalar('Backwarded Loss/Dice_Loss', loss.item(), global_step)

                    if net.n_classes > 2:
                        loss_forward_weighted_ce = Weighted_Cross_Entropy_Loss()(masks_pred, all_masks,
                                                                                 num_classes=net.n_classes,
                                                                                 weighted=ce_weighted,
                                                                                 softmax=True)
                        loss += loss_forward_weighted_ce
                        writer.add_scalar('Backwarded Loss/Weighted_CE_Loss',
                                          loss_forward_weighted_ce.item(), global_step)
                    train_loss = loss_hard.item()
                else:
                    # start first forward
                    loss_first_forward, loss_first_forward_hard = criterion(masks_pred, all_masks,
                                                                            num_classes=net.n_classes,
                                                                            return_hard_dice=True,
                                                                            softmax=True)
                    referred_mask_loss = loss_first_forward_hard.item()
                    writer.add_scalar('Backwarded Loss/Dice_Loss_First', loss_first_forward.item(),
                                      global_step)
                    if net.n_classes > 2:
                        loss_first_forward_weighted_ce = Weighted_Cross_Entropy_Loss()(masks_pred,
                                                                                       all_masks,
                                                                                       num_classes=net.n_classes,
                                                                                       weighted=ce_weighted,
                                                                                       softmax=True)
                        loss_first_forward += loss_first_forward_weighted_ce
                        writer.add_scalar('Backwarded Loss/Weighted_CE_Loss_First',
                                          loss_first_forward_weighted_ce.item(), global_step)
                    optimizer.zero_grad()
                    loss_first_forward.backward()
                    optimizer.step()


                    # start second forward
                    mask_pred_first_forward = masks_pred.detach()
                    mask_pred_first_forward = torch.softmax(mask_pred_first_forward, dim=1)
                    masks_pred_second_forward = net(imgs, seg=mask_pred_first_forward)
                    loss, loss_hard = criterion(masks_pred_second_forward, all_masks,
                                                num_classes=net.n_classes, return_hard_dice=True,
                                                softmax=True)

                    writer.add_scalar('Backwarded Loss/Dice_Loss_Second', loss.item(), global_step)
                    if net.n_classes > 2:
                        loss_second_forward_weighted_ce = Weighted_Cross_Entropy_Loss()(
                            masks_pred_second_forward,
                            all_masks, num_classes=net.n_classes, weighted=ce_weighted, softmax=True)
                        loss += loss_second_forward_weighted_ce
                        writer.add_scalar('Backwarded Loss/Weighted_CE_Loss_Second',
                                          loss_second_forward_weighted_ce.item(), global_step)
                    train_loss = loss_hard.item()

                epoch_loss += train_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_postfix(**{'Dice': train_loss})
                pbar.update(1)
                global_step += 1


                net.eval()
                imgs = val_batch[0]
                all_masks = val_batch[1]
                site_label = val_batch[2]

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                all_masks = all_masks.to(device=device, dtype=mask_type)

                with torch.no_grad():
                    masks_pred = net(imgs)
                    # loss = criterion(masks_pred, all_masks)
                    # val_loss = loss.item()
                    if spade_aux_blocks == '':
                        loss, loss_hard = criterion(masks_pred, all_masks, num_classes=net.n_classes,
                                                    return_hard_dice=True, softmax=True)
                        val_loss = loss_hard.item()
                    else:
                        mask_pred_first_forward = masks_pred.detach()

                        _, loss_hard_first_forward = criterion(mask_pred_first_forward, all_masks,
                                                               num_classes=net.n_classes,
                                                               return_hard_dice=True, softmax=True)
                        mask_pred_first_forward = torch.softmax(mask_pred_first_forward, dim=1)
                        masks_pred_second_forward = net(imgs, seg=mask_pred_first_forward)
                        loss, loss_hard = criterion(masks_pred_second_forward, all_masks, num_classes=net.n_classes,
                                                    return_hard_dice=True, softmax=True)
                        val_loss = loss_hard.item()
                        val_loss_first_forward = loss_hard_first_forward.item()

                if spade_aux_blocks == '':
                    writer.add_scalars('Loss/Dice_Loss', {'train': train_loss,
                                                          'val': val_loss}, global_step)
                else:
                    writer.add_scalars('Loss/Dice_Loss', {'train': train_loss,
                                                          'val_first': val_loss_first_forward,
                                                          'val_second': val_loss
                                                          }, global_step)
                if global_step % (n_train // (eval_freq * bs)) == 0:
                    if net.n_classes == 2:
                        writer.add_images('masks/true', (all_masks[:4, ...][:4, ...].cpu().unsqueeze(1)),
                                          global_step)
                        writer.add_images('masks/pred',
                                          torch.softmax(masks_pred[:4, ...], dim=1)[:, 1:2, :, :].cpu(),
                                          global_step)

                    elif net.n_classes > 2:
                        writer.add_images('masks/true', (all_masks[:4, ...].cpu().unsqueeze(1)), global_step)
                        writer.add_images('masks/pred',
                                          torch.argmax(torch.softmax(masks_pred[:4, ...], dim=1), dim=1).unsqueeze(
                                              1).cpu(), global_step)
                if spade_aux_blocks != '':
                    losses_first_forward.append(loss_first_forward.item())

                losses.append(train_loss)

                if global_step % 50 == 0 and optimizer.param_groups[0]['lr'] > 1e-5:
                    scheduler.step(np.mean(losses[-50:]))
                if global_step % (n_train // (eval_freq * bs)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    if eval_site is None:
                        eval_site = site
                    test_scores, test_asds = eval_net(net, test_list, device, fold_idx, global_step, dir_eval_csv,
                                                      csv_files_prefix=csv_files_prefix, whitening=whitening,
                                                      eval_site=eval_site, spade_aux=(spade_aux_blocks != ''),
                                                      excluded_classes=excluded_classes, dataset_name=dataset_name
                                                      )
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    if epoch >= 0:
                        if len(site) == 1:
                            metric_site = site
                        else:
                            metric_site = 'Overall'
                        if net.n_classes == 2:
                            is_best = test_scores[metric_site] > best_score
                            best_score = max(test_scores[metric_site], best_score)
                            if is_best:
                                try:
                                    os.makedirs(dir_checkpoint)
                                    logging.info('Created checkpoint directory')
                                except OSError:
                                    pass
                                torch.save(net.state_dict(),
                                           dir_checkpoint + f'model_best.pth')
                                logging.info(
                                    f'Best model saved ! ( Dice Score:{best_score}) on Site {metric_site})')

                        elif net.n_classes > 2:
                            scores_all_classes = 0
                            for c in test_scores.keys():
                                scores_all_classes += test_scores[c][metric_site]
                            scores_all_classes /= len(test_scores.keys())
                            is_best = scores_all_classes > best_score
                            best_score = max(scores_all_classes, best_score)
                            if is_best:
                                try:
                                    os.makedirs(dir_checkpoint)
                                    logging.info('Created checkpoint directory')
                                except OSError:
                                    pass

                                torch.save(net.state_dict(),
                                           dir_checkpoint + f'model_best.pth')
                                logging.info(f'Best model saved ! ( Dice Score:{best_score} on Site {metric_site})')
                    if net.n_classes == 2:
                        if len(eval_site) > 1:
                            sites_print = list(eval_site) + ['Overall']
                        else:
                            sites_print = list(eval_site)

                        test_performance_dict = {}
                        test_performance_dict['fold'] = fold_idx
                        test_performance_dict['global_step'] = global_step
                        for st in sites_print:
                            print('\nSite: {}'.format(st))
                            print('\nTest Dice Coeff: {}'.format(test_scores[st]))
                            print('\nTest ASD: {}'.format(test_asds[st]))
                        for st in sites_print:
                            test_performance_dict[f'Dice_{st}'] = [format(test_scores[st], '.4f')]
                        for st in sites_print:
                            test_performance_dict[f'ASD_{st}'] = [format(test_asds[st], '.2f')]
                        if spade_aux_blocks != '':
                            for st in sites_print:
                                test_performance_dict[f'Dice_{st}_first_forward'] = [
                                    format(test_scores[st + '_first_forward'], '.4f')]
                            for st in sites_print:
                                test_performance_dict[f'ASD_{st}_first_forward'] = [
                                    format(test_asds[st + '_first_forward'], '.2f')]
                        df = pd.DataFrame.from_dict(test_performance_dict)
                        df.to_csv(
                            dir_eval_csv + csv_files_prefix + f'site_performance.csv',
                            mode='a', header=csv_header, index=False)
                        csv_header = False
                        print('\n' + tensorboard_logdir)
                        # write with tensorboard
                        writer.add_scalars('Test/Dice_Score', test_scores, global_step)
                        writer.add_scalars('Test/ASD', test_asds, global_step)

                    elif net.n_classes > 2:
                        if dataset_name == 'ABD-8':
                            abdominal_organ_dict = {1: 'spleen', 2: 'r_kidney', 3: 'l_kidney', 4: 'gallbladder',
                                                    5: 'pancreas', 6: 'liver', 7: 'stomach', 8: 'aorta'}
                            if excluded_classes is None:
                                organ_dict = abdominal_organ_dict
                            else:
                                print('Original Organ dict')
                                print(abdominal_organ_dict)
                                post_mapping_dict = {}
                                original_classes = list(range(net.n_classes + len(excluded_classes)))
                                remain_classes = [item for item in original_classes if item not in excluded_classes]
                                for new_value, value in enumerate(remain_classes):
                                    post_mapping_dict[value] = new_value
                                organ_dict = {}
                                for c in remain_classes:
                                    if c == 0:
                                        continue
                                    organ_dict[post_mapping_dict[c]] = abdominal_organ_dict[c]
                                print('Current Organ dict')
                                print(organ_dict)
                        elif dataset_name == 'ABD-6':
                            abdominal_organ_dict = {1: 'spleen', 2: 'l_kidney', 3: 'gallbladder', 4: 'liver',
                                                    5: 'stomach', 6: 'pancreas'}
                            if excluded_classes is None:
                                organ_dict = abdominal_organ_dict
                            else:
                                print('Original Organ dict')
                                print(abdominal_organ_dict)
                                post_mapping_dict = {}
                                original_classes = list(range(net.n_classes + len(excluded_classes)))
                                remain_classes = [item for item in original_classes if item not in excluded_classes]
                                for new_value, value in enumerate(remain_classes):
                                    post_mapping_dict[value] = new_value
                                organ_dict = {}
                                for c in remain_classes:
                                    if c == 0:
                                        continue
                                    organ_dict[post_mapping_dict[c]] = abdominal_organ_dict[c]
                                print('Current Organ dict')
                                print(organ_dict)
                        print(f'Organ Dict:{organ_dict}')
                        test_performance_dict = {}
                        test_performance_dict['fold'] = fold_idx
                        test_performance_dict['global_step'] = global_step


                        for organ_class in range(1, net.n_classes):
                            if len(eval_site) > 1:
                                sites_print = list(eval_site) + ['Overall']
                            else:
                                sites_print = list(eval_site)

                            for st in sites_print:
                                test_performance_dict[f'{organ_dict[organ_class]}_Dice_{st}'] = [
                                    format(test_scores[organ_class][st], '.4f')]
                            for st in sites_print:
                                test_performance_dict[f'{organ_dict[organ_class]}_ASD_{st}'] = [
                                    format(test_asds[organ_class][st], '.2f')]

                            writer.add_scalars(f'Test/Dice_Score_{organ_dict[organ_class]}',
                                               test_scores[organ_class], global_step)
                            writer.add_scalars(f'Test/ASD_{organ_dict[organ_class]}',
                                               test_asds[organ_class], global_step)

                        test_scores['AVG'] = {}
                        test_asds['AVG'] = {}
                        for st in sites_print:
                            scores_all_classes_avg = 0
                            asds_all_classes_avg = 0
                            for c in range(1, net.n_classes):
                                scores_all_classes_avg += test_scores[c][st]
                                asds_all_classes_avg += test_asds[c][st]
                            scores_all_classes_avg /= (net.n_classes - 1)
                            asds_all_classes_avg /= (net.n_classes - 1)
                            test_scores['AVG'][st] = scores_all_classes_avg
                            test_asds['AVG'][st] = asds_all_classes_avg
                            print(f'\nSite:{st}')
                            print(f'Average:')
                            print('Test Dice Coeff: {}'.format(test_scores['AVG'][st]))
                            print('Test ASD: {}'.format(test_asds['AVG'][st]))
                        for st in sites_print:
                            test_performance_dict[f'Average_Dice_{st}'] = [
                                format(test_scores['AVG'][st], '.4f')]
                        for st in sites_print:
                            test_performance_dict[f'Average_ASD_{st}'] = [
                                format(test_asds['AVG'][st], '.2f')]
                        writer.add_scalars(f'Test/Dice_Score_Average',
                                           test_scores['AVG'], global_step)
                        writer.add_scalars(f'Test/ASD_Average',
                                           test_asds['AVG'], global_step)

                        if spade_aux_blocks != '':
                            for organ_class in range(1, net.n_classes):
                                for st in sites_print:
                                    test_performance_dict[
                                        f'{organ_dict[organ_class]}_Dice_{st}_first_forward'] = [
                                        format(test_scores[organ_class][st + '_first_forward'], '.4f')]
                                for st in sites_print:
                                    test_performance_dict[
                                        f'{organ_dict[organ_class]}_ASD_{st}_first_forward'] = [
                                        format(test_asds[organ_class][st + '_first_forward'], '.2f')]
                        df = pd.DataFrame.from_dict(test_performance_dict)
                        df.to_csv(
                            dir_eval_csv + csv_files_prefix + f'site_performance.csv',
                            mode='a', header=csv_header, index=False)
                        csv_header = False
                        print('\n' + tensorboard_logdir)

        if save_cp:
            try:
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if (epoch + 1) < epochs:
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')


            else:
                torch.save(net.state_dict(),
                           dir_checkpoint + f'model_last.pth')

            if only_lastandbest:
                if os.path.exists(dir_checkpoint + f'CP_epoch{epoch}.pth'):
                    os.remove(dir_checkpoint + f'CP_epoch{epoch}.pth')
            print(dir_checkpoint)
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-i', '--iters', metavar='I', type=int, default=900,
                        help='Number of iters per epoch', dest='iters')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--site', type=int, default=0,
                        help='Choose the site(s), A,B,C (1,2,3) or Overall(0)')
    parser.add_argument('--gpu', type=str, default='0', help='train or test or guide')
    parser.add_argument('--save-folder', type=str, default='',
                        help='the output folder under the output directory to save checkpoints and logs')
    parser.add_argument('--aug', type=str2bool, default=True,
                        help='Use Image augmentation (shift and rotation) or not')
    parser.add_argument('--zoom', type=str2bool, default=False,
                        help='Use Image augmentation (random zoom then center crop) or not')
    parser.add_argument('--whitening', type=str2bool, default=True,
                        help='Use Whitening to preprocess images or not')
    parser.add_argument('--server', type=str, default='local-prostate', help='change mappings for different servers')
    parser.add_argument('--net', type=str, default='DNUnet',
                        help='choose network architecture')
    parser.add_argument('--fold', nargs='+', type=int, default=-1,
                        help='Choose the k-fold setting, default value:-1 means all 5 fold, otherwise choose the typed index folds '
                             '(e.g. 0; 0 1; 0 3 4)')
    parser.add_argument('--nonlinear', type=str, default='relu',
                        help='choose the non-linear function as activation layers')
    parser.add_argument('--sitename', nargs='+', type=str, default=None)
    parser.add_argument('--n-classes', type=int, default=2,
                        help='the number of classes (including background)')
    parser.add_argument('--save-lastbest', type=str2bool, default=True, help='only save the last or best checkpoints')
    parser.add_argument('--eval-freq', type=int, default=2,
                        help='checkpoint saving frequency every epoch')
    parser.add_argument('--eval-site', type=str, default=None)
    parser.add_argument('--norm-type', type=str, default='BN',
                        help='choose the type of normalization')
    parser.add_argument('--spade-seg-mode', type=str, default='soft',
                        help='use soft or hard semantic mask')
    parser.add_argument('--spade-inferred-mode', type=str, default='mask',
                        help='use mask/features/features_normed to get the normalization params')
    parser.add_argument('--spade-aux-blocks', nargs='+', type=str, default='',
                        help='select blocks for using auxilary spatially-adaptive normalization(SPADE)')
    parser.add_argument('--freeze-except', nargs='+', type=str, default=None,
                        help='keywords except for freezing ')
    parser.add_argument('--loaded-model-name', type=str, default='model_last',
                        help='the file name of the model to be loaded')
    parser.add_argument('--ce-weighted', type=str2bool, default=False, help='whether use weighted Cross Entropy loss')
    parser.add_argument('--spade-reduction', type=int, default=2,
                        help='The reduction ratio for the SPADE hidden layer')
    parser.add_argument('--excluded-classes', nargs='+', type=int, default=None,
                        help='set a list of index for mask excluding')

    return parser.parse_args()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data_dir = {'local-prostate': 'G:/Dataset/Prostate_Multi_Site',
                'local-ABD-8': 'G:/Dataset/Abdominal_Single_Site_8organs',
                'local-ABD-6': 'G:/Dataset/Abdominal_Multi_Site_6organs',
                }

    save_dir = {'local-prostate': 'G:/DualNorm-Unet/',
                'local-ABD-8': 'G:/DualNorm-Unet/',
                'local-ABD-6': 'G:/DualNorm-Unet/',
                }
    data_root = data_dir[args.server]
    output_root = save_dir[args.server]

    print('Dataset Path:', data_root)

    sitename = ''.join(args.sitename)

    if args.eval_site is None:
        eval_site = sitename
    else:
        eval_site = args.eval_site
    all_list = {}
    for folder in sorted(os.listdir(data_root)):
        if folder[-1] in sitename or folder[-1] in eval_site:
            all_list[folder[-1]] = readlist(data_root + f'/{folder}/all_list.txt')
    kf = {}
    for idx, site in enumerate(all_list.keys()):
        kf[site] = KFold(n_splits=5, shuffle=True, random_state=idx)

    split_list = {}
    for site in all_list.keys():
        split_list[site] = list(kf[site].split(all_list[site]))


    logging.info(f'Using device :{args.gpu}')

    site_num = len(all_list.keys())

    if args.net == 'DNUnet':
        net = DualNorm_Unet(n_channels=3, n_classes=args.n_classes, bilinear=False, batchsize=args.batchsize // site_num,
                       nonlinear=args.nonlinear, norm_type=args.norm_type, spade_seg_mode=args.spade_seg_mode,
                       spade_inferred_mode=args.spade_inferred_mode, spade_aux_blocks=args.spade_aux_blocks,
                       spade_reduction=args.spade_reduction)

    print(net)
    print('Network Architecture:', net.__class__.__name__)
    print('# Network Parameters:', sum(param.numel() for param in net.parameters()))
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.fold == -1:
        selected_folds = [0, 1, 2, 3, 4]
    else:
        selected_folds = args.fold

    train_date = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print('Folds INFO:')
    print(selected_folds)

    print(f'Evaluate Sites: {args.eval_site}')
    for f_idx in selected_folds:
        try:
            train_net(net=net,
                      output_dir=output_root,
                      train_date=train_date,
                      epochs=args.epochs,
                      iters=args.iters,
                      only_lastandbest=args.save_lastbest,
                      eval_freq=args.eval_freq,
                      bs=args.batchsize,
                      lr=args.lr,
                      device=device,
                      fold_idx=f_idx,
                      site=sitename,
                      eval_site=args.eval_site,
                      gpus=args.gpu,
                      save_folder=args.save_folder,
                      aug=args.aug,
                      zoom=args.zoom,
                      whitening=args.whitening,
                      nonlinear=args.nonlinear,
                      norm_type=args.norm_type,
                      pretrained=args.load,
                      loaded_model_file_name=args.loaded_model_name,
                      spade_seg_mode=args.spade_seg_mode,
                      spade_inferred_mode=args.spade_inferred_mode,
                      spade_aux_blocks=args.spade_aux_blocks,
                      freeze_except=args.freeze_except,
                      ce_weighted=args.ce_weighted,
                      spade_reduction=args.spade_reduction,
                      excluded_classes=args.excluded_classes,
                      dataset_name=args.server[args.server.find('-')+1:]
                      )
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
