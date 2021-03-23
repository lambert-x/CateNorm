import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from dice_loss import DiceLoss
from medpy import metric
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader
from data_loader import SiteSet

import time
import os
import pandas as pd
import SimpleITK as sitk


def eval_net(net, test_list, device, fold_idx, global_step, dir_eval_csv, csv_files_prefix='',
             whitening=True, eval_site='ABC', spade_aux=False, save_prediction=False, nii_save_path=None,
             excluded_classes=None, dataset_name=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()

    if net.n_classes == 2:
        scores = {}
        asds = {}
        if spade_aux:
            scores_first = []
            scores_second = []
            asds_first = []
            asds_second = []
    elif net.n_classes > 2:
        scores = {}
        asds = {}
        for c in range(1, net.n_classes):
            scores[c] = {}
            asds[c] = {}
        if spade_aux:
            scores_first = {}
            asds_first = {}
            scores_second = {}
            asds_second = {}
            for c in range(1, net.n_classes):
                scores_first[c] = []
                asds_first[c] = []
                scores_second[c] = []
                asds_second[c] = []

    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    csv_header = True
    for idx, site in enumerate(list(eval_site)):
        n_cases = len(test_list[site])
        if net.n_classes == 2:
            tot = 0
            asd = 0
            if spade_aux:
                tot_first_forward = 0
                asd_first_forward = 0
        elif net.n_classes > 2:
            tot_multi = {}
            asd_multi = {}
            for c in range(1, net.n_classes):
                tot_multi[c] = 0
                asd_multi[c] = 0
            if spade_aux:
                tot_multi_first_forward = {}
                asd_multi_first_forward = {}
                for c in range(1, net.n_classes):
                    tot_multi_first_forward[c] = 0
                    asd_multi_first_forward[c] = 0
        with tqdm(total=n_cases, desc=f'Test round : Site{site}', unit='case', leave=True) as pbar:
            for case in test_list[site]:
                case_performance_dict = {}
                case_performance_dict['fold'] = fold_idx
                case_performance_dict['global_step'] = global_step
                case_performance_dict['site'] = site
                case_performance_dict['Case'] = os.path.basename(case.split(',')[0][:-7])

                test_set = SiteSet([case], training=False, augmentation=False, zoom_crop=False, whitening=whitening,
                                   source=site, n_classes=net.n_classes, excluded_classes=excluded_classes)

                loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
                true_masks = np.array([]).reshape(0, 384, 384)
                pred_masks = np.array([]).reshape(0, net.n_classes, 384, 384)
                pred_masks_first_forward = np.array([]).reshape(0, net.n_classes, 384, 384)



                for batch in loader:
                    img = batch[0]
                    true_mask = batch[1]
                    img = img.to(device=device, dtype=torch.float32)
                    true_mask = true_mask.to(device=device, dtype=mask_type)
                    with torch.no_grad():
                        if not spade_aux:  # no aux spade norm
                            pred_mask = net(img)
                        elif spade_aux:
                            pred_mask_first_forward = net(img)
                            pred_mask_first_forward = torch.softmax(pred_mask_first_forward, dim=1)
                            pred_mask = net(img, seg=pred_mask_first_forward)

                    pred_mask = torch.softmax(pred_mask, dim=1)
                    true_masks = np.concatenate((true_masks, true_mask.cpu().numpy()), 0)
                    pred_masks = np.concatenate((pred_masks, pred_mask.cpu().numpy()), 0)
                    if spade_aux:
                        pred_masks_first_forward = np.concatenate(
                            (pred_masks_first_forward, pred_mask_first_forward.cpu().numpy()), 0)

                if net.n_classes == 2:
                    true_masks = true_masks.transpose([1, 2, 0])

                    pred_masks = pred_masks[:, 1, ...].transpose([1, 2, 0])
                    pred_masks = (pred_masks > 0.5).astype(float)
                    pred_masks = _connectivity_region_analysis(pred_masks)
                    case_tot = (1 - _eval_dice(pred_masks, true_masks))
                    case_asd = (metric.binary.asd(pred_masks, true_masks))
                    tot += case_tot
                    asd += case_asd
                    if not spade_aux:
                        case_performance_dict['Dice'] = [format(case_tot, '.4f')]
                        case_performance_dict['ASD'] = [format(case_asd, '.2f')]
                    else:
                        case_performance_dict['Dice_second_forward'] = [format(case_tot, '.4f')]
                        case_performance_dict['ASD_second_forward'] = [format(case_asd, '.2f')]
                        pred_masks_first_forward = pred_masks_first_forward[:, 1, ...].transpose([1, 2, 0])
                        pred_masks_first_forward = (pred_masks_first_forward > 0.5).astype(float)
                        pred_masks_first_forward = _connectivity_region_analysis(pred_masks_first_forward)
                        case_tot_first_forward = (1 - _eval_dice(pred_masks_first_forward, true_masks))
                        case_asd_first_forward = (metric.binary.asd(pred_masks_first_forward, true_masks))
                        tot_first_forward += case_tot_first_forward
                        asd_first_forward += case_asd_first_forward
                        case_performance_dict['Dice_first_forward'] = [format(case_tot_first_forward, '.4f')]
                        case_performance_dict['ASD_first_forward'] = [format(case_asd_first_forward, '.2f')]

                    df = pd.DataFrame.from_dict(case_performance_dict)
                    df.to_csv(
                        dir_eval_csv + csv_files_prefix + 'case_performance.csv',
                        mode='a', header=csv_header, index=False)

                    csv_header = False
                    if save_prediction:
                        if not spade_aux:
                            zero_padding = np.zeros([384, 384, 1])
                            prediction_mask_array = np.concatenate((zero_padding, pred_masks, zero_padding), axis=2)
                            mask_gt_filename = os.path.basename(case.split(',')[1])
                            if nii_save_path is None:
                                nii_save_path = os.path.dirname(case.split(',')[1])

                            mask_gt = sitk.ReadImage(case.split(',')[1])
                            prediction_mask = sitk.GetImageFromArray(prediction_mask_array.transpose([2, 0, 1]))
                            prediction_mask.SetSpacing(mask_gt.GetSpacing())
                            prediction_mask.SetOrigin(mask_gt.GetOrigin())
                            prediction_mask.SetDirection(mask_gt.GetDirection())
                            mask_pred_filename = mask_gt_filename.replace('segmentation', 'prediction')
                            sitk.WriteImage(prediction_mask, os.path.join(nii_save_path, mask_pred_filename))
                        else:
                            zero_padding = np.zeros([384, 384, 1])
                            prediction_mask_array = np.concatenate((zero_padding, pred_masks, zero_padding), axis=2)
                            mask_gt_filename = os.path.basename(case.split(',')[1])
                            if nii_save_path is None:
                                nii_save_path = os.path.dirname(case.split(',')[1])

                            mask_gt = sitk.ReadImage(case.split(',')[1])
                            prediction_mask = sitk.GetImageFromArray(prediction_mask_array.transpose([2, 0, 1]))
                            prediction_mask.SetSpacing(mask_gt.GetSpacing())
                            prediction_mask.SetOrigin(mask_gt.GetOrigin())
                            prediction_mask.SetDirection(mask_gt.GetDirection())
                            mask_pred_filename = mask_gt_filename.replace('segmentation', '_second_forward_prediction')
                            sitk.WriteImage(prediction_mask, os.path.join(nii_save_path, mask_pred_filename))

                            prediction_mask_array = np.concatenate(
                                (zero_padding, pred_masks_first_forward, zero_padding), axis=2)

                            prediction_mask = sitk.GetImageFromArray(prediction_mask_array.transpose([2, 0, 1]))
                            prediction_mask.SetSpacing(mask_gt.GetSpacing())
                            prediction_mask.SetOrigin(mask_gt.GetOrigin())
                            prediction_mask.SetDirection(mask_gt.GetDirection())
                            mask_pred_filename = mask_gt_filename.replace('segmentation', '_first_forward_prediction')
                            sitk.WriteImage(prediction_mask, os.path.join(nii_save_path, mask_pred_filename))

                elif net.n_classes > 2:

                    abdominal_organ_dict = {1: 'spleen', 2: 'r_kidney', 3: 'l_kidney', 4: 'gallbladder',
                                            5: 'pancreas',
                                            6: 'liver', 7: 'stomach', 8: 'aorta'}
                    if excluded_classes is None:
                        organ_dict = abdominal_organ_dict
                    else:
                        if dataset_name == 'ABD-8':
                            abdominal_organ_dict = {1: 'spleen', 2: 'r_kidney', 3: 'l_kidney', 4: 'gallbladder',
                                                    5: 'pancreas',
                                                    6: 'liver', 7: 'stomach', 8: 'aorta'}
                            if excluded_classes is None:
                                organ_dict = abdominal_organ_dict
                            else:
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
                        elif dataset_name == 'ABD-6':
                            abdominal_organ_dict = {1: 'spleen', 2: 'l_kidney', 3: 'gallbladder', 4: 'liver',
                                                    5: 'stomach', 6: 'pancreas'}
                            if excluded_classes is None:
                                organ_dict = abdominal_organ_dict
                            else:
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


                    true_masks = torch.tensor(true_masks)
                    gt_onehot = F.one_hot(true_masks.long(), num_classes=net.n_classes)

                    pred_masks = torch.tensor(pred_masks)
                    pred_masks = torch.argmax(pred_masks, dim=1)
                    pred_onehot = F.one_hot(pred_masks.long(), num_classes=net.n_classes)

                    for c in range(1, net.n_classes):
                        pred_masks_c = pred_onehot[..., c].permute(1, 2, 0).cpu().numpy()
                        true_masks_c = gt_onehot[..., c].permute(1, 2, 0).cpu().numpy()
                        pred_masks_c = _connectivity_region_analysis(pred_masks_c)

                        if pred_masks_c.sum() > 0 and true_masks_c.sum() > 0:
                            case_tot_c = (1 - _eval_dice(pred_masks_c, true_masks_c))
                            asd_tot_c = (metric.binary.asd(pred_masks_c, true_masks_c))
                        elif pred_masks_c.sum() > 0 and true_masks_c.sum() == 0:
                            case_tot_c = 0
                            asd_tot_c = 0
                        else:
                            case_tot_c = 1
                            asd_tot_c = 0
                        tot_multi[c] += case_tot_c
                        asd_multi[c] += asd_tot_c

                        case_performance_dict[f'{organ_dict[c]}_Dice'] = [format(case_tot_c, '.4f')]
                        case_performance_dict[f'{organ_dict[c]}_ASD'] = [format(asd_tot_c, '.2f')]

                    if spade_aux:
                        pred_masks_first_forward = torch.tensor(pred_masks_first_forward)
                        pred_masks_first_forward_softmax = pred_masks_first_forward.clone()
                        pred_masks_first_forward = torch.argmax(pred_masks_first_forward, dim=1)
                        pred_onehot_first_forward = F.one_hot(pred_masks_first_forward.long(),
                                                              num_classes=net.n_classes)

                        for c in range(1, net.n_classes):
                            pred_masks_c_first_forward = pred_onehot_first_forward[..., c].permute(1, 2,
                                                                                                   0).cpu().numpy()
                            true_masks_c = gt_onehot[..., c].permute(1, 2, 0).cpu().numpy()
                            pred_masks_c_first_forward = _connectivity_region_analysis(pred_masks_c_first_forward)

                            if pred_masks_c_first_forward.sum() > 0 and true_masks_c.sum() > 0:
                                case_tot_c_first_forward = (1 - _eval_dice(pred_masks_c_first_forward, true_masks_c))
                                asd_tot_c_first_forward = (metric.binary.asd(pred_masks_c_first_forward, true_masks_c))
                            elif pred_masks_c_first_forward.sum() > 0 and true_masks_c.sum() == 0:
                                case_tot_c_first_forward = 0
                                asd_tot_c_first_forward = 0
                            else:
                                case_tot_c_first_forward = 1
                                asd_tot_c_first_forward = 0
                            #
                            # case_tot_c_first_forward = (1 - _eval_dice(pred_masks_c_first_forward, true_masks_c))
                            # asd_tot_c_first_forward = (metric.binary.asd(pred_masks_c_first_forward, true_masks_c))
                            tot_multi_first_forward[c] += case_tot_c_first_forward
                            asd_multi_first_forward[c] += asd_tot_c_first_forward
                            case_performance_dict[f'{organ_dict[c]}_Dice_first_forward'] = [
                                format(case_tot_c_first_forward, '.4f')]
                            case_performance_dict[f'{organ_dict[c]}_ASD_first_forward'] = [
                                format(asd_tot_c_first_forward, '.2f')]

                    df = pd.DataFrame.from_dict(case_performance_dict)
                    df.to_csv(
                        dir_eval_csv + csv_files_prefix + 'case_performance.csv',
                        mode='a', header=csv_header, index=False)
                    csv_header = False
                    if save_prediction:
                        if not spade_aux:
                            zero_padding = np.zeros([384, 384, 1])
                            prediction_mask_array = np.concatenate(
                                (zero_padding, pred_masks.permute(1, 2, 0).cpu().numpy(), zero_padding), axis=2)
                            # prediction_mask_array = prediction_mask_array.transpose([2, 0, 1])
                            mask_gt_filename = os.path.basename(case.split(',')[1])
                            if nii_save_path is None:
                                nii_save_path = os.path.dirname(case.split(',')[1])

                            mask_gt = sitk.ReadImage(case.split(',')[1])

                            prediction_mask = sitk.GetImageFromArray(prediction_mask_array.transpose([2, 0, 1]))
                            prediction_mask.SetSpacing(mask_gt.GetSpacing())
                            prediction_mask.SetOrigin(mask_gt.GetOrigin())
                            prediction_mask.SetDirection(mask_gt.GetDirection())
                            mask_pred_filename = mask_gt_filename.replace('segmentation', 'prediction')
                            sitk.WriteImage(prediction_mask, os.path.join(nii_save_path, mask_pred_filename))
                        else:
                            zero_padding = np.zeros([384, 384, 1])
                            prediction_mask_array = np.concatenate(
                                (zero_padding, pred_masks.permute(1, 2, 0).cpu().numpy(), zero_padding), axis=2)
                            mask_gt_filename = os.path.basename(case.split(',')[1])
                            if nii_save_path is None:
                                nii_save_path = os.path.dirname(case.split(',')[1])

                            mask_gt = sitk.ReadImage(case.split(',')[1])
                            prediction_mask = sitk.GetImageFromArray(prediction_mask_array.transpose([2, 0, 1]))
                            prediction_mask.SetSpacing(mask_gt.GetSpacing())
                            prediction_mask.SetOrigin(mask_gt.GetOrigin())
                            prediction_mask.SetDirection(mask_gt.GetDirection())
                            mask_pred_filename = mask_gt_filename.replace('segmentation', '_second_forward_prediction')
                            sitk.WriteImage(prediction_mask, os.path.join(nii_save_path, mask_pred_filename))
                            
                            # handle first forward prediction
                            prediction_mask_array = np.concatenate(
                                (zero_padding, pred_masks_first_forward.permute(1, 2, 0).cpu().numpy(), zero_padding),
                                axis=2)
                            prediction_mask = sitk.GetImageFromArray(prediction_mask_array.transpose([2, 0, 1]))
                            prediction_mask.SetSpacing(mask_gt.GetSpacing())
                            prediction_mask.SetOrigin(mask_gt.GetOrigin())
                            prediction_mask.SetDirection(mask_gt.GetDirection())
                            mask_pred_filename = mask_gt_filename.replace('segmentation', '_first_forward_prediction')
                            sitk.WriteImage(prediction_mask, os.path.join(nii_save_path, mask_pred_filename))
                            first_forward_softmax_filename = mask_pred_filename.split('.')[0] + '.npy'
                            np.save(os.path.join(nii_save_path, first_forward_softmax_filename),
                                    pred_masks_first_forward_softmax.cpu().numpy())
                pbar.update(1)
                # torch.cuda.empty_cache()
        if net.n_classes == 2:
            scores[site] = (tot / n_cases)
            asds[site] = (asd / n_cases)
            if spade_aux:
                scores[site + '_first_forward'] = (tot_first_forward / n_cases)
                asds[site + '_first_forward'] = (asd_first_forward / n_cases)
                scores_first.append((tot_first_forward / n_cases))
                asds_first.append((asd_first_forward / n_cases))
                scores_second.append((tot / n_cases))
                asds_second.append((asd / n_cases))
        elif net.n_classes > 2:
            for c in range(1, net.n_classes):
                scores[c][site] = (tot_multi[c] / n_cases)
                asds[c][site] = (asd_multi[c] / n_cases)
            if spade_aux:
                for c in range(1, net.n_classes):
                    scores[c][site + '_first_forward'] = (tot_multi_first_forward[c] / n_cases)
                    asds[c][site + '_first_forward'] = (asd_multi_first_forward[c] / n_cases)
                    scores_first[c].append((tot_multi_first_forward[c] / n_cases))
                    asds_first[c].append((asd_multi_first_forward[c] / n_cases))
                    scores_second[c].append((tot_multi[c] / n_cases))
                    asds_second[c].append((asd_multi[c] / n_cases))
    if net.n_classes == 2:
        if not spade_aux:
            scores['Overall'] = np.mean(list(scores.values()))
            asds['Overall'] = np.mean(list(asds.values()))
        else:
            scores['Overall'] = np.mean(scores_second)
            asds['Overall'] = np.mean(asds_second)
            scores['Overall_first_forward'] = np.mean(scores_first)
            asds['Overall_first_forward'] = np.mean(asds_first)
    elif net.n_classes > 2:
        if not spade_aux:
            for c in range(1, net.n_classes):
                scores[c]['Overall'] = np.mean(list(scores[c].values()))
                asds[c]['Overall'] = np.mean(list(asds[c].values()))
        else:
            for c in range(1, net.n_classes):
                scores[c]['Overall'] = np.mean(scores_second[c])
                asds[c]['Overall'] = np.mean(asds_second[c])
                scores[c]['Overall_first_forward'] = np.mean(scores_first[c])
                asds[c]['Overall_first_forward'] = np.mean(asds_first[c])
    net.train()
    return scores, asds


def _connectivity_region_analysis(mask):
    s = [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    label_im, nb_labels = ndimage.label(mask)  # , structure=s)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im


def _eval_dice(pred, gt):
    dice = (2 * np.sum(pred * gt) + 1e-6) / (np.sum(pred + gt) + 1e-6)
    return 1 - dice
