import argparse
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from eval import eval_net
from unet import DualNorm_Unet
from unet.unet_parts import *



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


def test(net,
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
         spade_aux_blocks='',
         nii_save_path=None,
         save_prediction=False,
         excluded_classes=None,
         dataset_name=None
         ):

    net.apply(weights_init)
    global_step = 9000
    pretrained_model_dir = pretrained + f'/Fold_{fold_idx}/{loaded_model_file_name}.pth'
    pretrained_dict_load = torch.load(pretrained_model_dir)
    model_dict = net.state_dict()

    pretrained_dict = {}
    for k, v in pretrained_dict_load.items():
        if (k in model_dict) and (model_dict[k].shape == pretrained_dict_load[k].shape):
            pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    logging.info(f'Model loaded from {pretrained_model_dir}')

    pretrain_suffix = ''
    if pretrained:
        pretrain_suffix = f'_Pretrained'

    block_names = ['inc', 'down1', 'down2', 'down3', 'down4', 'mid', 'up1', 'up2', 'up3', 'up4']

    spade_blocks_suffix = ''

    if spade_aux_blocks != '':
        spade_blocks_suffix += f'_SPADE_{spade_seg_mode}_Aux_'
        for blockname in spade_aux_blocks:
            block_idx = block_names.index(blockname)
            spade_blocks_suffix += str(block_idx)

    dir_results = output_dir
    tensorboard_logdir = dir_results + 'logs/' + f'{save_folder}/' + f'{train_date}_Site_{site}_GPUs_{gpus}/' + \
                         f'BS_{bs}_Epochs_{epochs}_Aug_{aug}_Zoom_{zoom}_Nonlinear_{nonlinear}_Norm_{norm_type}' + \
                         pretrain_suffix + spade_blocks_suffix + f'/Fold_{fold_idx}'
    writer = SummaryWriter(log_dir=tensorboard_logdir)
    if nii_save_path is None:
        nii_save_path = tensorboard_logdir.replace('logs', 'prediction_nii')
        if not os.path.exists(nii_save_path):
            os.makedirs(nii_save_path)
    print(tensorboard_logdir)
    dir_eval_csv = dir_results + 'eval_csv/' + f'{save_folder}/' + \
                   f'{train_date}_Site_{site}_GPUs_{gpus}/' + f'Epochs_{epochs}_Aug_{aug}_Zoom_{zoom}_Nonlinear_{nonlinear}_Norm_{norm_type}' + \
                   pretrain_suffix + spade_blocks_suffix + '/'
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

    # for site_idx in ['A', 'B', 'C']:
    if eval_site is None:
        sites_inferred = list(site)
    else:
        sites_inferred = list(set(site + eval_site))
    sites_inferred.sort()
    print(sites_inferred)
    for site_idx in sites_inferred:
        # for site_idx in ['D', 'E', 'F']:
        test_list[site_idx] = all_list[site_idx][split_list[site_idx][fold_idx][1]].tolist()
        if site_idx in site:
            test_list['Overall'].append(test_list[site_idx])
    print('-----------------------------------------')
    print('Dataset Info:')
    for site_key in sites_inferred:
        if site_key in ['Overall', 'ABC_mixed']:
            case_total_test = 0
            for site_list_train, site_list_test in zip(train_list[site_key], test_list[site_key]):
                case_total_test += len(site_list_test)
            print(f'{site_key}: {len(train_list[site_key])} sites'
                  f'Test: {case_total_test} cases')
        else:
            print(f'Site {site_key} Test:  {len(test_list[site_key])} cases')
    if fold_idx == 0:
        logging.info(f'''Starting Testing:
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
            Fold Index:      {fold_idx}
            Site:            {site}
            Shift+Rotation:  {aug}
            Zoom+Crop:       {zoom}
            Whitening:       {whitening}
            Pretrain:        {pretrained}
            Classes:         {net.n_classes}
        ''')



    csv_header = True
    test_scores, test_asds = eval_net(net, test_list, device, fold_idx, global_step, dir_eval_csv,
                                      csv_files_prefix=csv_files_prefix, whitening=whitening, eval_site=eval_site,
                                      spade_aux=(spade_aux_blocks != ''), save_prediction=save_prediction,
                                      nii_save_path=nii_save_path
                                      )

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
            print('\nTest Average Symmetric Distance: {}'.format(test_asds[st]))

        for st in sites_print:
            test_performance_dict[f'Dice_{st}'] = [format(test_scores[st], '.4f')]
        for st in sites_print:
            test_performance_dict[f'ASD_{st}'] = [format(test_asds[st], '.2f')]
        if spade_aux_blocks != '':
            for st in sites_print:
                test_performance_dict[f'Dice_{st}_first_forward'] = [format(test_scores[st + '_first_forward'], '.4f')]
            for st in sites_print:
                test_performance_dict[f'ASD_{st}_first_forward'] = [format(test_asds[st + '_first_forward'], '.2f')]

        df = pd.DataFrame.from_dict(test_performance_dict)
        df.to_csv(
            dir_eval_csv + csv_files_prefix + f'site_performance.csv',
            mode='a', header=csv_header, index=False)
        csv_header = False
        print('\n' + tensorboard_logdir)


    elif net.n_classes > 2:
        if dataset_name == 'ABD-8':
            abdominal_organ_dict = {1: 'spleen', 2: 'r_kidney', 3: 'l_kidney', 4: 'gallbladder',
                                    5: 'pancreas',
                                    6: 'liver', 7: 'stomach', 8: 'aorta'}
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
        if len(eval_site) > 1:
            sites_print = list(eval_site) + ['Overall']
        else:
            sites_print = list(eval_site)
        for organ_class in range(1, net.n_classes):
            for st in sites_print:
                print(f'\nSite: {st}, Organ: {organ_dict[organ_class]}')
                print('Test Dice Coeff: {}'.format(test_scores[organ_class][st]))
                print('Test Average Symmetric Distance: {}'.format(test_asds[organ_class][st]))
            for st in sites_print:
                test_performance_dict[f'{organ_dict[organ_class]}-Dice-{st}'] = [
                    format(test_scores[organ_class][st], '.4f')]
            for st in sites_print:
                test_performance_dict[f'{organ_dict[organ_class]}-ASD-{st}'] = [
                    format(test_asds[organ_class][st], '.2f')]

        if spade_aux_blocks != '':
            for organ_class in range(1, net.n_classes):
                for st in sites_print:
                    test_performance_dict[f'{organ_dict[organ_class]}_Dice_{st}_first_forward'] = [
                        format(test_scores[organ_class][st + '_first_forward'], '.4f')]
                for st in sites_print:
                    test_performance_dict[f'{organ_dict[organ_class]}_ASD_{st}_first_forward'] = [
                        format(test_asds[organ_class][st + '_first_forward'], '.2f')]

        df = pd.DataFrame.from_dict(test_performance_dict)
        df.to_csv(
            dir_eval_csv + csv_files_prefix + f'site_performance.csv',
            mode='a', header=csv_header, index=False)
        csv_header = False


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
    parser.add_argument('--load-mode', dest='load_mode', type=str, default='default',
                        help='The mode for model loading, default mode is to load same or less')

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
    parser.add_argument('--server', type=str, default='local-prostate',
                        help='change mappings for different servers')
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
    parser.add_argument('--save-lastbest', type=str2bool, default=False, help='only save the last or best checkpoints')
    parser.add_argument('--eval-freq', type=int, default=2,
                        help='checkpoint saving frequency every epoch')
    parser.add_argument('--eval-site', type=str, default=None)
    parser.add_argument('--norm-type', type=str, default='BN',
                        help='choose the type of normalization')
    parser.add_argument('--spade-seg-mode', type=str, default='soft',
                        help='use soft or hard semantic mask')
    parser.add_argument('--spade-aux-blocks', nargs='+', type=str, default='',
                        help='select blocks for using auxilary spatially-adaptive normalization(SPADE)')
    parser.add_argument('--freeze-except', nargs='+', type=str, default=None,
                        help='keywords except for freezing ')
    parser.add_argument('--loaded-model-name', type=str, default='model_last',
                        help='the file name of the model to be loaded')
    parser.add_argument('--save-prediction', type=str2bool, default=False, help='save prediction or not')
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
                            spade_aux_blocks=args.spade_aux_blocks)

    print(net)
    print('Network Architecture:', net.__class__.__name__)
    print('# Network Parameters:', sum(param.numel() for param in net.parameters()))
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.to(device=device)

    if args.fold == -1:
        selected_folds = [0, 1, 2, 3, 4]
    else:
        selected_folds = args.fold

    train_date = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print('Folds INFO:')

    print(f'Evaluate Sites: {args.eval_site}')
    for f_idx in selected_folds:
        try:
            test(net=net,
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
                 spade_seg_mode=args.spade_seg_mode,
                 spade_aux_blocks=args.spade_aux_blocks,
                 loaded_model_file_name=args.loaded_model_name,
                 save_prediction=args.save_prediction,
                 excluded_classes=args.excluded_classes,
                 dataset_name=args.server[args.server.find('-') + 1:]
                 )
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
