import os, argparse, math
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import time
from utils.xrayloader import XrayDataset_val
from utils.metrics import *
from collections import OrderedDict
import statistics
import csv
VERSION = 25

def get_cfg(fold):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=1e-4)  
    parser.add_argument('--n_epochs', type=int, default=1)  
    parser.add_argument('--bt_size', type=int, default=1)
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--val_csv_dir', type=str, default='')
    parser.add_argument('--results_dir', default='')

    parse_config = parser.parse_args()
    return parse_config

def compute_metrics(outputs, targets, loss_fn, roc_name, args):
    
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()
    loss = loss_fn(outputs, targets).cpu().item()
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    precision = Precision(outputs, targets)
    ppv = PPV(outputs, targets)
    npv = NPV(outputs, targets)
    # kappa = Cohen_Kappa(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    specificity = spe(outputs, targets)
    auc, _, _ = roc(outputs, targets, args.results_dir, roc_name)
    metrics = OrderedDict([
        ('loss', loss),
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('ppv', ppv),
        ('npv', npv),
        ('auc', auc),
        ('confusion_matrix',cm),
        ('specificity',specificity)
    ])
        
    return metrics

def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_score(y_true, y_pred):
    """
    y_true:(b,c,h,w) label
    y_pred:(b,c,h,w) prediction
    按照每个样本计算
    """
    return ((2 * (y_true * y_pred).sum(-1).sum(-1).sum(-1) + 1e-15) / (y_true.sum(-1).sum(-1).sum(-1) + y_pred.sum(-1).sum(-1).sum(-1) + 1e-15)).mean()

def normalize(img):
    return (img-img.min()) / (img.max()-img.min())

# -------------------------- test func --------------------------#
def test(epoch, model, loader_eval, loss_fn, fold, args):
    print("-------------testing-----------")
    model.eval()
    accuracy_4keys = 0
    total_correct = 0
    total_samples = 0
    predictions = []
    labels = []
    previous_labels = set()  

    with torch.no_grad():
        for img, label, name in tqdm(loader_eval):
            img = img.cuda().float()
            label = label.cuda()
            _, output = model(img)

            if isinstance(output, (tuple, list)):
                output = output[0]
            predictions.append(output)
            labels.append(label)

    evaluation_metrics = compute_metrics(predictions, labels, loss_fn, 'ROC_fold{}'.format(fold), args)
    return evaluation_metrics

if __name__ == '__main__':
    global_sensitivity = []  
    global_specificity = []
    global_accuracy = []
    global_F1_Score = []
    global_auc = []
    global_PPV = []
    global_NPV = []
    global_preds = []
    for fold in range(5):
        print('The fold {}:'.format(fold))
        # -------------------------- get args --------------------------#
        gc.collect()
        torch.cuda.empty_cache()
        parse_config = get_cfg(fold)
        print(parse_config.val_csv_dir)

        # -------------------------- build dataloaders --------------------------#
        transform = transforms.Compose([
        transforms.Resize((parse_config.img_size, parse_config.img_size)),
        transforms.ToTensor()
        ])
        dataset = XrayDataset_val(parse_config, transform)
        loader_eval = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        # -------------------------- build models --------------------------#
        from models.resnet import resnet50
        from models.deeplabv3plus import DeepLabHeadV3Plus, SegmentationModel
        bkbone = resnet50().cuda()
        head = DeepLabHeadV3Plus(in_channels=512, low_level_channels=256, num_classes=1)
        model = SegmentationModel(bkbone, head).cuda()
        pretrained = True
        if pretrained:
            model_dict = model.state_dict()
            model_weights = torch.load(parse_config.weight_path.format(fold))
            pretrained_dict = model_weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict) 
        cls_loss2 = nn.CrossEntropyLoss()

        # -------------------------- build loggers and savers --------------------------#
        os.makedirs(parse_config.results_dir, exist_ok=True)
        writer = SummaryWriter(parse_config.results_dir)
        log_path = parse_config.results_dir
        EPOCHS = parse_config.n_epochs
        logging.basicConfig(filename=os.path.join(log_path,'train_log.log'),
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        # start training
        for epoch in range(1, EPOCHS + 1):
            logging.info('The fold {}'.format(fold))
            start = time.time()
            eval_metrics = test(epoch, model, loader_eval, cls_loss2, fold, args=parse_config)
            time_elapsed = time.time() - start
            print(
                'Training on epoch:{} complete in {:.0f}m {:.0f}s'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
            print('valid/loss', eval_metrics['loss'])
            print('valid/acc',  eval_metrics['acc'])
            print('valid/f1', eval_metrics['f1'])
            print('valid/auc', eval_metrics['auc'])
            print('valid/specificity', eval_metrics['specificity'])
            print('valid/recall', eval_metrics['recall'])
            print('valid/ppv', eval_metrics['ppv'])
            print('valid/npv', eval_metrics['npv'])
            print('confusion_matrix:',eval_metrics['confusion_matrix'])
            logging.info(
                ' \n valid/loss:{} \n valid/acc:{} \n valid/f1:{} \n valid/auc:{} \n valid/recall:{} \n valid/specificity:{} \n valid/ppv:{} \n valid/npv:{} \n confusion_matrix:{} \n accuracy_4keys:{} \n ----------------------------------------------------------------------------------------------------'.
                    format(
                        eval_metrics['loss'], 
                        eval_metrics['acc'], 
                        eval_metrics['f1'], 
                        eval_metrics['auc'],
                        eval_metrics['recall'],
                        eval_metrics['specificity'],
                        eval_metrics['ppv'],
                        eval_metrics['npv'],
                        eval_metrics['confusion_matrix']))
            
        
        global_sensitivity.append(round(eval_metrics['recall'], 4))
        global_specificity.append(round(eval_metrics['specificity'], 4))
        global_accuracy.append(round(eval_metrics['acc'], 4))
        global_F1_Score.append(round(eval_metrics['f1'], 4))
        global_auc.append(round(eval_metrics['auc'], 4))
        global_PPV.append(round(eval_metrics['ppv'], 4))
        global_NPV.append(round(eval_metrics['npv'], 4))

    output_file_path2 = os.path.join(parse_config.results_dir, 'c95_prob{}.csv'.format(VERSION))
    with open(output_file_path2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'label', 'pre', 'prob'])

    logging.info(
    '\n global_sensitivity:{} \n global_specificity:{} \n global_accuracy:{} \n global_F1_Score:{} \n global_auc:{} \n global_NPV:{} \n'.
        format(global_sensitivity, global_specificity, global_accuracy, global_F1_Score, global_auc, global_NPV))
    



