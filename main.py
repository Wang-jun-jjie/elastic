# utility package
import argparse
import logging
import time
import os
from pathlib import Path

# adversarial attack and showcase
parser = argparse.ArgumentParser( description='Adversarial attacks')
parser.add_argument('--cuda',               default='0',            type=str,   help='select gpu on the server. (default: 0)')
parser.add_argument('--description', '--de',default='default',      type=str,   help='description used to define different model')
parser.add_argument('--prefix',             default='',             type=str,   help='prefix to specify checkpoints')
parser.add_argument('--seed',               default=5158,           type=int,   help='random seed')

parser.add_argument('--batch-size', '-b',   default=160,            type=int,    help='mini-batch size (default: 160)')
parser.add_argument('--epochs',             default=80,             type=int,    help='number of total epochs to run')
# parser.add_argument('--lr-min', default=0.005, type=float, help='minimum learning rate for optimizer')
parser.add_argument('--lr-max',             default=0.001,          type=float,  help='learning rate for optimizer')
# parser.add_argument('--momentum', '--mm', default=0.9, type=float, help='momentum for optimizer')
# parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float, help='weight decay for model training')

parser.add_argument('--target', '-t',       default=None,           type=int,    help='adversarial attack target (default: None, non-target)')
parser.add_argument('--iteration', '-i',    default=4,              type=int,    help='adversarial attack iterations (default: 20)')
parser.add_argument('--step-size', '--ss',  default=0.005,          type=float,  help='step size for adversarial attacks')
parser.add_argument('--epsilon',            default=1,              type=float,  help='epsilon for adversarial attacks')
parser.add_argument('--kernel-size', '-k',  default=11,             type=int,    help='kernel size for adversarial attacks, must be odd integer')

parser.add_argument('--image-size', '--is', default=224,            type=int,    help='image size (default: 224 for ImageNet)')
parser.add_argument('--dataset-root', '--ds', default='/tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC', \
    type=str, help='input dataset, default: Imagenet-1k')
parser.add_argument('--ckpt-root', '--ckpt', default='/tmp2/aislab/adv_ckpt', \
    type=str, help='root directory of checkpoints')
parser.add_argument('--opt-level', '-o',    default='O0',           type=str,    help='Nvidia apex optimation level (default: O1)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np

from lib.utils import get_loaders, input_normalize
from lib.loom import Loom
from lib.attack import *

def main():
    print('pytorch version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load dataset (Imagenet)
    train_loader, test_loader = get_loaders(args.dataset_root, args.batch_size, \
                                            image_size=args.image_size,)

    # Load model and optimizer
    model = models.resnet50(pretrained=True).to(device)
    # Add weight decay into the model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay
                                )
    if args.prefix:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model, [optimizer, optimizer2] = amp.initialize(model, optimizer, \
            opt_level=args.opt_level, verbosity=1)
        ckpt_path = Path(args.ckpt_root) / args.description / (args.prefix + '.pth')
        checkpoint = torch.load(ckpt_path)
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        amp.load_state_dict(checkpoint['amp_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model..')
        epoch_start = 0
        prev_acc = 0.0
        model, optimizer = amp.initialize(model, optimizer, \
            opt_level=args.opt_level, verbosity=1)
    
    loom = Loom(args).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    loom.eval()

    # Logger
    # log_dir = Path('./logs/')
    # log_dir.mkdir(parents=True, exist_ok=True)
    # log_path = log_dir / (args.description + '.log')
    # if log_path.exists():
    #     log_path.unlink()
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(
    #     format='[%(asctime)s] - %(message)s',
    #     datefmt='%Y/%m/%d %H:%M:%S',
    #     level=logging.INFO,
    #     filename=log_path
    # )
    # logger.info(args)

    def attack(record=False):
        correct_normal, correct_adv, target_adv, total = 0, 0, 0, 0

        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)

            adv_label = None
            if args.target:
                adv_label = torch.full(label.shape, args.target).to(device)

            # if batch_idx == 0:
            #     adv = baseline_1_1(model, loom, optimizer, data, label, \
            #         args.epsilon, args.step_size, args.iteration, \
            #         adv_label=adv_label, record=True)
            # else:
            #     adv = baseline_1_1(model, loom, optimizer, data, label, \
            #         args.epsilon, args.step_size, args.iteration, \
            #         adv_label=adv_label)

            # if batch_idx == 0:
            #     adv = baseline_2_1(model, loom, optimizer, data, label, \
            #         args.epsilon, args.kernel_size, args.step_size, args.iteration, \
            #         adv_label=adv_label, record=True)
            # else:
            #     adv = baseline_2_1(model, loom, optimizer, data, label, \
            #         args.epsilon, args.kernel_size, args.step_size, args.iteration, \
            #         adv_label=adv_label)
            
            # if batch_idx == 0:
            #     adv = baseline_2_2(model, loom, optimizer, data, label, \
            #         args.epsilon, args.kernel_size, args.step_size, args.iteration, \
            #         adv_label=adv_label, record=True)
            # else:
            #     adv = baseline_2_2(model, loom, optimizer, data, label, \
            #         args.epsilon, args.kernel_size, args.step_size, args.iteration, \
            #         adv_label=adv_label)

            if batch_idx == 0:
                adv = baseline_2_5(model, loom, optimizer, data, label, \
                    args.epsilon, args.kernel_size, args.step_size, args.iteration, \
                    adv_label=adv_label, record=True)
            else:
                adv = baseline_2_5(model, loom, optimizer, data, label, \
                    args.epsilon, args.kernel_size, args.step_size, args.iteration, \
                    adv_label=adv_label)

            # if batch_idx == 0:
            #     adv = baseline_0_1(model, loom, data, \
            #         args.epsilon, 128, 11, \
            #         record=True)
            # else:
            #     adv = baseline_0_1(model, loom, data, \
            #         args.epsilon, 128, 11, \
            #         )

            '''
            Evaluate attack performance
            '''
            with torch.no_grad():
                y_normal = model(input_normalize(data))
                preds_normal = F.softmax(y_normal, dim=1)
                preds_top_p, preds_top_class = preds_normal.topk(1, dim=1)
                correct_normal += (preds_top_class.view(label.shape) == label).sum().item()
                nor_p, nor_c = preds_top_p, preds_top_class

                y_adv = model(input_normalize(adv))
                preds_adv = F.softmax(y_adv, dim=1)
                preds_top_p, preds_top_class = preds_adv.topk(1, dim=1)
                correct_adv += (preds_top_class.view(label.shape) == label).sum().item()
                adv_p, adv_c = preds_top_p, preds_top_class

                # target attack
                if args.target:
                    target_adv += (preds_top_class.view(label.shape) == adv_label).sum().item()
                # get the prediction for the recorded batch
                if batch_idx == 0:
                    # load the class label mapping
                    classPath = '/tmp2/dataset/imagenet/LOC_synset_mapping.txt'
                    classes = list()
                    with open(classPath) as class_file:
                        for line in class_file:
                            class_name = line[10:].strip().split(',')[0]
                            classes.append(class_name)
                    classes = tuple(classes)
                    # result
                    true_label = classes[label[0].cpu().numpy()]
                    nor_class = classes[int(nor_c[0].cpu().numpy())]
                    nor_confidence = nor_p[0].cpu().numpy()
                    adv_class = classes[int(adv_c[0].cpu().numpy())]
                    adv_confidence = adv_p[0].cpu().numpy()
                    print('true label: {}'.format(true_label))
                    print('normal data class: {}, with confidence: {}'.format(nor_class, nor_confidence))
                    print('adversarial data class: {}, with confidence: {}'.format(adv_class, adv_confidence))

                total += label.size(0)
            
            if batch_idx == 50:
                print('==> early break in testing')
                break

        return  (correct_normal / total), (correct_adv / total), (target_adv / total)
    
    # Run
    accuracy_normal, accuracy_adv, target_attack_rate = attack(record=True)
    print('Model original accuracy: {:.6f}'.format(accuracy_normal))
    print('Model adversarial accuracy: {:.6f}'.format(accuracy_adv))
    if args.target:
        print('Target attack success rate: {:.6f}'.format(target_attack_rate))

if __name__ == "__main__":
    main()