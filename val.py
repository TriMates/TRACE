import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='result',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('result/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arche'])
    print("=> creating model %s" % config['archd'])

    model_enc = archs.__dict__[config['arche']]()
    model_dec = archs.__dict__[config['archd']](config['num_classes'])
    model_enc = model_enc.cuda()
    model_dec = model_dec.cuda()

    # Data loading code
    img_ids = glob(os.path.join('../data', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    print(img_ids)
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model_enc.load_state_dict(torch.load('result/model_enc.pth'))
    model_dec.load_state_dict(torch.load('result/model_dec.pth'))
    model_enc.eval()
    model_dec.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('../data', config['dataset'], 'images'),
        mask_dir=os.path.join('../data', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()
    import imageio

    count = 0

    if not os.path.exists(os.path.join('images', config['name'])):
        os.makedirs(os.path.join('images', config['name']))

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            # compute output
            t4, t3, t2, t1, out, _ = model_enc(input)
            output = model_dec(t4, t3, t2, t1, out)

            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            target = target.cpu().numpy()
            input = input.cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    print(1)
                    imageio.imwrite(os.path.join('images', config['name'], meta['img_id'][i] + '.jpg'),
                                    (output[i, c] * 255).astype('uint8'))
                    imageio.imwrite(os.path.join('images', config['name'], meta['img_id'][i] + '_target.jpg'),
                                    (target[i, c] * 255).astype('uint8'))
                    imageio.imwrite(os.path.join('images', config['name'], meta['img_id'][i] + '_input.jpg'),
                                    (input[i, c]).astype('uint8'))


    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
