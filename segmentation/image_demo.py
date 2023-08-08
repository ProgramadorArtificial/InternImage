# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint, save_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file or folder of images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff'],
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.6,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # Save without optimizer parameters (smaller file size)
    # save_checkpoint(model, 'smaller_model.pth', None, checkpoint.get('meta', {}))

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
        palette = checkpoint['meta']['PALETTE']
    else:
        model.CLASSES = get_classes(args.palette)
        palette = get_palette(args.palette)

    if os.path.isdir(args.img):
        files = os.listdir(args.img)
        all_files = []
        for filename in files:
            all_files.append(f'{args.img}/{filename}')
        args.input = all_files
    else:
        all_files = [args.img]

    for path_img in all_files:
        # test a single image
        result = inference_segmentor(model, path_img)

        # show the results
        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(path_img, result,
                                palette=palette,
                                show=False, opacity=args.opacity)
        mmcv.mkdir_or_exist(args.out)
        folder_legend = f'{args.out}/legend'
        mmcv.mkdir_or_exist(folder_legend)
        out_path = osp.join(args.out, osp.basename(path_img))
        out_path_legend = osp.join(folder_legend, osp.basename(path_img))
        cv2.imwrite(out_path, img)

        # Save another image with legends
        unique_class = np.unique(result)
        handles = [Rectangle((1, 1), 1, 1, color=[c / 255 for c in palette[idx_class]]) for idx_class in unique_class]
        labels = [model.CLASSES[idx_class] for idx_class in unique_class]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(50, 40))
        plt.subplot(2, 1, 1)
        plt.imshow(img_rgb)
        plt.subplot(2, 1, 2)
        plt.rcParams.update({'legend.fontsize': 50})
        plt.legend(handles, labels, mode='expand', ncol=3)
        print(args.out)

        plt.axis('off')
        plt.savefig(out_path_legend, bbox_inches='tight', pad_inches=0)

        print(f"Result is save at {out_path}")

if __name__ == '__main__':
    main()