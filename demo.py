import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

# uv1 is a list of cell
def get_uv2(uv1, flo):
    uv2 = []
    for uv in uv1:

        x = uv[0]
        y = uv[1]

        pt2 = (int(x+flo[y][x][0]), int(y+flo[y][x][1]))

        uv2.append(pt2)

    return uv2

def viz(img1, img2, flo):
    img_1 = img1[0].permute(1,2,0).cpu().numpy()
    img_2 = img2[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo_image = flow_viz.flow_to_image(flo)

    img_1_u = img_1.astype('uint8').copy()
    img_2_u = img_2.astype('uint8').copy()

    for i in range(15):
        for j in range(20):


            # pt1 = (160, 160)
            # x = pt1[0]
            # y = pt1[1]
            x = j*30
            y = i*30

            pt1 = (x, y)

            pt2 = (int(x+flo[y][x][0]), int(y+flo[y][x][1]))

            point_size = 1
            point_color = (0, 255, 0)  # BGR
            thickness = 4  # 可以为 0 、4、8


            cv2.circle(img_1_u, pt1, 2, point_color, -1)
            cv2.circle(img_2_u, pt2, 2, point_color, -1)

    img_flo = np.concatenate([img_1_u, img_2_u, flo_image], axis=0)


    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.imwrite('res.png', img_flo)



def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = load_image_list(images)
        for i in range(images.shape[0]-1):
            image1 = images[i,None]
            image2 = images[i+1,None]

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, image2, flow_up)

            k = cv2.waitKey(200) * 0xFF
            if k == 27:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
