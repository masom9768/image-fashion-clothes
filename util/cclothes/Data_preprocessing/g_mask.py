import os
import cv2
import glob
import numpy as np
from utils import make_folder


label_list = [
                'background',
                'skin',
                'Upper-clothes',
                'eye_g',
                'Skirt',
                'Pants',
                'Dress',
                'Belt',
                'Left-shoe',
                'Right-shoe',
                'Face',
                'Left-leg',
                'Right-leg',
                'hair',
                'hat',
                'Left-arm',
                'Right-arm',
                'Bag',
                'Scarf']

folder_base = '/users/farhad/dfc/Clothes/Clothes-mask-anno'
folder_save = '/users/farhad/dfc/Clothes/Clothes-mask-mask'
img_num = 30000

make_folder(folder_save)

for k in range(img_num):
    folder_num = int(k / 2000)
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            print(label, idx + 1)
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = (idx + 1)

    filename_save = os.path.join(folder_save, str(k) + '.png')
    print(filename_save)
    cv2.imwrite(filename_save, im_base)

