#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Guanglei Ding
# @contact : dingguanglei.ai@gmail.com
# @Site    : https://github.com/dingguanglei
# @File    : tvm.py

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, iradon_sart
from skimage.restoration import denoise_tv_bregman
from PIL import Image
from tqdm import tqdm
import imageio

file_path = r"00114.png"
angles = np.arange(-80., 80., 2)

with Image.open(file_path) as img:
    img = img.convert("L")
P = np.asarray(img) / 255.
plt.imshow(P, cmap=plt.cm.Greys_r)
plt.show()

sinogram = radon(P, theta=angles, circle=True)

# SART
iter = 30
relaxation = 0.8
weight = 1
max_iter = 5
epsilon = 1e-6

recon_SART = iradon_sart(sinogram, theta=angles, relaxation=relaxation)
last_error = np.mean((recon_SART - P) ** 2)
frames = []
for i in tqdm(range(iter)):
    recon_SART = iradon_sart(sinogram, theta=angles, image=recon_SART, relaxation=relaxation)
    frames.append(recon_SART)
    recon_SART = denoise_tv_bregman(recon_SART, weight=weight, max_iter=max_iter)
    frames.append(recon_SART)
    now_error = np.mean((recon_SART - P) ** 2)
    if abs(last_error - now_error) > epsilon:
        # print("ERROR:%04f" % now_error)
        last_error = now_error
    else:
        print("Finish!")
        break
plt.imshow(recon_SART, cmap=plt.cm.Greys_r)
plt.show()

imageio.mimsave("tvm_processing.gif", frames, 'GIF', duration=0.5)
