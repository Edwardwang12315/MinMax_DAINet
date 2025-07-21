#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import preprocess
import glob

class DARKDetection(data.Dataset):
	def __init__(self, list_file, mode='train'):
		super().__init__()
		self.mode = mode
		self.filepath = []

		img_list = glob.glob(os.path.join(list_file, '*.png'))
		for img_path in img_list:
			self.filepath.append(img_path)

	def __len__(self):
		return len(self.filepath)

	def __getitem__(self, index):
		img = self.pull_item(index)
		return img

	def pull_item(self, index):
		image_path = self.filepath[index]
		img = Image.open(image_path)
		if img.mode == 'L':
			img = img.convert('RGB')

		return torch.from_numpy(np.array(img)),image_path
