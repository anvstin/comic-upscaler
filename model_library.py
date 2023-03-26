import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from torchvision import datasets, models, transforms
import numpy as np

import torch

class ModelSelector:
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.model_path = model_path

        self.model = None
        self.netscale = None
        self.model_path = None
        self.file_url = None

        self.select_model()

    def real_esrgan_x4plus(self):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.netscale = 4
        self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

    def real_esrnet_x4plus(self):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.netscale = 4
        self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']

    def real_esrgan_x4plus_anime_6B(self):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        self.netscale = 4
        self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']

    def real_esrgan_x2plus(self):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.netscale = 2
        self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']

    def real_esrgan_animevideov3(self):
        self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        self.netscale = 4
        self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']

    def real_esrgan_general_x4v3(self):
        self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        self.netscale = 4
        self.file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    def get(self):
        if self.model_path is None:
            self.model_path = self.get_model('RealESRGAN_x4plus', self.file_url)
        return self.model, self.netscale, self.model_path

    def get_model(self):
        if self.model_path is not None and os.path.isfile(self.model_path):
            return

        self.model_path = os.path.join('weights', self.model_name + '.pth')

        if os.path.isfile(self.model_path):
            return

        root_dir = os.path.dirname(os.path.abspath(__file__))
        for url in self.file_url:
            self.model_path = load_file_from_url(url=url, model_dir=os.path.join(root_dir, 'weights'), progress=True, file_name=None)

    def select_model(self):
        # Find a method that matches the model name
        method = getattr(self, self.model_name.lower(), None)
        if not method:
            raise NotImplementedError("Model {} not implemented".format(self.model_name))
        method()
