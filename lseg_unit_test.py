
import os
import math

import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import clip

from utils.clip_mapping_utils import load_pose, load_semantic, load_obj2cls_dict, save_map, cvt_obj_id_2_cls_id, depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point

from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image

#ai2thor image depth get
from ai2thor.controller import Controller
import numpy as np
import matplotlib.pyplot as plt 

'''img_size = 480
c = Controller(scene= 'FloorPlan13', gridSize=0.25, renderDepthImage=True, renderClassImage=True, renderObjectImage=True, renderImage=True, width=img_size, height=img_size, fieldOfView=90)

rgb, depth = c.last_event.frame, c.last_event.depth_frame
# rgb to float 
rgb = rgb.astype(np.float32) / 255.0
c.stop()'''

############
class LSegPredictor:
    def __init__(self):

        lang = "frdige,countertop,sink,cabinet,floor,wall,ceiling,window"
        labels = lang.split(",")

        # loading models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(device)
        clip_version = "ViT-B/32"
        clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                        'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
        print("Loading CLIP model...")
        clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
        self.clip_model = clip_model.to(device).eval()
        lang_token = clip.tokenize(labels)
        lang_token = lang_token.to(device)
        with torch.no_grad():
            text_feats = clip_model.encode_text(lang_token)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.cpu().numpy() # (langs, 512)

        model = LSegEncNet(lang, arch_option=0,
                                block_depth=0,
                                activation='lrelu',
                                crop_size=128)

        model_state_dict = model.state_dict()
        pretrained_state_dict = torch.load("lseg/checkpoints/demo_e200.ckpt", map_location=torch.device(device))
        pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

        model.eval()
        self.model = model.to(device)

        self.norm_mean= [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]
        self.padding = [0.0] * 3
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def get_text_feat(self, labels):
        lang_token = clip.tokenize(labels)
        lang_token = lang_token.to(self.device)
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(lang_token)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.cpu().numpy() # (langs, 512)

        return text_feats

    def predict(self, rgb, labels):
        img_size = rgb.shape[0]

        lseg_out_dict = self.get_lseg_feat(self.model, rgb, labels, self.transform, img_size, img_size, self.norm_mean, self.norm_std)

        pix_logit = lseg_out_dict['logits'].squeeze(0).cpu() # (langs, H, W)
        pix_feat = lseg_out_dict['outputs'].squeeze(0).cpu().numpy() # (512, H, W)

        pix_cls = torch.argmax(pix_logit, dim=0)
        pix_score = pix_logit.max(dim=0).values

        return {
            'pix_cls': pix_cls,
            'pix_score': pix_score,
            'pix_feat': pix_feat,
        }
    
    def get_lseg_feat(self, model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                    base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        vis_image = image.copy()
        image = transform(image).unsqueeze(0).cpu()
        img = image[0].permute(1,2,0)
        img = img * 0.5 + 0.5

        batch, _, h, w = image.size()
        stride_rate = 2.0/3.0
        stride = int(crop_size * stride_rate)

        long_size = base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height


        cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

        if long_size <= crop_size:
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
            with torch.no_grad():
                outputs, logits = model(pad_img, labels)
                # print(outputs.shape, logits.shape) # torch.Size([1, 512, 512, 512]) torch.Size([1, 3, 512, 512])
            outputs = crop_image(outputs, 0, height, 0, width)
            logits = crop_image(logits, 0, height, 0, width)

            return {
                'outputs': outputs,
                'logits': logits,
            }
