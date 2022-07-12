import io
import os
import logging
import torch
import numpy as np
import zipfile
import cv2
import base64
import pickle
import codecs

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler


logger = logging.getLogger(__name__)


with zipfile.ZipFile("extra.zip", 'r') as zip_ref:
    zip_ref.extractall('./')
    print("NTTAI-LOG Extracted")

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def transform_logits(logits, center, scale, width, height, input_size):

    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:,:,i],
            trans,
            (int(width), int(height)), #(int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits,axis=2)
    return target_logits


class SCHP_ATR(VisionHandler):

    def __init__(self):
        super(SCHP_ATR, self).__init__()
        self.input_size = [512, 512]
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        self.palette = get_palette(18)
        self.schp_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        # 4: 'Upper-clothes', 5: 'Skirt', 6: 'Pants', 7: 'Dress'
        self.label_atr = [4, 5, 6, 7]

    def preprocess(self, data):
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        # image = Image.open(io.BytesIO(image))
        image =  cv2.imdecode(np.frombuffer(io.BytesIO(image).read(), np.uint8), cv2.IMREAD_COLOR)

        h, w, _ = image.shape

        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input_image = cv2.warpAffine(
            image,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input_image = self.schp_transform(input_image)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return (image, input_image, meta)

    def inference(self, data):
        ori_image, input_image, meta = data
        input_image = torch.unsqueeze(input_image, 0)
        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']

        with torch.no_grad():
            output = self.model(input_image.cuda())
            upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
            parsing_result = np.argmax(logits_result, axis=2)

        return (ori_image, parsing_result)

    def postprocess(self, inference_output):
        ori_image, parsing_result = inference_output
        outputs = []

        mask_dict = self._extract(parsing_result)

        for label, mask in mask_dict.items():
            output, trim_mask = self._trim(ori_image * mask, mask)
            output = Image.fromarray(np.asarray(output, dtype=np.uint8))
            b64_string = codecs.encode(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL), "base64").decode('latin1')
            outputs.append(b64_string)

        # reconstituted = pickle.loads(codecs.decode(b64_string.encode('latin1'), "base64"))

        # [[1, 2], ...]
        return [outputs]

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def _extract(self, mask, threshold=0.3):

        mask_dict = dict()

        for label in self.label_atr:
            tmp_mask = 1 * np.expand_dims(mask == label, axis=2)
            ratio = np.sum(tmp_mask) / np.sum(mask != 0)
            if ratio > threshold:
                mask_dict[label] = tmp_mask

        return mask_dict

    def _trim(self, image, mask):
        sum_horizontal = np.sum(mask, axis=1)
        sum_horizontal = list(sum_horizontal)
        sum_vertical = np.sum(mask, axis=0)
        sum_vertical = list(sum_vertical)

        y_min = next((i for i, v in enumerate(sum_horizontal) if v > 0), None)
        y_max = len(sum_horizontal) - next(
            (i for i, v in enumerate(sum_horizontal[::-1]) if v > 0), None)
        x_min = next((i for i, v in enumerate(sum_vertical) if v > 0), None)
        x_max = len(sum_vertical) - next(
            (i for i, v in enumerate(sum_vertical[::-1]) if v > 0), None)

        return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]