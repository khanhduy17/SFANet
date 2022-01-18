from PIL import Image
import numpy as np
import cv2
import random
from torchvision.transforms import functional


class Transforms(object):
    def __init__(self, scale, crop, stride, gamma, dataset):
        self.scale = scale
        self.crop = crop
        self.stride = stride
        self.gamma = gamma
        self.dataset = dataset

    def __call__(self, image, density_mask, density_nomask, attention):
        # random resize
        height, width = image.size[1], image.size[0]
        if self.dataset == 'SHA':
            if height < width:
                short = height
            else:
                short = width
            if short < 512:
                scale = 512.0 / short
                height = int(round(height * scale))
                width = int(round(width * scale))
                image = image.resize((width, height), Image.BILINEAR)
                density_mask = cv2.resize(density_mask, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
                density_nomask = cv2.resize(density_nomask, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
                attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)
        if self.dataset == 'NFM':
            height = int(round(height * 0.5))
            width = int(round(width * 0.5))
            image = image.resize((width, height), Image.BILINEAR)
            density_mask = cv2.resize(density_mask, (width, height), interpolation=cv2.INTER_LINEAR) / 0.5 / 0.5
            density_nomask = cv2.resize(density_nomask, (width, height), interpolation=cv2.INTER_LINEAR) / 0.5 / 0.5
            attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        scale = random.uniform(self.scale[0], self.scale[1])
        height = int(round(height * scale))
        width = int(round(width * scale))
        image = image.resize((width, height), Image.BILINEAR)
        density_mask = cv2.resize(density_mask, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
        density_nomask = cv2.resize(density_nomask, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
        attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        # random crop
        h, w = self.crop[0], self.crop[1]
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        image = image.crop((dw, dh, dw + w, dh + h))
        density_mask = density_mask[dh:dh + h, dw:dw + w]
        density_nomask = density_nomask[dh:dh + h, dw:dw + w]
        attention = attention[dh:dh + h, dw:dw + w]

        # random flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            density_mask = density_mask[:, ::-1]
            density_nomask = density_nomask[:, ::-1]
            attention = attention[:, ::-1]

        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image = functional.adjust_gamma(image, gamma)

        # random to gray
        if self.dataset == 'SHA':
            if random.random() < 0.1:
                image = functional.to_grayscale(image, num_output_channels=3)

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        density_mask = cv2.resize(density_mask, (density_mask.shape[1] // self.stride, density_mask.shape[0] // self.stride),
                             interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        density_nomask = cv2.resize(density_nomask, (density_nomask.shape[1] // self.stride, density_nomask.shape[0] // self.stride),
                             interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        attention = cv2.resize(attention, (attention.shape[1] // self.stride, attention.shape[0] // self.stride),
                               interpolation=cv2.INTER_LINEAR)

        density_mask = np.reshape(density_mask, [1, density_mask.shape[0], density_mask.shape[1]])
        density_nomask = np.reshape(density_nomask, [1, density_nomask.shape[0], density_nomask.shape[1]])
        attention = np.reshape(attention, [1, attention.shape[0], attention.shape[1]])

        return image, density_mask, density_nomask, attention
