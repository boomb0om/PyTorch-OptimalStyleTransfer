import torch
from PIL import Image
import numpy as np
from torchvision import transforms

transformer_to_tensor = transforms.ToTensor()
transformer_to_image = transforms.ToPILImage()

def img2tensor(pil_image):
    tensor = transformer_to_tensor(pil_image)
    return tensor.unsqueeze(0).type(dtype=torch.float64)

def tensor2img(tensor):
    tensor = tensor.squeeze(0).cpu().clamp(0,1)
    return transformer_to_image(tensor)

def img_resize(pil_image, rescale):
    return pil_image.resize((int(pil_image.size[0]*rescale), int(pil_image.size[1]*rescale)))