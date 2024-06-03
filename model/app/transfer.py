from torchvision.transforms import v2
import torch
from PIL.Image import Image
import gc
from torch.nn.functional import sigmoid
from function import output_min_max, output_mean_std
from typing import Tuple

from config import CONFIG

def parse_size(size: str):
    w, h = list(map(int, size.split('x')))
    return h, w

def preprocess(
        original_image: Image,
        style_image: Image,
        model_name: str,
        content_size: str = None,
        style_size: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding

    :param original_image: Original image
    :param style_image: Style image
    :return: tuple of torch.Tensor
    """
    
    if content_size:
        original_image = v2.Resize(parse_size(content_size))(original_image)
    if style_size:
        style_image = v2.Resize(parse_size(style_size))(style_image)
    if model_name == "AesFA":
        original_image = v2.Pad(CONFIG['AesFA_pad'], padding_mode='reflect')(original_image)
    
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])      
    original_img_tensor = transform(original_image).unsqueeze(0)
    style_img_tensor = transform(style_image).unsqueeze(0)
    return original_img_tensor, style_img_tensor

def transfer(
        package: dict,
        original_image: Image,
        style_image: Image,
        model_name: str,
        content_size: str,
        style_size: str
) -> torch.Tensor:
    """
    Run model and get result

    :param package: dict from fastapi state including model and preocessing objects
    :param package: list of input values
    :return: numpy array of model output
    """
    # Clear cache and free resources
    if CONFIG['DEVICE'] == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Transform PIL.Image to torch.Tensor and perform transforms (if any)
    content, style = preprocess(original_image, style_image, model_name, content_size, style_size)

    # Run model
    model = package[model_name]
    with torch.no_grad():
        content = content.to(CONFIG['DEVICE'])
        style = style.to(CONFIG['DEVICE'])
        output = model(content, style)

    # Move result to CPU and postprocess it
    output = output.cpu().squeeze(0)
    if model_name == "AesFA":
        s = CONFIG['AesFA_pad']
        output = output[:, s:-s, s:-s]
    
    # Transform [0, 1] float torch.Tensor to [0, 255] int numpy.array
    output = output.mul(255).add_(0.5).permute(1, 2, 0).clamp_(0, 255).to("cpu", torch.uint8).numpy()

    return output
