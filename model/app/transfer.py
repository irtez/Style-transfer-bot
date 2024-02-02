from torchvision.transforms import v2
import torch
from PIL.Image import Image
import gc
from torch.nn.functional import sigmoid
from function import output_min_max, output_mean_std

from config import CONFIG, new_folder



def preprocess(
        original_image: Image,
        style_image: Image
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding

    :param original_image: Original image
    :param style_image: Style image
    :return: tuple of torch.Tensor
    """

    transform_list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    transform = v2.Compose(transform_list)
    original_img_tensor = transform(original_image).unsqueeze(0)
    style_img_tensor = transform(style_image).unsqueeze(0)

    return original_img_tensor, style_img_tensor

def transfer(
        package: dict,
        original_image: Image,
        style_image: Image,
        alpha: float,
        model_name: str
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
    content, style = preprocess(original_image, style_image)

    # Run model
    model = package[f'model_{model_name}']
    with torch.no_grad():
        content = content.to(CONFIG['DEVICE'])
        style = style.to(CONFIG['DEVICE'])
        output = model(content, style, alpha)

    # Move result to CPU and postprocess it
    output = output.cpu().squeeze(0)
    if model_name == "new" and new_folder == "185k_sigmoid":
        output = sigmoid(output)
    
    # Scale Tensor values to [0, 1] (not exact if mean_std is chosen)
    if alpha != 1.0:
        # output = output_min_max(output)
        output = output_mean_std(output)
    
    # Transform [0, 1] float torch.Tensor to [0, 255] int numpy.array
    output = output.mul(255).add_(0.5).permute(1, 2, 0).clamp_(0, 255).to("cpu", torch.uint8).numpy()

    return output
