import os
import sys
from typing import Annotated, Optional
import logging

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, Response, File, Query
from fastapi.logger import logger
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import gc

import torch

import net_microAST as net_ma
import net_AesFA as net_ae
from transfer import transfer
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *

# Initialize API Server
app = FastAPI(
    title="Style Transfer Model",
    description="This model can transfer a style from any image to a target (content) image.",
    version="0.9.0",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

def on_startup():
    global app
    logger.info(f"Running envirnoment: {CONFIG['ENV']}")
    logger.info(f"PyTorch using device: {CONFIG['DEVICE']}")

    # Create empty models
    microast_models = [
        net_ma.Encoder(), # content_encoder
        net_ma.Encoder(), # style_encoder
        net_ma.Modulator(), # modulator
        net_ma.Decoder() # decoder
    ]
    names = ['CE', 'SE', 'MOD', 'DEC']

    # Eval each model
    for model in microast_models:
        model.eval()

    # Load state dict for each model from config
    for model, name in zip(microast_models, names):
        model.load_state_dict(
            torch.load(
                CONFIG['MicroAST'][name],
                map_location=CONFIG['DEVICE']
            )
        )

    # Create both nets
    micro_ast = net_ma.TestNet(*microast_models).to(CONFIG['DEVICE'])
    
    # aesfa
    aesfa = net_ae.AesFA_test().to(CONFIG['DEVICE'])

    dict_model = torch.load(CONFIG['AesFA'], map_location=CONFIG['DEVICE'])
    aesfa.netE.load_state_dict(dict_model['netE'])
    aesfa.netS.load_state_dict(dict_model['netS'])
    aesfa.netG.load_state_dict(dict_model['netG'])

    # Add model and other preprocess tools too app state
    app.package = {
        "MicroAST": micro_ast,
        "AesFA": aesfa
    }

@app.post('/api/v1/transfer',
          response_class=FileResponse,
          response_description="Resulting image with content from \
            original image and style from style image.",
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
)
async def make_stylized_image(
    model_name: Annotated[str, Query(pattern=r'^MicroAST$|^AesFA$', description='Model names.')],
    content_file: Annotated[UploadFile, File(description="Image with content.")],
    style_file: Annotated[UploadFile, File(description="Image with style.")],
    content_size: Annotated[
        str,
        Query(
            pattern=r'\d{2,4}x\d{2,4}',
            description="Size of resulting image."
        )
    ] = None,
    style_size: Annotated[
        str,
        Query(
            pattern=r'\d{2,4}x\d{2,4}',
            description="Size of style image. Lower value means more spatial information is transferred \
                if using AesFA model."
        )
    ] = None
):
    """
    Perform style transfer
    """
    logger.info(f'API predict called with {model_name=}, {content_size=}, {style_size=}')

    # Convert images from bytes to PIL.Image
    original_img = Image.open(content_file.file)
    style_img = Image.open(style_file.file)

    # Run model inference
    result_ndarr = transfer(app.package, original_img, style_img, model_name, content_size, style_size)
    
    # Convert numpy array to PIL.Image
    result_img = Image.fromarray(result_ndarr)

    # Convert PIL.Image to bytes
    buff = io.BytesIO()
    result_img.save(buff, format='JPEG', quality=95)
    result_bytes = buff.getvalue()

    logger.info(f'sending result')
    
    # Clear cache and free resources
    if CONFIG['DEVICE'] == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Return resonse
    return Response(content=result_bytes, media_type='image/jpeg')


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

on_startup()

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=5000,
        reload=True,
        log_config="log.ini"
    )
else:
    # Configure logging if main.py executed from Docker
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    gunicorn_logger = logging.getLogger("gunicorn")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers = gunicorn_error_logger.handlers
    logger.handlers = gunicorn_error_logger.handlers
    logger.setLevel(gunicorn_logger.level)
