import os
import sys
from typing import Annotated
import logging
from copy import deepcopy

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

import net_microAST as net
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


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    logger.info(f"Running envirnoment: {CONFIG['ENV']}")
    logger.info(f"PyTorch using device: {CONFIG['DEVICE']}")

    # Create empty models
    new_models = [
        net.Encoder(), # content_encoder
        net.Encoder(), # style_encoder
        net.Modulator(), # modulator
        net.Decoder() # decoder
    ]
    old_models = deepcopy(new_models)
    names = ['CE', 'SE', 'MOD', 'DEC']

    # Eval each model
    for model in new_models + old_models:
        model.eval()

    # Load state dict for each model from config
    for model_new, model_old, name in zip(new_models, old_models, names):
        model_new.load_state_dict(
            torch.load(
                CONFIG['NEW'][name],
                map_location=CONFIG['DEVICE']
            )
        )
        model_old.load_state_dict(
            torch.load(
                CONFIG['OLD'][name],
                map_location=CONFIG['DEVICE']
            )
        )

    # Create both nets
    model_old = net.TestNet(*old_models).to(CONFIG['DEVICE'])
    model_new = net.TestNet(*new_models).to(CONFIG['DEVICE'])

    # Add model and other preprocess tools too app state
    app.package = {
        "model_old": model_old,
        "model_new": model_new
    }


@app.post('/api/v1/transfer',
          response_class=FileResponse,
          response_description="Resulting image with content from \
            original image and style from style image.",
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
)
async def make_stylized_image(
    alpha: Annotated[float, Query(ge=0.0, le=1.0, description='The weight that \
                              controls the degree of stylization. Should be 0<=alpha<=1.')],
    model_name: Annotated[str, Query(pattern=r'^new$|^old$', description='"old" \
                                    for old model, "new" for new model.')],
    contentFile: Annotated[UploadFile, File(description="Image with content.")],
    styleFile: Annotated[UploadFile, File(description="Image with style.")]
):
    """
    Perform style transfer
    """
    logger.info(f'API predict called with alpha={alpha} and model={model_name}')

    # Convert images from bytes to PIL.Image
    original_img = Image.open(contentFile.file)
    style_img = Image.open(styleFile.file)

    # Run model inference
    result_ndarr = transfer(app.package, original_img, style_img, alpha, model_name)
    
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
