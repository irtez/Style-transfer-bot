from pydantic import BaseModel, Field, ConfigDict
from fastapi import UploadFile
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse


# class InferenceInput(BaseModel):
#     """
#     Input values for model inference
#     """
#     # model_config = ConfigDict(arbitrary_types_allowed=True)
#     alpha: float = Field(..., ge=0.0, le=1.0, title='The weight that \
#                              controls the degree of stylization. Should be 0<=alpha<=1.')

    



# class InferenceResponse(BaseModel):
#     """
#     Output response for model inference
#     """
#     error: bool = Field(..., example=False, title='Whether there is error')
#     result: FileResponse = Field(..., title='Resulting image with content from original image \
#                                 and style from style image.', )


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')
