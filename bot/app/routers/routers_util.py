import traceback
import io
import os
import aiohttp
import logging
from aiogram import Bot
from aiogram.types import BufferedInputFile
from aiogram.types.message import Message
from aiogram.types.input_media_photo import InputMediaPhoto
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from PIL import Image, UnidentifiedImageError
import config as cfg
from typing import Optional, Tuple
from aiogram.types.callback_query import CallbackQuery
import re
from keyboards import *

class Transfer(StatesGroup):
    choosing_content = State()
    uploading_content = State()
    choosing_style = State()
    uploading_style = State()
    choosing_model_sizes = State()
    choosing_content_size = State()
    choosing_style_size = State()
    waiting_for_result = State()
    checking_document = State()


async def clear_state(state: FSMContext):
    """
    Clear state and save number of generations for user.
    Default state.clear() will delete this variable,
    while we want to save it.
    """
    user_data = await state.get_data()
    num = 0
    if 'num_generations' in user_data:
        num = user_data['num_generations']
    await state.clear()
    await state.update_data(num_generations=num)

async def send_request(
        content: bytes,
        style: bytes,
        media_type: str = "image/jpeg",
        model_name: str = "microAST",
        content_size: Optional[str] = None,
        style_size: Optional[str] = None
):
    params = {
        'model_name': model_name,
    }
    if content_size:
        params['content_size'] = content_size
    if style_size:
        params['style_size'] = style_size
    headers = {
        'Accept': '*/*'
    }
    url = f"http://{cfg.model_host}/api/v1/transfer"
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('content_file', content, content_type=media_type)
            data.add_field('style_file', style, content_type=media_type)
            image_bytes = None
            response = await session.post(url, params=params, data=data, headers=headers)
            if response.status == 200:
                image_bytes = await response.content.read()
                return response.status, image_bytes
            else:
                return response.status, await response.text()
    except:
        logging.error(traceback.format_exc())
        return 500, None
    
async def check_document(message: Message, bot: Bot):
    await bot.send_chat_action(
        chat_id=message.chat.id,
        action='typing'
    )
    if message.photo:
        file_size = message.photo[-1].file_size
        if file_size > cfg.max_file_size:
            return (
                "Image size should not exceed "
                f"{cfg.max_file_size/(2**20):.2f} MB.\n"
                f"You send an image with the size of {file_size/(2**20):.2f} MB "
                "(after processing on Telegram servers)."
            )
    elif message.document:
        mime_type = message.document.mime_type
        if not mime_type in cfg.mime_types:
            return (
                "Document format is not supproted.\n"
                f"Available formats: {', '.join(cfg.mime_types)}.\n"
                f"Your document format: {mime_type}."
            )

        file_size = message.document.file_size
        if file_size > cfg.max_file_size:
            return (
                "Image size should not exceed "
                f"{cfg.max_file_size/(2**20):.2f} MB.\n"
                f"You send an image with the size of {file_size/(2**20):.2f} MB.\n"
                "Try to send it as photo, not as document."
            )
        
        doc = await bot.download(message.document)
        try:
            Image.open(doc)
        except UnidentifiedImageError:
            return "Document is corrupted or not supported."
    return None

async def check_size(text: str) -> Tuple[str, bool]:
    pattern = r"\d{2,4}x\d{2,4}"
    if text.isdigit():
        try:
            size = int(text)
        except:
            return "Error converting size to int.", False
        if size < 64 or size > 4096:
            return "Size cannot be lower then 64 or higher then 4096.", False
        return f"{size}x{size}", True
    elif re.fullmatch(pattern, text):
        try:
            w, h = list(map(int, text.split('x')))
        except:
            return "Error converting size to int.", False
        if w < 64 or w > 4096:
            return "Width cannot be lower then 64 or higher then 4096.", False
        if h < 64 or h > 4096:
            return "Height cannot be lower then 64 or higher then 4096.", False
        return f"{w}x{h}", True
    return """Wrong input. Type one integer for both width and height to be the same size \
or intxint for width and height respectively.""", False

async def generate_image(chat_id: int, state: FSMContext, bot: Bot, call: CallbackQuery):
    await state.set_state(Transfer.waiting_for_result)
    await bot.send_message(
        chat_id=chat_id,
        text='Wait a moment...'
    )
    await bot.send_chat_action(chat_id=chat_id, action='upload_document', request_timeout=60)
    user_data = await state.get_data()
    content_data = user_data['content_photo']
    style_data = user_data['style_photo']
    if 'example' in content_data:
        _, content_image_id = content_data.split()
        content = cfg.content_bytes[int(content_image_id)]
    else:
        content_io = await bot.download(content_data) # io.BytesIO
        content = content_io.getvalue()
    if 'example' in style_data:
        _, style_image_id = style_data.split()
        style = cfg.style_bytes[int(style_image_id)]
    else:
        style_io = await bot.download(style_data) # io.BytesIO
        style = style_io.getvalue()
    
    content_size = None if user_data['content_size'] == 'orig' else user_data['content_size']
    style_size = None if user_data['style_size'] == 'orig' else user_data['style_size']

    status, img_bytes = await send_request(
        content=content,
        style=style,
        model_name=user_data['model_name'],
        content_size=content_size,
        style_size=style_size
    )
    folder = f'outputs/{chat_id}'
    if status == 200:
        if not 'num_generations' in user_data.keys():
            os.makedirs(folder, exist_ok=True)
            with open(f"{folder}/info_{chat_id}.txt", "w") as f:
                data = call.from_user
                text = f"First name: {data.first_name or '-'}\n"
                text += f"Last name: {data.last_name or '-'}\n"
                text += f"Username: {'@' if data.username else ''}{data.username or '-'}"
                f.write(text)
            num = 1

        else:
            num = user_data['num_generations'] + 1
        
        buff = io.BytesIO(img_bytes)

        result_img = Image.open(buff)
        
        os.makedirs(folder, exist_ok=True)
        filename = f'{chat_id}-{num}.jpg'
        path = f'{folder}/{filename}'

        result_img.save(path, format='JPEG', quality=95)

        image_file = BufferedInputFile(
            file=buff.getvalue(),
            filename=filename
        )

        await bot.send_document(
            chat_id=chat_id,
            document=image_file
        )
        await state.update_data(num_generations=num)
        await clear_state(state)
        await bot.send_message(
            chat_id=chat_id,
            text='For new generation use /transfer.'
        )
        logging.info(f"{chat_id} transfer succes")
    else:
        await bot.send_message(
            chat_id=chat_id,
            text=f"Error code: {status}.\nFor new generation use /transfer."
        )
        await clear_state(state)
        logging.error(f"{chat_id} transfer error\n {img_bytes}")

async def edit_example_message(chat_id: int, state: FSMContext, bot: Bot, mode: list[str], example: set = None):
    user_data = await state.get_data()
    if example:
        try:
            user_data[mode[0]].pop(example[1])
            await bot.delete_messages(chat_id=chat_id, message_ids=user_data[mode[0]])
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=user_data[mode[1]],
                text=f"You've chosen: {example[0]}",
                reply_markup=None
            )
        except:
            pass
    else:
        try:
            await bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=user_data[mode[0]],
                reply_markup=None
            )
        except:
            pass

async def content_chosen(chat_id: int, state: FSMContext, bot: Bot):
    await state.set_state(Transfer.choosing_style)
    photos = [InputMediaPhoto(media=file_id, caption=name) for (name, file_id) in cfg.examples_style.items()]
    msgs_photo = await bot.send_media_group(chat_id, photos)
    msg_text = await bot.send_message(
        text=(
            'Send an image with the desired style that should '
            'be applied to the original.\n'
            'Or choose one of the examples above.'
        ),
        chat_id=chat_id,
        reply_markup=example_markup(cfg.examples_style, 'style')
    )
    await state.update_data(style_message=msg_text.message_id)
    await state.update_data(style_photos=[msg_photo.message_id for msg_photo in msgs_photo])

async def style_chosen(chat_id: int, state: FSMContext, bot: Bot):
    await state.set_state(Transfer.choosing_model_sizes)
    await state.update_data(
        content_size="orig",
        style_size="orig",
        model_name="MicroAST"
    )
    await edit_final_message(
        chat_id=chat_id,
        message_id=None,
        state=state,
        bot=bot
    )

async def edit_final_message(chat_id: int, state: FSMContext, bot: Bot):
    user_data = await state.get_data()
    text = f"""Choose resulting image size, style size and model.
<b>Image size:</b> you can experiment with different \
image sizes, which affects stylization of the image. \
You can choose one of listed below (e.g. 256x256), as well \
as keeping original size or choosing custom size.
<b>Style size:</b> it affects degree of stylization and transfer of \
spatial information. Small size results in better stylized images, \
while big size just transfers color from the style image. \
You can choose this size the same way as choosing image size.
<b>Model</b>: MicroAST or AesFA. AesFA transfers spatial information \
better, while MicroAST is better at preserving content image features.

Current settings:
<b>Image size:</b> {user_data['content_size']}\n\
<b>Style size:</b> {user_data['style_size']}\n\
<b>Model:</b> {user_data['model_name']}"""

    try:
        if 'generate_msg_id' in user_data:
            await bot.edit_message_text(
                text=text,
                chat_id=chat_id,
                message_id=user_data['generate_msg_id'],
                reply_markup=generate_markup()
            )
        else:
            msg = await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=generate_markup()
            )
            await state.update_data(generate_msg_id=msg.message_id)
    except:
        pass
