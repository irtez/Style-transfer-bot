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
from keyboards import *

class Transfer(StatesGroup):
    choosing_content = State()
    uploading_content = State()
    choosing_style = State()
    uploading_style = State()
    choosing_alpha = State()
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
        alpha: float = 1.0,
        model_name: str = "old"
):
    params = {
        'alpha': alpha,
        'model_name': model_name
    }
    headers = {
        'Accept': '*/*'
    }
    url = f"http://model:80/api/v1/transfer"
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('contentFile', content, content_type=media_type)
            data.add_field('styleFile', style, content_type=media_type)
            image_bytes = None
            response = await session.post(url, params=params, data=data, headers=headers)
            if response.status == 200:
                image_bytes = await response.content.read()
            return response.status, image_bytes
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


async def generate_image(chat_id: int, state: FSMContext, bot: Bot):
    await state.set_state(Transfer.waiting_for_result)
    await bot.send_message(
        chat_id=chat_id,
        text='Wait a moment...'
    )
    await bot.send_chat_action(chat_id=chat_id, action='upload_document', request_timeout=60)
    user_data = await state.get_data()
    alpha = user_data['alpha']
    model_name = user_data['model_name']
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
        
    status, img_bytes = await send_request(
        content=content,
        style=style,
        alpha=alpha,
        model_name=model_name
    )
    if status == 200:
        if not 'num_generations' in user_data.keys():
            num = 1
        else:
            num = user_data['num_generations'] + 1
        
        buff = io.BytesIO(img_bytes)

        result_img = Image.open(buff)
        
        filename = f'{chat_id}-{num}.jpg'
        path = f'results/{filename}'

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
            text=f"Errro code: {status}.\nFor new generation use /transfer."
        )
        await clear_state(state)
        logging.error(f"{chat_id} transfer error")

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
    await state.set_state(Transfer.choosing_alpha)
    alpha = 100
    model_name = "old"
    await state.update_data(alpha=alpha/100, model_name=model_name)
    await edit_alpha_message(
        chat_id=chat_id,
        message_id=None,
        state=state,
        bot=bot
    )

async def edit_alpha_message(chat_id: int, message_id: int, state: FSMContext, bot: Bot):
    user_data = await state.get_data()
    alpha = int(user_data['alpha'] * 100)
    model_name = user_data['model_name']
    text = (
        'Choose the degree of stylization, where 100% is '
        'more content-oriented, 0% is style-oriented; '
        'also you can choose the model: old or new. '
        'New model transfers style slightly better, but '
        'some object boundaries may be a little blurred.\n'
        'NOTE: for now it is strongly recommended to leave '
        'degree of stylization at 100% content, because the images '
        'both models generate when other values are used are '
        'a little strange.'
        f'\n\nCurrent settings:\n<b>Content:</b> {alpha}%\n'
        f'<b>Style:</b> {100-alpha}%\n<b>Model:</b> '
        f'{model_name}'
    )
    try:
        if message_id:
            await bot.edit_message_text(
                text=text,
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=alpha_markup()
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=alpha_markup()
            )
    except:
        pass
