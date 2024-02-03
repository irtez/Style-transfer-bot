import os
import logging
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types.message import Message
from aiogram.fsm.context import FSMContext
from aiogram.types.input_media_photo import InputMediaPhoto
import config as cfg
from routers.routers_util import *
from keyboards import *


router = Router()


@router.message(Command('cancel'))
async def help(message: Message, state: FSMContext, bot: Bot):
    await clear_state(state)
    text = "All actions are canceled."
    await message.answer(text)

@router.message(Command('help'))
async def help(message: Message, state: FSMContext, bot: Bot):
    user_data = await state.get_data()
    num = 0
    if 'num_generations' in user_data.keys():
        num = user_data['num_generations']
    text = (
        f"Number of images you have generated: {num}."
        "\n\nFor style transferring use /transfer."
        "\nIf you want to cancel any action, use /cancel."
    )
    await message.answer(text)

@router.message(Transfer.choosing_content, (F.photo | F.document))
async def process_content_photo(message: Message, state: FSMContext, bot: Bot):
    await state.set_state(Transfer.checking_document)
    check_res = await check_document(message, bot)
    if check_res:
        await state.set_state(Transfer.choosing_content)
        await message.answer(check_res)
        return
    await state.set_state(Transfer.uploading_content)
    if message.photo:
        photo_id = message.photo[-1].file_id
    elif message.document:
        photo_id = message.document.file_id
    else:
        await message.answer("Unknown error. All actions are cancelled.")
        await clear_state(state)
        return
    await state.update_data(content_photo=photo_id)
    await edit_example_message(
        chat_id=message.chat.id,
        state=state,
        bot=bot,
        mode=['content_message']
    )
    await content_chosen(message.chat.id, state, bot)
@router.message(Transfer.choosing_content)
async def content_invalid_message(message: Message, state: FSMContext, bot: Bot):
    await message.answer('Image was not found in your message.')   

@router.message(Transfer.choosing_style, (F.photo | F.document))
async def process_style_photo(message: Message, state: FSMContext, bot: Bot):
    await state.set_state(Transfer.checking_document)
    check_res = await check_document(message, bot)
    if check_res:
        await state.set_state(Transfer.choosing_style)
        await message.answer(check_res)
        return
    await state.set_state(Transfer.uploading_style)
    if message.photo:
        photo_id = message.photo[-1].file_id
    elif message.document:
        photo_id = message.document.file_id
    else:
        await message.answer("Unknown error. All actions are cancelled.")
        await clear_state(state)
        return
    await state.update_data(style_photo=photo_id)
    await edit_example_message(
        chat_id=message.chat.id,
        state=state,
        bot=bot,
        mode=['style_message']
    )
    await style_chosen(message.chat.id, state, bot)
@router.message(Transfer.choosing_style)
async def style_invalid_message(message: Message, state: FSMContext, bot: Bot):
    await message.answer('Image was not found in your message.')

@router.message((F.photo) & (F.from_user.id == os.environ['OWNER_ID']) & (F.caption == 'u'))
async def photo_upload(message: Message, state: FSMContext, bot: Bot):
    idx = [[photo.file_id, photo.width, photo.height, photo.file_size] for photo in message.photo]
    text = ""
    for file in idx:
        file_id, w, h, size = file
        text += f"{w}x{h}:\nID: {file_id}\nSize: {size / 2**20:.4f} MB\n\n"
    await message.answer(text)

@router.message(Command('start'))
async def start(message: Message, state: FSMContext, bot: Bot):
    logging.info(f'{message.chat.id} start')
    await message.answer(
        "Use /help for help.\nFor generating images use /transfer."
    )

@router.message(Command('transfer'))
async def transfer(message: Message, state: FSMContext, bot: Bot):
    logging.info(f'{message.chat.id} new transfer')
    await state.set_state(Transfer.choosing_content)
    photos = [InputMediaPhoto(media=file_id, caption=name) for (name, file_id) in cfg.examples_content.items()]
    msgs_photo = await bot.send_media_group(message.chat.id, photos)
    msg_text = await message.answer(
        'Send the image with original content, '
        'which has to be stylized.\n'
        'Or choose one of the examples above.',
        reply_markup=example_markup(cfg.examples_content, 'content')
    )
    await state.update_data(content_message=msg_text.message_id)
    await state.update_data(content_photos=[msg_photo.message_id for msg_photo in msgs_photo])


@router.message()
async def unknown(message: Message, state: FSMContext, bot: Bot):
    pass
