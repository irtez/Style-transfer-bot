import logging
from aiogram import Bot, F, Router
from aiogram.types.callback_query import CallbackQuery
from aiogram.fsm.context import FSMContext
from routers.routers_util import *


router = Router()


@router.callback_query(F.data == 'cancel')
async def call_cancel(call: CallbackQuery, state: FSMContext, bot: Bot):
    await clear_state(state)
    logging.info(f"{call.message.chat.id} cancel")
    try:
        await bot.send_message(
            chat_id=call.message.chat.id,
            text="All actions are canceled."
        )
        await bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text="Canceled.",
            reply_markup=None
        )
    except:
        pass

@router.callback_query(F.data == 'generate!')
async def call_generate(call: CallbackQuery, state: FSMContext, bot: Bot):
    if await state.get_state() == Transfer.choosing_model_sizes:
        try:
            await bot.edit_message_reply_markup(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                reply_markup=None
            )
        except:
            pass
        await generate_image(
            chat_id=call.message.chat.id,
            state=state,
            bot=bot
        )
    else:
        try:
            await bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text='You are not in the state of choosing models.',
                reply_markup=None
            )
        except:
            pass
        return

@router.callback_query(F.data.contains('size') | F.data.contains('model'))
async def before_generate(call: CallbackQuery, state: FSMContext, bot: Bot):
    if await state.get_state() == Transfer.choosing_model_sizes:
        mode = call.data.split()[-1]
        if mode == 'size':
            size, img_type, _ = call.data.split()
            if size.isdigit():
                await state.update_data(**{f"{img_type}_size": f"{size}x{size}"})
            elif size == 'orig':
                await state.update_data(**{f"{img_type}_size": "orig"})
            elif size == 'custom':
                await call.message.answer(
                    f"""Send new {'image' if img_type == 'content' else 'style'} size.
Write one <i>int</i> (e.g. <i>256</i>) for image to be 256x256 \
or <i>intxint</i> (<i>widthxheight</i>) for exact size \
(e.g. <i>1920x1080</i>)."""
                )
                if img_type == 'content':
                    await state.set_state(Transfer.choosing_content_size)
                elif img_type == 'style':
                    await state.set_state(Transfer.choosing_style_size)
                
        elif mode == 'model':
            await state.update_data(model_name=call.data.split()[0])
        await edit_final_message(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            state=state,
            bot=bot
        )
    else:
        try:
            await bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text='You are not in the state of choosing models.',
                reply_markup=None
            )
        except:
            pass
        return

@router.callback_query(F.data.contains('example'))
async def call_example(call: CallbackQuery, state: FSMContext, bot: Bot):
    example_id, mode = call.data.split()[:2]
    if mode == 'content':
        if await state.get_state() == Transfer.choosing_content:
            await state.update_data(content_photo=f'example {example_id}')
            image_name, _ = list(cfg.examples_content.items())[int(example_id)]
            await edit_example_message(
                chat_id=call.message.chat.id,
                state=state,
                bot=bot,
                mode=['content_photos', 'content_message'],
                example=(image_name, int(example_id))
            )
            await content_chosen(call.message.chat.id, state, bot)
        else:
            try:
                await bot.edit_message_text(
                    chat_id=call.message.chat.id,
                    message_id=call.message.message_id,
                    text='You are not in the state of choosing content.',
                    reply_markup=None
                )
            except:
                pass
            return

    elif mode == 'style':
        if await state.get_state() == Transfer.choosing_style:
            await state.update_data(style_photo=f'example {example_id}')
            image_name, _ = list(cfg.examples_style.items())[int(example_id)]
            await edit_example_message(
                chat_id=call.message.chat.id,
                state=state,
                bot=bot,
                mode=['style_photos', 'style_message'],
                example=(image_name, int(example_id))
            )
            await style_chosen(call.message.chat.id, state, bot)
        else:
            try:
                await bot.edit_message_text(
                    chat_id=call.message.chat.id,
                    message_id=call.message.message_id,
                    text='You are not in the state of choosing style.',
                    reply_markup=None
                )
            except:
                pass
            return
    
    try:
        await bot.edit_message_reply_markup(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=None
        )
    except:
        return
