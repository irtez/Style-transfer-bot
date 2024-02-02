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

@router.callback_query(F.data.contains('alpha') | F.data.contains('model'))
async def alpha_model(call: CallbackQuery, state: FSMContext, bot: Bot):
    if await state.get_state() == Transfer.choosing_alpha:
        value, mode = call.data.split()
        if mode == 'alpha':
            alpha = float(value)
            await state.update_data(alpha=alpha/100)
        elif mode == 'model':
            await state.update_data(model_name=value)
        await edit_alpha_message(
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
