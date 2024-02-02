from aiogram.types import InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


def example_markup(examples: dict, mode: str):
    builder = InlineKeyboardBuilder()
    for i, name in enumerate(examples.keys()):
        builder.row(
            InlineKeyboardButton(
                text=f'{i+1}. {name}',
                callback_data=f"{i} {mode} example"
            )
        )
    builder.row(InlineKeyboardButton(text='Cancel', callback_data='cancel'))
    return builder.as_markup()

def alpha_markup():
    builder = InlineKeyboardBuilder()
    for i in range(5):
        alpha = i * 25
        builder.add(
            InlineKeyboardButton(
                text=str(alpha),
                callback_data=f"{alpha} alpha"
            )
        )
    builder.row(InlineKeyboardButton(text='Old', callback_data='old model'))
    builder.add(InlineKeyboardButton(text='New', callback_data='new model'))
    builder.row(InlineKeyboardButton(text='Generate!', callback_data='generate!'))
    builder.row(InlineKeyboardButton(text='Cancel', callback_data='cancel'))
    return builder.as_markup()
