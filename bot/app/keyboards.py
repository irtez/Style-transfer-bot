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

def generate_markup():
    builder = InlineKeyboardBuilder()
    builder.row(
        *[InlineKeyboardButton(
            text=str(i),
            callback_data=f"{i} content size"
        ) for i in [256, 512, 1024, 'orig', 'custom']]
    )
    builder.row(
        *[InlineKeyboardButton(
            text=str(i),
            callback_data=f"{i} style size"
        ) for i in [256, 512, 1024, 'orig', 'custom']]
    )
    builder.row(InlineKeyboardButton(text='MicroAST', callback_data='MicroAST model'))
    builder.add(InlineKeyboardButton(text='AesFA', callback_data='AesFA model'))
    builder.row(InlineKeyboardButton(text='Generate!', callback_data='generate!'))
    builder.row(InlineKeyboardButton(text='Cancel', callback_data='cancel'))
    return builder.as_markup()
