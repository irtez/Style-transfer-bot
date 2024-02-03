import os
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from middleware.album import AlbumMiddleware
from routers.allmessages import router as allmsg_router
from routers.callbacks import router as callback_router


async def dp_init() -> tuple[Dispatcher, Bot]:
    bot = Bot(token=os.environ['TOKEN'], parse_mode='HTML')
    from redis_storage import storage
    try:
        await storage.redis.ping()
        logging.info('Connected to Redis storage')
    except:
        storage = MemoryStorage()
        logging.warning('Connection to Redis failed. Using MemoryStorage.')
    dp = Dispatcher(storage=storage)
    dp.message.outer_middleware(AlbumMiddleware())
    dp.include_router(allmsg_router)
    dp.include_router(callback_router)
    return dp, bot
