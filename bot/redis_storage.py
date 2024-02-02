import os
from aioredis import Redis
from aiogram.fsm.storage.redis import RedisStorage


redis = Redis(host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'])
storage = RedisStorage(redis=redis)
