import os
from aioredis import Redis
from aiogram.fsm.storage.redis import RedisStorage


redis = Redis(host="redis", port=6379)
storage = RedisStorage(redis=redis)
