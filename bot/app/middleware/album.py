from aiogram import BaseMiddleware
from aiogram.types import Message
from typing import Dict, Any, Callable, Awaitable
import time


class AlbumMiddleware(BaseMiddleware):
    """
    Set timeout for handle messages with album attached.
    Without this middleware bot will react to every image
    in the album treating every image as a standalone message,
    which will cause errors and undesired behaviour.
    """
    def __init__(self) -> None:
        self.user_data = {}

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:
        if not event.photo and not event.document:
            return await handler(event, data)
        
        user_id = event.from_user.id
        cur_time = time.time()
        
        if not user_id in self.user_data.keys():
            self.user_data[user_id] = cur_time
        else:
            old_time = self.user_data[user_id]
            self.user_data[user_id] = cur_time
            if cur_time - old_time < 1:
                return
            self.user_data[user_id] = cur_time
        
        return await handler(event, data)
