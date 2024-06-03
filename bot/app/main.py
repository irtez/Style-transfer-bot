import asyncio
import logging.config
#from dotenv import load_dotenv

async def main():
    dp, bot = await dp_init()
    await bot.delete_webhook(drop_pending_updates=True)
    logging.info('Started')
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.config.fileConfig('log.ini')
    #load_dotenv('../.env')
    from dispatcher import dp_init
    asyncio.run(main())
