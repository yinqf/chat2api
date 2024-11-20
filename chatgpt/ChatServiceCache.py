import asyncio
import time
from utils.Logger import logger

class ChatServiceCache:
    def __init__(self, ttl=360, cleanup_interval=60):
        self.cache = {}
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval

    def set(self, key, value):
        """将新项添加到缓存中，并记录添加时间。"""
        self.cache[key] = {'value': value, 'timestamp': time.time()}

    def get(self, key):
        """返回缓存中的值，并从缓存中删除该项。"""
        item = self.cache.pop(key, None)  # 获取并删除缓存项
        return item['value'] if item else None

    def get_cache_size(self):
        """返回缓存中当前剩余项的数量。"""
        return len(self.cache)

    async def start_cleanup_loop(self):
        logger.info("开启缓存清理")
        """定期清理过期缓存的循环任务。"""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            logger.info("开始清理")
            await self._cleanup_expired_items()

    async def _cleanup_expired_items(self):
        """清理所有过期的缓存项。"""
        current_time = time.time()
        expired_keys = [key for key, item in self.cache.items() if (current_time - item['timestamp']) >= self.ttl]
        for key in expired_keys:
            await self._expire_item(key)

    async def _expire_item(self, key):
        """清理过期的缓存项并关闭相关资源。"""
        item = self.cache.pop(key, None)
        if item:
            await item['value'].close_client()
            logger.info(f"{key}  清理成功 剩余: {self.get_cache_size()}")

