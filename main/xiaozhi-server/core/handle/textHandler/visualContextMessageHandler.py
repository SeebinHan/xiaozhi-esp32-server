import time
from typing import Dict, Any

from core.handle.textMessageHandler import TextMessageHandler
from core.handle.textMessageType import TextMessageType

TAG = __name__


class VisualContextMessageHandler(TextMessageHandler):
    """视觉上下文消息处理器，接收ESP32摄像头拍照分析结果"""

    @property
    def message_type(self) -> TextMessageType:
        return TextMessageType.VISUAL_CONTEXT

    async def handle(self, conn, msg_json: Dict[str, Any]) -> None:
        description = msg_json.get("description", "")
        if description and description != "none":
            conn.visual_context = description
            conn.visual_context_time = time.time()
            conn.logger.bind(tag=TAG).info(f"收到视觉上下文: {description}")
        else:
            conn.visual_context = None
