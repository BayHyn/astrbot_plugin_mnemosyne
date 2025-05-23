from typing import List, Dict, Optional, Any # 引入类型注解
import time # 用于获取时间戳
from astrbot.core.log import LogManager # AstrBot 日志管理器
from astrbot.api.event import AstrMessageEvent # AstrBot 消息事件对象
# 导入 MessageCounter 用于类型注解和实际使用
from .message_counter import MessageCounter
# 从常量模块导入日志名称和字典键名
from ..core.constants import (
    CONTEXT_MANAGER_LOG_NAME, CTX_HISTORY, CTX_LAST_SUMMARY_TIME, CTX_EVENT,
    CTX_MESSAGE_ROLE, CTX_MESSAGE_CONTENT, CTX_MESSAGE_TIMESTAMP
)

# 获取日志记录器，使用常量中定义的名称
logger = LogManager.GetLogger(log_name=CONTEXT_MANAGER_LOG_NAME)

class ConversationContextManager:
    """
    会话上下文管理器。
    负责在内存中管理各个会话的对话历史、上次总结时间以及相关的事件对象。
    与 `MessageCounter` 交互以持久化和加载上次总结时间。
    """

    def __init__(self, message_counter: MessageCounter):
        """
        初始化会话上下文管理器。

        Args:
            message_counter (MessageCounter): `MessageCounter` 的实例，用于数据库交互以持久化上次总结时间。
        """
        # self.conversations 结构:
        # {
        #   "session_id_1": {
        #     CTX_HISTORY: [{"role": "user", "content": "...", "timestamp": "..."}, ...],
        #     CTX_LAST_SUMMARY_TIME: 1678886400.0,
        #     CTX_EVENT: AstrMessageEvent_instance
        #   },
        #   ...
        # }
        self.conversations: Dict[str, Dict[str, Any]] = {} # 存储所有会话的上下文信息
        self.message_counter: MessageCounter = message_counter # 消息计数器实例，用于持久化
        logger.debug(f"{CONTEXT_MANAGER_LOG_NAME} 初始化完成，已关联 MessageCounter。")

    def init_conv(self, session_id: str, contexts: List[Dict[str, str]], event: AstrMessageEvent) -> None:
        """
        初始化指定会话 ID 的上下文信息。
        如果会话已存在，则不执行任何操作。
        新的会话将使用提供的 `contexts` (通常来自 AstrBot 的历史消息) 和 `event` 对象进行初始化。
        上次总结时间会尝试从数据库加载，如果不存在，则设置为当前时间并持久化。

        Args:
            session_id (str): 要初始化的会话的唯一标识符。
            contexts (List[Dict[str, str]]): 该会话的初始对话历史记录列表。
                                            每个元素是一个包含 "role" 和 "content" 的字典。
            event (AstrMessageEvent): 与此会话初始化相关的 `AstrMessageEvent` 对象。
        """
        if session_id in self.conversations: # 如果会话已存在于内存中
            logger.debug(f"会话 {session_id} 已存在于内存上下文中，跳过初始化。")
            return

        # 初始化会话字典结构
        self.conversations[session_id] = {
            CTX_HISTORY: contexts, # CTX_HISTORY 是 "history"
            CTX_EVENT: event       # CTX_EVENT 是 "event"
        }
        logger.info(f"为会话 {session_id} 初始化了对话历史和事件对象。")

        # 从数据库加载或设置并持久化上次总结时间
        loaded_summary_time: Optional[float] = self.message_counter.get_last_summary_time(session_id)
        if loaded_summary_time is not None:
            # CTX_LAST_SUMMARY_TIME 是 "last_summary_time"
            self.conversations[session_id][CTX_LAST_SUMMARY_TIME] = loaded_summary_time
            logger.info(f"会话 {session_id} 的上次总结时间戳 {loaded_summary_time} 从数据库加载成功。")
        else: # 数据库中没有记录，则使用当前时间并存入数据库
            current_time: float = time.time()
            self.conversations[session_id][CTX_LAST_SUMMARY_TIME] = current_time
            self.message_counter.update_last_summary_time(session_id, current_time) # 持久化
            logger.info(f"会话 {session_id} 在数据库中无历史总结时间，已初始化为当前时间戳 {current_time} 并已持久化。")
        return

    def add_message(self, session_id: str, role: str, content: str, event: Optional[AstrMessageEvent] = None) -> None:
        """
        向指定会话的上下文中添加一条新的对话消息。
        如果会话首次出现（通过此方法而非 `init_conv`），则会为其创建新的上下文条目，
        并设置和持久化初始的“上次总结时间”。

        Args:
            session_id (str): 消息所属的会话 ID。
            role (str): 消息发送者的角色 (例如, "user", "assistant")。
            content (str): 消息的文本内容。
            event (Optional[AstrMessageEvent], optional): 与此消息相关的 `AstrMessageEvent` 对象。
                在隐式创建新会话时，如果提供了此事件，它将被存储。默认为 None。
        """
        # 如果会话 ID 在内存中不存在，则为其创建新的上下文条目
        if session_id not in self.conversations:
            logger.info(f"会话 {session_id} 在添加消息时未在内存上下文中找到，将创建新的上下文条目。")
            current_time: float = time.time()
            self.conversations[session_id] = {
                CTX_HISTORY: [], # 初始化空历史
                CTX_LAST_SUMMARY_TIME: current_time, # 设置初始总结时间
                CTX_EVENT: event # 存储事件对象 (如果提供)
            }
            if event is None: # 如果未提供 event，记录一个警告，因为某些依赖人格ID的功能可能受影响
                logger.warning(f"为新会话 {session_id} 创建上下文时未提供关联的 event 对象。依赖人格ID的功能可能无法正常工作。")

            # 将新会话的初始上次总结时间持久化到数据库
            self.message_counter.update_last_summary_time(session_id, current_time)
            logger.info(f"新会话 {session_id} 的上次总结时间戳已初始化为 {current_time} 并存入数据库。")

        # 向会话历史中追加新消息
        # CTX_MESSAGE_ROLE, CTX_MESSAGE_CONTENT, CTX_MESSAGE_TIMESTAMP 是常量
        conversation_entry: Dict[str, Any] = self.conversations[session_id]
        conversation_entry[CTX_HISTORY].append(
            {
                CTX_MESSAGE_ROLE: role,
                CTX_MESSAGE_CONTENT: content,
                CTX_MESSAGE_TIMESTAMP: time.strftime("%Y-%m-%d %H:%M:%S"), # 记录消息时间戳 (主要用于调试或未来扩展)
            }
        )
        logger.debug(f"消息 (角色: {role}) 已添加到会话 {session_id} 的历史记录中。")

    def get_summary_time(self, session_id: str) -> Optional[float]: # 返回类型改为 Optional[float]
        """
        获取指定会话的上次总结时间戳。

        Args:
            session_id (str): 会话 ID。

        Returns:
            Optional[float]: 上次总结的 Unix 时间戳 (秒)。如果会话不存在或未记录时间，则返回 `None`。
        """
        if session_id in self.conversations:
            # CTX_LAST_SUMMARY_TIME 是 "last_summary_time"
            return self.conversations[session_id].get(CTX_LAST_SUMMARY_TIME) # 使用 .get() 更安全
        else:
            logger.warning(f"尝试获取不存在的会话 {session_id} 的总结时间。")
            return None # 会话不存在时返回 None

    def update_summary_time(self, session_id: str) -> None:
        """
        更新指定会话的上次总结时间为当前时间，并将其持久化到数据库。

        Args:
            session_id (str): 需要更新总结时间的会话 ID。
        """
        if session_id in self.conversations:
            current_time: float = time.time()
            # CTX_LAST_SUMMARY_TIME 是 "last_summary_time"
            self.conversations[session_id][CTX_LAST_SUMMARY_TIME] = current_time # 更新内存中的时间戳
            self.message_counter.update_last_summary_time(session_id, current_time) # 持久化到数据库
            logger.info(f"会话 {session_id} 的上次总结时间已更新为当前时间戳 {current_time} 并已持久化。")
        else:
            # 尝试更新一个不存在于内存上下文的会话的总结时间，这可能指示逻辑问题或竞争条件
            logger.warning(f"尝试更新不存在于内存上下文的会话 {session_id} 的总结时间。操作被忽略。")

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取指定会话的完整对话历史记录。

        Args:
            session_id (str): 会话 ID。

        Returns:
            List[Dict[str, Any]]: 包含该会话所有消息的列表。如果会话不存在，则返回空列表。
                                   每个消息字典通常包含 "role", "content", 和 "timestamp"。
        """
        if session_id in self.conversations:
            # CTX_HISTORY 是 "history"
            return self.conversations[session_id].get(CTX_HISTORY, []) # 使用 .get() 并提供默认空列表
        else:
            logger.debug(f"尝试获取不存在的会话 {session_id} 的历史记录，返回空列表。")
            return [] # 会话不存在时返回空列表

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定会话 ID 对应的完整上下文信息（包括历史、上次总结时间、事件对象）。

        Args:
            session_id (str): 会话 ID。

        Returns:
            Optional[Dict[str, Any]]: 包含会话所有上下文信息的字典。如果会话不存在，则返回 `None`。
        """
        if session_id in self.conversations:
            return self.conversations[session_id]
        else:
            logger.debug(f"尝试获取不存在的会话 {session_id} 的完整上下文，返回 None。")
            return None # 会话不存在时返回 None