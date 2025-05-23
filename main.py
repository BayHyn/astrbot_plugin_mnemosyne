# -*- coding: utf-8 -*-
"""
Mnemosyne - 基于 RAG 的 AstrBot 长期记忆插件主文件
负责插件注册、初始化流程调用、事件和命令的绑定。
"""

import asyncio
from typing import List, Optional, Union, AsyncGenerator
import re # 正则表达式模块，用于处理字符串

# --- AstrBot 核心导入 ---
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.commands import command_group
from astrbot.core.log import LogManager

# --- 插件内部模块导入 ---
from .core import initialization  # 导入初始化逻辑模块
from .core import memory_operations # 导入记忆操作逻辑模块
from .core import summarization_service # 导入总结服务模块
from .core import commands  # 导入命令处理实现模块
from .core.constants import ( # 导入所有常量
    LOG_NAME, DEFAULT_COLLECTION_NAME,
    DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS, DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS,
    DEFAULT_EMBEDDING_SERVICE, DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL, EMBEDDING_ADAPTER_STAR_NAME,
    CONFIG_KEY_EMBEDDING_SERVICE, CONFIG_KEY_EMBEDDING_MODEL, CONFIG_KEY_EMBEDDING_KEY,
    CONFIG_KEY_EMBEDDING_URL, CONFIG_KEY_SUMMARY_CHECK_TASK, CONFIG_KEY_LLM_PROVIDERS,
    CONFIG_KEY_SUMMARY_CHECK_INTERVAL, CONFIG_KEY_SUMMARY_TIME_THRESHOLD,
    CONFIG_KEY_PLATFORM_SETTINGS, CONFIG_KEY_UNIQUE_SESSION
)
from .core.tools import is_group_chat # 导入工具函数

# --- 类型定义和依赖库 ---
from pymilvus import CollectionSchema # Milvus 集合 Schema
from .memory_manager.message_counter import MessageCounter # 消息计数器
from .memory_manager.vector_db.milvus_manager import MilvusManager # Milvus 管理器
from .memory_manager.embedding import OpenAIEmbeddingAPI, GeminiEmbeddingAPI # 嵌入服务 API
from .memory_manager.context_manager import ConversationContextManager # 上下文管理器


@register(
    name="Mnemosyne", # 插件注册名
    author="lxfight", # 插件作者
    description="一个AstrBot插件，实现基于RAG技术的长期记忆功能。", # 插件描述
    version="0.4.0", # 插件版本号
    repo_url="https://github.com/lxfight/astrbot_plugin_mnemosyne", # 插件仓库地址
)
class Mnemosyne(Star):
    """
    Mnemosyne 插件主类。
    负责初始化插件资源、处理事件钩子、注册和执行命令。
    """
    def __init__(self, context: Context, config: dict):
        """
        插件初始化方法。

        Args:
            context (Context): AstrBot 插件上下文对象，提供对机器人核心功能的访问。
            config (dict): 插件的配置字典，从配置文件加载。
        """
        super().__init__(context)
        self.config: dict = config # 插件配置
        self.context: Context = context # AstrBot 上下文
        self.logger = LogManager.GetLogger(log_name=LOG_NAME) # 日志记录器

        # --- 初始化核心组件状态 ---
        # 这些属性将在初始化流程中被赋值
        self.collection_schema: Optional[CollectionSchema] = None # Milvus 集合的 Schema
        self.index_params: dict = {} # Milvus 索引参数
        self.search_params: dict = {} # Milvus 搜索参数
        self.output_fields_for_query: List[str] = [] # 查询时 Milvus 返回的字段列表
        self.collection_name: str = DEFAULT_COLLECTION_NAME # 当前使用的 Milvus 集合名称
        self.milvus_manager: Optional[MilvusManager] = None # Milvus 管理器实例
        self.msg_counter: Optional[MessageCounter] = None # 消息计数器实例
        self.context_manager: Optional[ConversationContextManager] = None # 对话上下文管理器实例
        self.ebd: Optional[Union[OpenAIEmbeddingAPI, GeminiEmbeddingAPI, Star]] = None # 嵌入服务实例 (可以是具体API或适配器Star)
        self.provider = None # LLM 服务提供者实例

        # --- 初始化嵌入服务 (Embedding Service) ---
        # 尝试从 AstrBot 注册的嵌入服务适配器插件获取服务
        try:
            # EMBEDDING_ADAPTER_STAR_NAME 是 "astrbot_plugin_embedding_adapter"
            embedding_adapter_star = self.context.get_registered_star(EMBEDDING_ADAPTER_STAR_NAME)
            if embedding_adapter_star and hasattr(embedding_adapter_star.star_cls, 'get_dim') and hasattr(embedding_adapter_star.star_cls, 'get_model_name'):
                self.ebd = embedding_adapter_star.star_cls # 获取适配器插件的实例
                dim = self.ebd.get_dim() # type: ignore # 获取嵌入维度
                model_name = self.ebd.get_model_name() # type: ignore # 获取模型名称
                if dim is not None and model_name is not None:
                    # 使用从适配器获取的维度和模型名称更新配置
                    self.config["embedding_dim"] = dim
                    # 根据模型名称生成特定的集合名称，避免冲突
                    self.config["collection_name"] = "ea_" + re.sub(r'[^a-zA-Z0-9]', '_', model_name)
                    self.logger.info(f"已通过嵌入服务适配器加载嵌入服务: {model_name} (维度: {dim})")
                else:
                    # 适配器未返回有效信息
                    raise ValueError("嵌入服务适配器未正确注册或未返回有效的维度和模型名称。")
            else:
                # 适配器未找到或不符合接口要求
                self.logger.debug(f"未找到或不兼容的嵌入服务适配器插件: '{EMBEDDING_ADAPTER_STAR_NAME}'。")
                self.ebd = None # 明确设为 None
        except Exception as e:
            self.logger.warning(f"嵌入服务适配器插件 '{EMBEDDING_ADAPTER_STAR_NAME}' 加载失败: {e}", exc_info=True)
            self.ebd = None # 确保加载失败时 ebd 为 None

        # 如果通过适配器未能加载嵌入服务，则尝试根据配置自行初始化
        if self.ebd is None:
            self.logger.info("将根据插件配置自行初始化嵌入服务...")
            # CONFIG_KEY_EMBEDDING_SERVICE 是 "embedding_service"
            # DEFAULT_EMBEDDING_SERVICE 是 "openai"
            embedding_service_name: str = config.get(CONFIG_KEY_EMBEDDING_SERVICE, DEFAULT_EMBEDDING_SERVICE).lower()

            if embedding_service_name == "gemini":
                self.ebd = GeminiEmbeddingAPI(
                    model=config.get(CONFIG_KEY_EMBEDDING_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL),
                    api_key=config.get(CONFIG_KEY_EMBEDDING_KEY),
                )
                self.logger.info(f"已选择 Gemini ({self.ebd.model}) 作为嵌入服务提供商。")
            else: # 默认为 OpenAI 或其他 OpenAI 兼容 API
                self.ebd = OpenAIEmbeddingAPI(
                    model=config.get(CONFIG_KEY_EMBEDDING_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL),
                    api_key=config.get(CONFIG_KEY_EMBEDDING_KEY),
                    base_url=config.get(CONFIG_KEY_EMBEDDING_URL), # 允许自定义 OpenAI API 端点
                )
                self.logger.info(f"已选择 OpenAI ({self.ebd.model}) 作为嵌入服务提供商。")

        # --- 后台总结任务配置 (Periodic Summarization Task) ---
        self._summary_check_task: Optional[asyncio.Task] = None # 后台总结检查任务的句柄

        # CONFIG_KEY_SUMMARY_CHECK_TASK 是 "summary_check_task"
        summary_check_config: dict = config.get(CONFIG_KEY_SUMMARY_CHECK_TASK, {})
        # CONFIG_KEY_SUMMARY_CHECK_INTERVAL 是 "SUMMARY_CHECK_INTERVAL_SECONDS"
        # DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS 是 300
        self.summary_check_interval: int = summary_check_config.get(
            CONFIG_KEY_SUMMARY_CHECK_INTERVAL, DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS
        )
        # CONFIG_KEY_SUMMARY_TIME_THRESHOLD 是 "SUMMARY_TIME_THRESHOLD_SECONDS"
        # DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS 是 1800
        self.summary_time_threshold: int = summary_check_config.get(
            CONFIG_KEY_SUMMARY_TIME_THRESHOLD, DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS
        )

        # 如果配置的总结时间阈值无效 (小于等于0)，则禁用基于时间的自动总结
        if self.summary_time_threshold <= 0:
            self.logger.warning(
                f"配置的总结时间阈值 ({self.summary_time_threshold}秒) 无效，将禁用基于时间的自动总结。"
            )
            self.summary_time_threshold = float("inf") # 设置为无穷大以禁用

        # 配置 Milvus 是否在插入后立即刷新 (flush)
        self.flush_after_insert: bool = bool(config.get("milvus_flush_after_insert", False))
        self.logger.debug(f"Milvus 插入后自动刷新配置: {self.flush_after_insert}")

        self.logger.info("开始执行 Mnemosyne 插件初始化流程...")
        try:
            # 调用初始化模块中的函数，按顺序执行初始化步骤
            initialization.initialize_config_check(self) # 1. 配置检查
            initialization.initialize_config_and_schema(self)  # 2. 初始化插件配置和 Milvus Schema
            initialization.initialize_milvus(self)  # 3. 初始化 Milvus 连接和设置
            initialization.initialize_components(self)  # 4. 初始化其他核心组件 (如上下文管理器)

            # --- 启动后台总结检查任务 ---
            # 仅当上下文管理器已初始化且时间阈值有效时启动
            if self.context_manager and self.summary_time_threshold != float("inf"):
                self._summary_check_task = asyncio.create_task(
                    summarization_service._periodic_summarization_check(self) # 调用总结服务中的检查函数
                )
                self.logger.info(f"后台总结检查任务已启动 (检查间隔: {self.summary_check_interval}s, 总结阈值: {self.summary_time_threshold}s)。")
            elif self.summary_time_threshold == float("inf"):
                self.logger.info("基于时间的自动总结已禁用，不启动后台检查任务。")
            else: # context_manager 未初始化 (理论上不应发生，因为 initialize_components 会处理)
                self.logger.warning(
                    "上下文管理器 (Context manager) 未初始化，无法启动后台总结检查任务。"
                )

            self.logger.info("Mnemosyne 插件核心组件初始化成功。")
        except Exception as e:
            # 捕获初始化过程中的任何严重错误
            self.logger.critical(
                f"Mnemosyne 插件初始化过程中发生严重错误，插件可能无法正常工作: {e}",
                exc_info=True, # 记录完整的异常堆栈信息
            )
            # 此处不应重新抛出异常，否则可能导致 AstrBot 插件加载流程中断
            # 插件应处于部分可用或不可用状态，并通过日志指示问题

    # --- 事件处理钩子 (Event Hooks) ---
    # 调用 memory_operations.py 中的实现
    @filter.on_llm_request()
    async def query_memory(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """
        [事件钩子] 在 LLM 请求发送前被调用。
        负责查询长期记忆并将相关信息注入到 LLM 请求中 (RAG)。

        Args:
            event (AstrMessageEvent): 触发事件的消息对象。
            req (ProviderRequest): 即将发送给 LLM 服务提供者的请求对象。
        """
        # 当会话第一次发生时，插件会从AstrBot中获取上下文历史，之后的会话历史由插件自动管理
        try:
            # 初始化 LLM 服务提供者 (如果尚未初始化)
            # CONFIG_KEY_LLM_PROVIDERS 是 "LLM_providers"
            if not self.provider:
                provider_id: str = self.config.get(CONFIG_KEY_LLM_PROVIDERS, "")
                if not provider_id:
                    self.logger.warning("配置中未指定 LLM_providers，无法自动初始化 Provider。依赖总结等功能可能受限。")
                else:
                    self.provider = self.context.get_provider_by_id(provider_id)
                    if not self.provider:
                        self.logger.error(f"无法通过 ID '{provider_id}' 获取 LLM Provider。")

            # 调用记忆操作模块处理 RAG 查询和注入
            await memory_operations.handle_query_memory(self, event, req)
        except Exception as e:
            self.logger.error(
                f"处理 on_llm_request 钩子时发生未捕获异常: {e}", exc_info=True
            )
        return # 钩子函数通常不返回特定值，而是通过修改 req 对象产生影响

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        """
        [事件钩子] 在接收到 LLM 响应后被调用。
        负责处理 LLM 的响应，例如将其添加到短期记忆或触发总结。

        Args:
            event (AstrMessageEvent): 触发事件的消息对象。
            resp (LLMResponse): 从 LLM 服务提供者返回的响应对象。
        """
        try:
            # 调用记忆操作模块处理 LLM 响应
            await memory_operations.handle_on_llm_resp(self, event, resp)
        except Exception as e:
            self.logger.error(
                f"处理 on_llm_response 钩子时发生未捕获异常: {e}", exc_info=True
            )
        return # 钩子函数通常不返回特定值

    # --- 命令处理 (Command Handling) ---
    # 定义方法并应用装饰器，实际实现调用 commands.py 中的函数

    @command_group("memory") # 定义名为 "memory" 的命令组
    def memory_group(self) -> None:
        """
        长期记忆管理命令组 `/memory`。
        此方法体为空，主要用于通过装饰器定义命令组。
        """
        pass # 这个方法体是空的，主要是为了定义组

    # 应用装饰器，并调用实现函数
    @memory_group.command("list")  # type: ignore # 定义 "list" 子命令
    async def list_collections_cmd(self, event: AstrMessageEvent) -> AsyncGenerator[None, AstrMessageEvent]:
        """
        列出当前 Milvus 实例中的所有集合。
        使用示例：`/memory list`

        Args:
            event (AstrMessageEvent): 命令触发的消息事件对象。

        Yields:
            AstrMessageEvent: 包含命令结果的纯文本消息事件。
        """
        # 调用 commands.py 中的实现，并代理 yield
        async for result in commands.list_collections_cmd_impl(self, event):
            yield result
        return

    @permission_type(PermissionType.ADMIN) # 设置命令权限为管理员
    @memory_group.command("drop_collection")  # type: ignore # 定义 "drop_collection" 子命令
    async def delete_collection_cmd(
        self,
        event: AstrMessageEvent,
        collection_name: str,
        confirm: Optional[str] = None,
    ) -> AsyncGenerator[None, AstrMessageEvent]:
        """
        [管理员] 删除指定的 Milvus 集合及其所有数据。
        使用示例：`/memory drop_collection <collection_name> --confirm`

        Args:
            event (AstrMessageEvent): 命令触发的消息事件对象。
            collection_name (str): 要删除的 Milvus 集合的名称。
            confirm (Optional[str]): 确认参数，必须为 "--confirm" 才执行删除。

        Yields:
            AstrMessageEvent: 包含命令结果的纯文本消息事件。
        """
        async for result in commands.delete_collection_cmd_impl(
            self, event, collection_name, confirm
        ):
            yield result

    @permission_type(PermissionType.ADMIN) # 设置命令权限为管理员
    @memory_group.command("list_records")  # type: ignore # 定义 "list_records" 子命令
    async def list_records_cmd(
        self,
        event: AstrMessageEvent,
        collection_name: Optional[str] = None,
        limit: int = 5 # 默认显示5条
    ) -> AsyncGenerator[None, AstrMessageEvent]:
        """
        查询指定集合的记忆记录 (按创建时间倒序显示)。
        使用示例: `/memory list_records [collection_name] [limit]`

        Args:
            event (AstrMessageEvent): 命令触发的消息事件对象。
            collection_name (Optional[str]): 要查询的集合名称。如果未提供，则使用插件当前配置的集合。
            limit (int): 要显示的记录数量，默认为 5。

        Yields:
            AstrMessageEvent: 包含命令结果的纯文本消息事件。
        """
        async for result in commands.list_records_cmd_impl(
            self, event, collection_name, limit
        ):
            yield result
        return

    @permission_type(PermissionType.ADMIN) # 设置命令权限为管理员
    @memory_group.command("delete_session_memory")  # type: ignore # 定义 "delete_session_memory" 子命令
    async def delete_session_memory_cmd(
        self, event: AstrMessageEvent, session_id: str, confirm: Optional[str] = None
    ) -> AsyncGenerator[None, AstrMessageEvent]:
        """
        [管理员] 删除指定会话 ID 相关的所有记忆信息。
        使用示例：`/memory delete_session_memory <session_id> --confirm`

        Args:
            event (AstrMessageEvent): 命令触发的消息事件对象。
            session_id (str): 要删除记忆的会话 ID。
            confirm (Optional[str]): 确认参数，必须为 "--confirm" 才执行删除。

        Yields:
            AstrMessageEvent: 包含命令结果的纯文本消息事件。
        """
        async for result in commands.delete_session_memory_cmd_impl(
            self, event, session_id, confirm
        ):
            yield result
        return

    @permission_type(PermissionType.MEMBER) # 设置命令权限为成员 (通常是群内任何用户)
    @memory_group.command("reset") # 定义 "reset" 子命令
    async def reset_session_memory_cmd(self, event: AstrMessageEvent, confirm: Optional[str] = None) -> AsyncGenerator[None, AstrMessageEvent]:
        """
        清除当前用户/群聊会话的记忆信息。
        在群聊中，如果未开启会话隔离 (`unique_session` 配置为 false)，此命令将被禁止。
        使用示例：`/memory reset --confirm`

        Args:
            event (AstrMessageEvent): 命令触发的消息事件对象。
            confirm (Optional[str]): 确认参数，必须为 "--confirm" 才执行清除。

        Yields:
            AstrMessageEvent: 包含命令结果或警告信息的纯文本消息事件。
        """
        # CONFIG_KEY_PLATFORM_SETTINGS 是 "platform_settings"
        # CONFIG_KEY_UNIQUE_SESSION 是 "unique_session"
        # 检查平台是否配置了会话隔离，以及当前是否为群聊
        if not self.context._config.get(CONFIG_KEY_PLATFORM_SETTINGS, {}).get(CONFIG_KEY_UNIQUE_SESSION, False) : # type: ignore
            if is_group_chat(event): # is_group_chat 是从 .core.tools 导入的
                yield event.plain_result("⚠️ 当前未开启群聊会话隔离，为了数据安全，禁止在群聊中清除所有人的长期记忆。")
                return
        # 获取当前会话 ID
        session_id = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        if not session_id:
            yield event.plain_result("⚠️ 无法获取当前会话 ID，无法执行重置操作。")
            return
        # 调用实际的删除逻辑
        async for result in commands.delete_session_memory_cmd_impl(
                self, event, session_id, confirm # 使用获取到的 session_id
        ):
            yield result
        return


    @memory_group.command("get_session_id")  # type: ignore # 定义 "get_session_id" 子命令
    async def get_session_id_cmd(self, event: AstrMessageEvent) -> AsyncGenerator[None, AstrMessageEvent]:
        """
        获取当前与您对话的会话 ID。
        使用示例：`/memory get_session_id`

        Args:
            event (AstrMessageEvent): 命令触发的消息事件对象。

        Yields:
            AstrMessageEvent: 包含当前会话 ID 或提示信息的纯文本消息事件。
        """
        async for result in commands.get_session_id_cmd_impl(self, event):
            yield result
        return

    # --- 插件生命周期方法 (Lifecycle Methods) ---
    async def terminate(self) -> None:
        """
        插件停止时的清理逻辑。
        负责取消后台任务、释放 Milvus 连接等资源。
        """
        self.logger.info(f"{LOG_NAME} 插件正在停止...")
        # --- 停止后台总结检查任务 ---
        if self._summary_check_task and not self._summary_check_task.done():
            self.logger.info("正在取消后台总结检查任务...")
            self._summary_check_task.cancel()  # 发送取消请求
            try:
                # 等待任务实际取消完成，设置一个超时避免卡住
                await asyncio.wait_for(self._summary_check_task, timeout=5.0) # 等待最多5秒
            except asyncio.CancelledError:
                # 这是预期的异常，表示任务已成功取消
                self.logger.info("后台总结检查任务已成功取消。")
            except asyncio.TimeoutError:
                # 如果超时，任务可能未能正常取消
                self.logger.warning("等待后台总结检查任务取消超时。任务可能仍在运行或已结束。")
            except Exception as e:
                # 捕获可能在任务取消或等待过程中抛出的其他异常
                self.logger.error(f"等待后台总结检查任务取消时发生错误: {e}", exc_info=True)
        self._summary_check_task = None  # 清理任务引用

        # --- 清理 Milvus 连接 ---
        if self.milvus_manager and self.milvus_manager.is_connected():
            try:
                # 如果不是 Milvus Lite 模式，并且集合已加载，则尝试释放集合
                # （Milvus Lite 的集合通常不需要显式释放，随进程结束）
                if (
                    not self.milvus_manager._is_lite # 检查是否为 Lite 模式
                    and self.milvus_manager.has_collection(self.collection_name)
                ):
                    self.logger.info(
                        f"正在从内存中释放 Milvus 集合 '{self.collection_name}'..."
                    )
                    self.milvus_manager.release_collection(self.collection_name)

                self.logger.info("正在断开与 Milvus 的连接...")
                self.milvus_manager.disconnect()
                self.logger.info("与 Milvus 的连接已成功断开。")

            except Exception as e:
                self.logger.error(f"停止插件时与 Milvus 交互出错: {e}", exc_info=True)
        else:
            self.logger.info("Milvus 管理器未初始化或已断开连接，无需在停止时额外操作。")

        self.logger.info(f"{LOG_NAME} 插件已停止。")
        return
