# -*- coding: utf-8 -*-
"""
Mnemosyne - 基于 RAG 的 AstrBot 长期记忆插件主文件
负责插件注册、初始化流程调用、事件和命令的绑定。

支持多种向量数据库后端：Milvus 和 FAISS
支持 AstrBot 原生嵌入服务和传统实现
"""

import asyncio
from datetime import datetime
from typing import List, Optional
import re

# --- AstrBot 核心导入 ---
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.star import Context, Star, register
from astrbot.api.all import *  # 导入 AstrBot API
from astrbot.api.message_components import *  # 导入消息组件
from astrbot.api import logger
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import StarTools

# --- 插件内部模块导入 ---
from .core import initialization  # 导入初始化逻辑模块
from .core import memory_operations  # 导入记忆操作逻辑模块
from .core import commands  # 导入命令处理实现模块
from .core.constants import *  # 导入所有常量
from .core.tools import is_group_chat

# --- 现代化的依赖库 ---
from .memory_manager.message_counter import MessageCounter
from .memory_manager.vector_db import VectorDatabase, VectorDatabaseFactory
from .memory_manager.embedding_adapter import (
    EmbeddingServiceAdapter,
    EmbeddingServiceFactory,
)
from .memory_manager.context_manager import ConversationContextManager

# --- Web界面模块 ---
from .web_interface import MnemosyneWebServer


@register(
    "astrbot_plugin_mnemosyne",
    "lxfight",
    "一个AstrBot插件，实现基于RAG技术的长期记忆功能。支持 Milvus 和 FAISS 向量数据库。",
    "0.6.0",
    "https://github.com/lxfight/astrbot_plugin_mnemosyne",
)
class Mnemosyne(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        self.config = config
        self.context = context
        self.plugin_data_path = StarTools.get_data_dir("astrbot_plugin_mnemosyne")

        # --- 初始化核心组件状态 ---
        self.collection_schema: Optional[dict] = None  # 通用 schema 格式
        self.index_params: dict = {}
        self.search_params: dict = {}
        self.output_fields_for_query: List[str] = []
        self.collection_name: str = DEFAULT_COLLECTION_NAME

        # 现代化的组件
        self.vector_db: Optional[VectorDatabase] = None  # 统一的向量数据库接口
        self.embedding_adapter: Optional[EmbeddingServiceAdapter] = (
            None  # 统一的嵌入服务接口
        )
        self.msg_counter: Optional[MessageCounter] = None
        self.context_manager: Optional[ConversationContextManager] = None
        self.provider = None

        # Web界面组件
        self.web_server: Optional[MnemosyneWebServer] = None

        # 初始化状态标志
        self._embedding_init_attempted = False
        self._core_components_initialized = False

        # --- 一个该死的计时器 ---
        self._summary_check_task: Optional[asyncio.Task] = None

        summary_check_config = config.get("summary_check_task")
        self.summary_check_interval: int = summary_check_config.get(
            "SUMMARY_CHECK_INTERVAL_SECONDS", DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS
        )
        self.summary_time_threshold: int = summary_check_config.get(
            "SUMMARY_TIME_THRESHOLD_SECONDS", DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS
        )
        if self.summary_time_threshold <= 0:
            logger.warning(
                f"配置的 SUMMARY_TIME_THRESHOLD_SECONDS ({self.summary_time_threshold}) 无效，将禁用基于时间的自动总结。"
            )
            self.summary_time_threshold = float("inf")
        # 是否需要刷新
        self.flush_after_insert = False
        logger.info("等待AstrBot初始化完成...")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        # print("AstrBot 初始化完成")
        try:
            # 1. 初始化嵌入服务
            self._initialize_embedding_service()

            # 2. 初始化配置检查
            initialization.initialize_config_check(self)

            # 3. 初始化向量数据库
            self._initialize_vector_database()

            # 4. 初始化其他核心组件
            initialization.initialize_components(self)

            # 5. 初始化Web界面
            self._initialize_web_interface()

            # 6. 启动后台总结检查任务
            self._start_background_tasks()

            self._core_components_initialized = True
            logger.info("Mnemosyne 插件核心组件初始化成功！")

        except Exception as e:
            logger.critical(
                f"Mnemosyne 插件初始化过程中发生严重错误: {e}",
                exc_info=True,
            )

    def _initialize_vector_database(self):
        """初始化向量数据库"""
        try:
            # 确定数据库类型
            db_type = self.config.get("vector_database_type", "milvus").lower()

            # 更新配置中的路径，使用插件专属数据目录
            config_with_paths = self._update_config_paths(self.config.copy())

            # 验证配置
            is_valid, error_msg = VectorDatabaseFactory.validate_config(
                db_type, config_with_paths
            )
            if not is_valid:
                raise ValueError(
                    f"Vector database config validation failed: {error_msg}"
                )

            # 创建数据库实例
            self.vector_db = VectorDatabaseFactory.create_database(
                db_type=db_type, config=config_with_paths
            )

            if not self.vector_db:
                raise RuntimeError(f"Failed to create {db_type} database instance")

            # 连接到数据库
            self.vector_db.connect()
            if not self.vector_db.is_connected():
                raise RuntimeError(f"Failed to connect to {db_type} database")

            # 设置集合名称
            self.collection_name = self.config.get(
                "collection_name", DEFAULT_COLLECTION_NAME
            )

            # 创建集合（如果不存在）
            if not self.vector_db.has_collection(self.collection_name):
                schema = self._create_collection_schema()
                if not self.vector_db.create_collection(self.collection_name, schema):
                    raise RuntimeError(
                        f"Failed to create collection '{self.collection_name}'"
                    )
                logger.info(f"Created new collection '{self.collection_name}'")
            else:
                logger.info(f"Using existing collection '{self.collection_name}'")

            logger.info(f"Successfully initialized {db_type} vector database")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}", exc_info=True)
            raise

    def _update_config_paths(self, config: dict) -> dict:
        """更新配置中的路径，使用插件专属数据目录"""
        import os

        # 更新 FAISS 数据路径
        faiss_config = config.get("faiss_config", {})
        if "faiss_data_path" in faiss_config:
            # 如果是相对路径，则基于插件数据目录
            faiss_path = faiss_config["faiss_data_path"]
            if not os.path.isabs(faiss_path):
                if "faiss_config" not in config:
                    config["faiss_config"] = {}
                config["faiss_config"]["faiss_data_path"] = os.path.join(
                    self.plugin_data_path, faiss_path
                )
        else:
            # 如果没有配置，使用默认路径
            if "faiss_config" not in config:
                config["faiss_config"] = {}
            config["faiss_config"]["faiss_data_path"] = os.path.join(
                self.plugin_data_path, "faiss_data"
            )

        # 更新 Milvus Lite 路径
        if "milvus_lite_path" in config and config["milvus_lite_path"]:
            milvus_path = config["milvus_lite_path"]
            if not os.path.isabs(milvus_path):
                config["milvus_lite_path"] = os.path.join(
                    self.plugin_data_path, milvus_path
                )

        logger.debug(
            f"Updated config paths - FAISS: {config.get('faiss_config', {}).get('faiss_data_path')}, Milvus: {config.get('milvus_lite_path')}"
        )
        return config

    def _create_collection_schema(self):
        """创建集合 schema"""
        embedding_dim = self.config.get("embedding_dim", DEFAULT_EMBEDDING_DIM)
        db_type = self.config.get("vector_database_type", "milvus").lower()

        if db_type == "milvus":
            # 为Milvus创建CollectionSchema对象
            from pymilvus import CollectionSchema, FieldSchema, DataType

            fields = [
                FieldSchema(
                    name=PRIMARY_FIELD_NAME,
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                    description="唯一记忆标识符",
                ),
                FieldSchema(
                    name="personality_id",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="与记忆关联的角色ID",
                ),
                FieldSchema(
                    name="session_id",
                    dtype=DataType.VARCHAR,
                    max_length=72,
                    description="会话ID",
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=4096,
                    description="记忆内容（摘要或片段）",
                ),
                FieldSchema(
                    name=VECTOR_FIELD_NAME,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=embedding_dim,
                    description="记忆的嵌入向量",
                ),
                FieldSchema(
                    name="create_time",
                    dtype=DataType.INT64,
                    description="创建记忆时的时间戳（Unix epoch）",
                ),
            ]
            schema = CollectionSchema(
                fields=fields, description="Mnemosyne memory collection"
            )
        else:
            # 为其他数据库（如FAISS）使用字典格式
            schema = {
                "vector_dim": embedding_dim,
                "fields": [
                    {
                        "name": PRIMARY_FIELD_NAME,
                        "type": "int64",
                        "is_primary": True,
                        "auto_id": True,
                        "description": "唯一记忆标识符",
                    },
                    {
                        "name": "personality_id",
                        "type": "varchar",
                        "max_length": 256,
                        "description": "与记忆关联的角色ID",
                    },
                    {
                        "name": "session_id",
                        "type": "varchar",
                        "max_length": 72,
                        "description": "会话ID",
                    },
                    {
                        "name": "content",
                        "type": "varchar",
                        "max_length": 4096,
                        "description": "记忆内容（摘要或片段）",
                    },
                    {
                        "name": VECTOR_FIELD_NAME,
                        "type": "float_vector",
                        "dim": embedding_dim,
                        "description": "记忆的嵌入向量",
                    },
                    {
                        "name": "create_time",
                        "type": "int64",
                        "description": "创建记忆时的时间戳（Unix epoch）",
                    },
                ],
            }

        self.collection_schema = schema
        return schema

    def _get_database_type_safe(self) -> str:
        """安全地获取数据库类型，兼容不同的数据库管理器"""
        if not self.vector_db:
            return self.config.get("vector_database_type", "unknown")

        try:
            if hasattr(self.vector_db, "get_database_type"):
                # FaissManager 有这个方法
                return self.vector_db.get_database_type().value
            elif hasattr(self.vector_db, "_is_lite"):
                # MilvusManager 有这个属性
                return "milvus_lite" if self.vector_db._is_lite else "milvus"
            else:
                # 从配置中获取数据库类型
                return self.config.get("vector_database_type", "unknown")
        except Exception:
            # 如果出现任何错误，回退到配置值
            return self.config.get("vector_database_type", "unknown")

    # --- 事件处理钩子 (调用 memory_operations.py 中的实现) ---
    @filter.on_llm_request()
    async def query_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        """[事件钩子] 在 LLM 请求前，查询并注入长期记忆。"""
        # 检查核心组件是否已初始化
        if not self._core_components_initialized:
            logger.debug("核心组件未初始化，跳过长期记忆查询")
            return

        # 当会话第一次发生时，插件会从AstrBot中获取上下文历史，之后的会话历史由插件自动管理
        try:
            if not self.provider:
                provider_id = self.config.get("LLM_providers", "")
                self.provider = self.context.get_provider_by_id(provider_id)

            await memory_operations.handle_query_memory(self, event, req)
        except Exception as e:
            logger.error(f"处理 on_llm_request 钩子时发生捕获异常: {e}", exc_info=True)
        return

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """[事件钩子] 在 LLM 响应后"""
        # 检查核心组件是否已初始化
        if not self._core_components_initialized:
            logger.debug("核心组件未初始化，跳过 LLM 响应处理")
            return

        try:
            await memory_operations.handle_on_llm_resp(self, event, resp)
        except Exception as e:
            logger.error(f"处理 on_llm_response 钩子时发生捕获异常: {e}", exc_info=True)
        return

    # --- 命令处理 (定义方法并应用装饰器，调用 commands.py 中的实现) ---

    def _check_initialization(self, event: AstrMessageEvent):
        """检查插件是否已完全初始化"""
        if not self._core_components_initialized:
            return event.plain_result("⚠️ 插件正在初始化中，请稍后再试...")
        return None

    @command_group("memory")
    def memory_group(self):
        """长期记忆管理命令组 /memory"""
        # 这个方法体是空的，主要是为了定义组
        pass

    # 应用装饰器，并调用实现函数
    @memory_group.command("list")  # type: ignore
    async def list_collections_cmd(self, event: AstrMessageEvent):
        """列出当前 Milvus 实例中的所有集合 /memory list
        使用示例：/memory list
        """
        # 检查初始化状态
        init_check = self._check_initialization(event)
        if init_check:
            yield init_check
            return

        # 调用 commands.py 中的实现，并代理 yield
        async for result in commands.list_collections_cmd_impl(self, event):
            yield result
        return

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("drop_collection")  # type: ignore
    async def delete_collection_cmd(
        self,
        event: AstrMessageEvent,
        collection_name: str,
        confirm: Optional[str] = None,
    ):
        """[管理员] 删除指定的 Milvus 集合及其所有数据
        使用示例：/memory drop_collection [collection_name] [confirm]
        """
        async for result in commands.delete_collection_cmd_impl(
            self, event, collection_name, confirm
        ):
            yield result

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("list_records")  # type: ignore
    async def list_records_cmd(
        self,
        event: AstrMessageEvent,
        collection_name: Optional[str] = None,
        limit: int = 5,
    ):
        """查询指定集合的记忆记录 (按创建时间倒序显示)
        使用示例: /memory list_records [collection_name] [limit]
        """
        async for result in commands.list_records_cmd_impl(
            self, event, collection_name, limit
        ):
            yield result
        return

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("delete_session_memory")  # type: ignore
    async def delete_session_memory_cmd(
        self, event: AstrMessageEvent, session_id: str, confirm: Optional[str] = None
    ):
        """[管理员] 删除指定会话 ID 相关的所有记忆信息
        使用示例：/memory delete_session_memory [session_id] [confirm]
        """
        async for result in commands.delete_session_memory_cmd_impl(
            self, event, session_id, confirm
        ):
            yield result
        return

    @permission_type(PermissionType.MEMBER)
    @memory_group.command("reset")
    async def reset_session_memory_cmd(
        self, event: AstrMessageEvent, confirm: Optional[str] = None
    ):
        """清除当前会话 ID 的记忆信息
        使用示例：/memory reset [confirm]
        """
        if not self.context._config.get("platform_settings").get("unique_session"):
            if is_group_chat(event):
                yield event.plain_result("⚠️ 未开启群聊会话隔离，禁止清除群聊长期记忆")
                return
        session_id = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        async for result in commands.delete_session_memory_cmd_impl(
            self, event, session_id, confirm
        ):
            yield result
        return

    @memory_group.command("get_session_id")  # type: ignore
    async def get_session_id_cmd(self, event: AstrMessageEvent):
        """获取当前与您对话的会话 ID
        使用示例：/memory get_session_id
        """
        async for result in commands.get_session_id_cmd_impl(self, event):
            yield result
        return

    # === 迁移相关命令 ===

    @memory_group.command("status")  # type: ignore
    async def migration_status_cmd(self, event: AstrMessageEvent):
        """查看当前插件配置和迁移状态
        使用示例：/memory status
        """
        async for result in commands.migration_status_cmd_impl(self, event):
            yield result
        return

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("migrate_config")  # type: ignore
    async def migrate_config_cmd(self, event: AstrMessageEvent):
        """[管理员] 迁移配置到新格式
        使用示例：/memory migrate_config
        """
        async for result in commands.migrate_config_cmd_impl(self, event):
            yield result
        return

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("migrate_to_faiss")  # type: ignore
    async def migrate_to_faiss_cmd(
        self, event: AstrMessageEvent, confirm: Optional[str] = None
    ):
        """[管理员] 迁移数据到 FAISS 数据库
        使用示例：/memory migrate_to_faiss [--confirm]
        """
        async for result in commands.migrate_to_faiss_cmd_impl(self, event, confirm):
            yield result
        return

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("migrate_to_milvus")  # type: ignore
    async def migrate_to_milvus_cmd(
        self, event: AstrMessageEvent, confirm: Optional[str] = None
    ):
        """[管理员] 迁移数据到 Milvus 数据库
        使用示例：/memory migrate_to_milvus [--confirm]
        """
        async for result in commands.migrate_to_milvus_cmd_impl(self, event, confirm):
            yield result
        return

    @memory_group.command("validate_config")  # type: ignore
    async def validate_config_cmd(self, event: AstrMessageEvent):
        """验证当前配置
        使用示例：/memory validate_config
        """
        async for result in commands.validate_config_cmd_impl(self, event):
            yield result
        return

    @memory_group.command("help")  # type: ignore
    async def help_cmd(self, event: AstrMessageEvent):
        """显示详细帮助信息
        使用示例：/memory help
        """
        async for result in commands.help_cmd_impl(self, event):
            yield result
        return

    # === Web界面管理命令 ===

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("web_start")  # type: ignore
    async def web_start_cmd(self, event: AstrMessageEvent):
        """[管理员] 启动Web可视化界面
        使用示例：/memory web_start
        """
        try:
            # 如果Web服务器不存在或已停止，创建新实例
            if not self.web_server or not self.web_server.is_running:
                self.web_server = MnemosyneWebServer(self)

            if self.web_server.is_running:
                yield event.plain_result("✅ Web界面已在运行中")
                yield event.plain_result(
                    f"🌐 访问地址: {self.web_server.get_status()['url']}"
                )

                # 显示访问令牌（如果启用认证）
                if self.web_server.auth_enabled and self.web_server.access_token:
                    yield event.plain_result(
                        f"🔑 访问令牌: {self.web_server.access_token}"
                    )
                return

            success = self.web_server.start()
            if success:
                status = self.web_server.get_status()
                yield event.plain_result("✅ Web界面启动成功！")
                yield event.plain_result(f"🌐 访问地址: {status['url']}")

                # 如果启用了认证，显示访问令牌
                if self.web_server.auth_enabled and self.web_server.access_token:
                    yield event.plain_result(
                        f"🔑 访问令牌: {self.web_server.access_token}"
                    )
                    yield event.plain_result(
                        "💡 提示：首次访问需要输入上述访问令牌进行认证"
                    )
                else:
                    yield event.plain_result(
                        "💡 提示：在浏览器中打开上述地址即可访问记忆管理界面"
                    )
            else:
                yield event.plain_result("❌ Web界面启动失败，请查看日志获取详细信息")

        except Exception as e:
            logger.error(f"启动Web界面失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 启动Web界面时发生错误: {str(e)}")

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("web_stop")  # type: ignore
    async def web_stop_cmd(self, event: AstrMessageEvent):
        """[管理员] 停止Web可视化界面
        使用示例：/memory web_stop
        """
        try:
            if not self.web_server or not self.web_server.is_running:
                yield event.plain_result("ℹ️ Web界面未在运行")
                return

            success = self.web_server.stop()
            if success:
                yield event.plain_result("✅ Web界面已停止")
            else:
                yield event.plain_result("❌ 停止Web界面失败，请查看日志获取详细信息")

        except Exception as e:
            logger.error(f"停止Web界面失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 停止Web界面时发生错误: {str(e)}")

    @memory_group.command("web_status")  # type: ignore
    async def web_status_cmd(self, event: AstrMessageEvent):
        """查看Web界面状态
        使用示例：/memory web_status
        """
        try:
            if not self.web_server:
                yield event.plain_result("ℹ️ Web界面未初始化")
                return

            status = self.web_server.get_status()

            status_text = "🌐 Web界面状态报告\n\n"
            status_text += (
                f"启用状态: {'✅ 已启用' if status['enabled'] else '❌ 已禁用'}\n"
            )
            status_text += (
                f"运行状态: {'🟢 运行中' if status['running'] else '🔴 已停止'}\n"
            )
            status_text += f"监听地址: {status['host']}:{status['port']}\n"
            status_text += f"自动停止: {'✅ 已启用' if status['auto_stop_enabled'] else '❌ 已禁用'}\n"

            if status["running"] and status["auto_stop_enabled"]:
                status_text += f"空闲超时: {status['idle_timeout_minutes']} 分钟\n"
                status_text += f"最后访问: {status.get('last_access_time', '未知')}\n"
                status_text += f"空闲时间: {status.get('idle_minutes', 0)} 分钟\n"
                status_text += f"剩余时间: {status.get('remaining_minutes', 0)} 分钟\n"

            if status["url"]:
                status_text += f"访问地址: {status['url']}\n"

                # 如果启用了认证，显示访问令牌信息
                if self.web_server.auth_enabled:
                    if self.web_server.access_token:
                        status_text += f"访问令牌: {self.web_server.access_token}\n"
                        status_text += "\n💡 提示：首次访问需要输入访问令牌进行认证"
                    else:
                        status_text += "\n⚠️ 认证已启用但访问令牌未生成"
                else:
                    status_text += (
                        "\n💡 提示：在浏览器中打开访问地址即可使用记忆管理界面"
                    )
            else:
                status_text += "\n💡 提示：使用 /memory web_start 启动Web界面"

            yield event.plain_result(status_text)

        except Exception as e:
            logger.error(f"获取Web界面状态失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 获取Web界面状态时发生错误: {str(e)}")

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("web_keepalive")  # type: ignore
    async def web_keepalive_cmd(self, event: AstrMessageEvent):
        """[管理员] 重置Web界面空闲时间
        使用示例：/memory web_keepalive
        """
        try:
            if not self.web_server or not self.web_server.is_running:
                yield event.plain_result("ℹ️ Web界面未在运行")
                return

            # 重置最后访问时间
            self.web_server.last_access_time = datetime.now()

            status = self.web_server.get_status()
            yield event.plain_result("✅ Web界面空闲时间已重置")

            if status["auto_stop_enabled"]:
                yield event.plain_result(
                    f"🕒 将在 {status['idle_timeout_minutes']} 分钟后自动停止（如无访问）"
                )

        except Exception as e:
            logger.error(f"重置Web界面空闲时间失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 重置Web界面空闲时间时发生错误: {str(e)}")

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("web_cleanup")  # type: ignore
    async def web_cleanup_cmd(self, event: AstrMessageEvent):
        """[管理员] 清理Web界面资源并重置
        使用示例：/memory web_cleanup
        """
        try:
            if self.web_server and self.web_server.is_running:
                yield event.plain_result("🛑 正在停止当前Web界面...")
                self.web_server.stop()

            # 重置Web服务器实例
            self.web_server = None

            yield event.plain_result("✅ Web界面资源已清理")
            yield event.plain_result("💡 提示：使用 /memory web_start 重新启动Web界面")

        except Exception as e:
            logger.error(f"清理Web界面失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 清理Web界面时发生错误: {str(e)}")

    # --- 插件生命周期方法 ---

    def _initialize_embedding_service(self):
        """初始化嵌入服务"""
        try:
            self.embedding_adapter = EmbeddingServiceFactory.create_adapter(
                context=self.context, config=self.config
            )

            if self.embedding_adapter:
                # 更新配置中的维度信息
                dim = self.embedding_adapter.get_dim()
                model_name = self.embedding_adapter.get_model_name()

                if dim is not None:
                    self.config["embedding_dim"] = dim

                if model_name and model_name != "unknown":
                    # 仅在用户未配置 collection_name 时，才根据模型名称生成默认名称
                    if not self.config.get("collection_name"):
                        safe_model_name = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
                        self.config["collection_name"] = f"mnemosyne_{safe_model_name}"
                        logger.info(f"未配置集合名称，根据模型自动生成: {self.config['collection_name']}")

                logger.info(
                    f"成功初始化嵌入服务: {self.embedding_adapter.service_name}"
                )
                self._embedding_init_attempted = True
            else:
                logger.error("嵌入服务初始化失败")

        except Exception as e:
            logger.error(f"初始化嵌入服务失败: {e}", exc_info=True)

    def _initialize_web_interface(self):
        """初始化Web界面"""
        try:
            # 每次初始化都创建新的Web服务器实例
            self.web_server = MnemosyneWebServer(self)

            # 如果配置中启用了Web界面，自动启动
            if self.web_server.enabled:
                success = self.web_server.start()
                if success:
                    status = self.web_server.get_status()
                    logger.info(f"Web界面已自动启动: {status['url']}")
                    if self.web_server.auth_enabled and self.web_server.access_token:
                        logger.info(f"访问令牌: {self.web_server.access_token}")
                else:
                    logger.warning("Web界面自动启动失败")
            else:
                logger.info("Web界面已禁用，使用 /memory web_start 手动启动")

        except Exception as e:
            logger.error(f"初始化Web界面失败: {e}", exc_info=True)
            # Web界面初始化失败不应该影响插件的正常运行
            self.web_server = None

    def _start_background_tasks(self):
        """启动后台任务"""
        # --- 启动后台总结检查任务 ---
        if self.context_manager and self.summary_time_threshold != float("inf"):
            # 确保 context_manager 已初始化且阈值有效
            self._summary_check_task = asyncio.create_task(
                memory_operations._periodic_summarization_check(self)
            )
            logger.info("后台总结检查任务已启动。")
        elif self.summary_time_threshold == float("inf"):
            logger.info("基于时间的自动总结已禁用，不启动后台检查任务。")
        else:
            logger.warning("Context manager 未初始化，无法启动后台总结检查任务。")

    async def terminate(self):
        """插件停止时的清理逻辑"""
        logger.info("Mnemosyne 插件正在停止...")

        # --- 停止后台总结检查任务 ---
        if self._summary_check_task and not self._summary_check_task.done():
            logger.info("正在取消后台总结检查任务...")
            self._summary_check_task.cancel()
            try:
                # 等待任务实际取消完成，设置一个超时避免卡住
                await asyncio.wait_for(self._summary_check_task, timeout=5.0)
            except asyncio.CancelledError:
                logger.info("后台总结检查任务已成功取消。")
            except asyncio.TimeoutError:
                logger.warning("等待后台总结检查任务取消超时。")
            except Exception as e:
                # 捕获可能在任务取消过程中抛出的其他异常
                logger.error(f"等待后台任务取消时发生错误: {e}", exc_info=True)
        self._summary_check_task = None  # 清理任务引用

        # --- 停止Web服务器 ---
        if self.web_server and self.web_server.is_running:
            try:
                logger.info("正在停止Web服务器...")
                if self.web_server.stop():
                    logger.info("Web服务器已成功停止。")
                else:
                    logger.warning("Web服务器停止时返回失败状态。")
            except Exception as e:
                logger.error(f"停止Web服务器时出错: {e}", exc_info=True)
        else:
            logger.info("Web服务器未运行，无需停止。")

        # --- 断开向量数据库连接 ---
        if self.vector_db and self.vector_db.is_connected():
            try:
                # 获取数据库类型，兼容不同的数据库管理器
                db_type = "unknown"
                if hasattr(self.vector_db, "get_database_type"):
                    # FaissManager 有这个方法
                    db_type = self.vector_db.get_database_type().value
                elif hasattr(self.vector_db, "_is_lite"):
                    # MilvusManager 有这个属性
                    db_type = "milvus_lite" if self.vector_db._is_lite else "milvus"
                else:
                    # 从配置中获取数据库类型
                    db_type = self.config.get("vector_database_type", "unknown")

                logger.info(f"正在断开与 {db_type} 数据库的连接...")

                # 调用断开连接方法
                disconnect_result = self.vector_db.disconnect()
                if disconnect_result is None or disconnect_result:
                    logger.info("向量数据库连接已成功断开。")
                else:
                    logger.warning("向量数据库断开连接时返回失败状态。")
            except Exception as e:
                logger.error(f"停止插件时与向量数据库交互出错: {e}", exc_info=True)
        else:
            logger.info("向量数据库未初始化或已断开连接，无需断开。")

        logger.info("Mnemosyne 插件已停止。")
        return
