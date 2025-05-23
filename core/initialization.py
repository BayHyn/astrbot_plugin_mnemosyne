# -*- coding: utf-8 -*-
"""
Mnemosyne 插件初始化逻辑。
包含配置加载、Schema 定义、Milvus 连接和设置、其他核心组件初始化等。
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Any
import asyncio # 虽然当前未使用，但保留以备将来异步初始化需求
import re # 正则表达式，用于从适配器生成集合名
from pymilvus import CollectionSchema, FieldSchema, DataType # Milvus Schema 相关

from astrbot.core.log import LogManager # AstrBot 日志管理器

# 导入插件内部的常量和工具函数
from .constants import (
    DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_DIM, PRIMARY_FIELD_NAME,
    VECTOR_FIELD_NAME, DEFAULT_OUTPUT_FIELDS, INIT_LOG_NAME,
    CONFIG_KEY_PROVIDER_SETTINGS, CONFIG_KEY_MAX_CONTEXT_LENGTH, CONFIG_KEY_NUM_PAIRS,
    CONFIG_KEY_CONTEXTS_MEMORY_LEN, CONFIG_KEY_EMBEDDING_DIM, CONFIG_KEY_COLLECTION_NAME,
    CONFIG_KEY_ENABLE_DYNAMIC_FIELD, CONFIG_KEY_INDEX_PARAMS, DEFAULT_INDEX_PARAMS,
    CONFIG_KEY_SEARCH_PARAMS, DEFAULT_SEARCH_PARAMS, CONFIG_KEY_OUTPUT_FIELDS,
    CONFIG_KEY_MILVUS_LITE_PATH, CONFIG_KEY_MILVUS_ADDRESS, CONFIG_KEY_MILVUS_DB_NAME,
    DEFAULT_DB_NAME, CONFIG_KEY_MILVUS_CONNECTION_ALIAS, DEFAULT_CONNECTION_ALIAS_PREFIX,
    CONFIG_KEY_MILVUS_AUTHENTICATION, CONFIG_KEY_AUTH_USER, CONFIG_KEY_AUTH_PASSWORD,
    CONFIG_KEY_AUTH_TOKEN, CONFIG_KEY_AUTH_SECURE, CONFIG_KEY_CREATE_INDEX_TIMEOUT,
    EMBEDDING_ADAPTER_STAR_NAME, CONFIG_KEY_EMBEDDING_SERVICE, CONFIG_KEY_EMBEDDING_MODEL,
    CONFIG_KEY_EMBEDDING_KEY, CONFIG_KEY_EMBEDDING_URL, DEFAULT_EMBEDDING_SERVICE,
    DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL
)
from .tools import parse_address # 地址解析工具

# 导入插件内的其他管理器和服务类
from ..memory_manager.message_counter import MessageCounter
from ..memory_manager.vector_db.milvus_manager import MilvusManager
from ..memory_manager.embedding import OpenAIEmbeddingAPI, GeminiEmbeddingAPI
from ..memory_manager.context_manager import ConversationContextManager

# 类型提示，用于避免循环导入问题
if TYPE_CHECKING:
    from ..main import Mnemosyne # 引用主插件类 Mnemosyne

# 获取初始化专用的日志记录器，使用常量中定义的名称
init_logger = LogManager.GetLogger(log_name=INIT_LOG_NAME)


def initialize_config_check(plugin: "Mnemosyne") -> None:
    """
    执行插件配置参数的必要检查和校验。
    此函数会在主要初始化流程之前调用，以确保关键配置有效。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。

    Raises:
        ValueError: 如果检测到配置不一致或无效，可能导致插件无法正常运行。
    """
    init_logger.debug("开始执行插件配置检查...")
    astrbot_config: Dict[str, Any] = plugin.context.get_config() # 获取 AstrBot 的全局配置

    # ------ 检查 num_pairs (记忆总结相关的对话轮次) ------
    # CONFIG_KEY_NUM_PAIRS 是 "num_pairs"
    num_pairs: int = plugin.config.get(CONFIG_KEY_NUM_PAIRS, 10) # 默认为10轮
    # CONFIG_KEY_PROVIDER_SETTINGS 是 "provider_settings"
    # CONFIG_KEY_MAX_CONTEXT_LENGTH 是 "max_context_length"
    provider_settings: Dict[str, Any] = astrbot_config.get(CONFIG_KEY_PROVIDER_SETTINGS, {})
    astrbot_max_context_length: int = provider_settings.get(CONFIG_KEY_MAX_CONTEXT_LENGTH, -1)

    # num_pairs 需要小于 AstrBot 配置的 max_context_length (LLM最大上下文窗口)
    # 如果 max_context_length 为 -1 (无限制) 或 0 (通常表示配置错误)，则跳过部分检查
    if astrbot_max_context_length > 0 and num_pairs > 2 * astrbot_max_context_length:
        # 乘以2是因为 num_pairs 指的是对话对，而 max_context_length 可能指消息条数或token数，这里做粗略比较
        raise ValueError(
            f"插件配置 `num_pairs` ({num_pairs}) 过大，可能超过 AstrBot 的最大上下文长度 "
            f"({CONFIG_KEY_MAX_CONTEXT_LENGTH}: {astrbot_max_context_length} * 2 = {2*astrbot_max_context_length} 条消息)。"
            "这可能导致总结时信息丢失或超出LLM处理范围。请调整 `num_pairs` 或 AstrBot 配置。"
        )
    elif astrbot_max_context_length == 0: # max_context_length 为0通常是无效配置
        raise ValueError(
            f"AstrBot 配置中的最大上下文长度 ({CONFIG_KEY_MAX_CONTEXT_LENGTH}) 为0，这是一个无效值。"
            "Mnemosyne 插件依赖此配置来确定合适的记忆处理参数，请将其设置为一个正整数或-1（无限制）。"
        )
    init_logger.debug(f"配置项 `num_pairs` ({num_pairs}) 与 AstrBot 上下文限制 ({astrbot_max_context_length}) 检查通过。")

    # ------ 检查 contexts_memory_len (注入到提示词的记忆长度) ------
    # CONFIG_KEY_CONTEXTS_MEMORY_LEN 是 "contexts_memory_len"
    contexts_memory_len: int = plugin.config.get(CONFIG_KEY_CONTEXTS_MEMORY_LEN, 0) # 默认为0，表示不限制或由其他逻辑处理
    if astrbot_max_context_length > 0 and contexts_memory_len > astrbot_max_context_length:
        raise ValueError(
            f"插件配置 `contexts_memory_len` ({contexts_memory_len}) 不能大于 AstrBot 的最大上下文长度 "
            f"({CONFIG_KEY_MAX_CONTEXT_LENGTH}: {astrbot_max_context_length})。"
            "过长的记忆注入可能导致提示词超出LLM处理上限。"
        )
    init_logger.debug(f"配置项 `contexts_memory_len` ({contexts_memory_len}) 检查通过。")
    init_logger.info("插件配置参数初步检查完成。")


def initialize_config_and_schema(plugin: "Mnemosyne") -> None:
    """
    解析插件配置，验证关键参数，并定义 Milvus 集合的 Schema 和相关索引/搜索参数。
    这些参数将存储在插件实例 (plugin) 的相应属性中。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。

    Raises:
        ValueError: 如果核心配置项 (如 embedding_dim) 无效。
        Exception: 其他在配置解析或 Schema 定义过程中发生的意外错误。
    """
    init_logger.debug("开始初始化插件配置和 Milvus Collection Schema...")
    try:
        # 获取嵌入向量的维度，这是定义 Schema 的关键参数
        # CONFIG_KEY_EMBEDDING_DIM 是 "embedding_dim", DEFAULT_EMBEDDING_DIM 是 768
        embedding_dim: int = plugin.config.get(CONFIG_KEY_EMBEDDING_DIM, DEFAULT_EMBEDDING_DIM)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"配置项 `{CONFIG_KEY_EMBEDDING_DIM}` ({embedding_dim}) 必须是一个正整数。")

        # 定义 Milvus 集合的字段 (Fields)
        fields: List[FieldSchema] = [
            FieldSchema(
                name=PRIMARY_FIELD_NAME, # "memory_id"
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True, # 自动生成主键 ID
                description="记忆条目的唯一标识符 (主键)",
            ),
            FieldSchema(
                name="personality_id", # 人格/角色 ID
                dtype=DataType.VARCHAR,
                max_length=256, # 适当的最大长度
                description="与此记忆关联的人格或角色标识",
            ),
            FieldSchema(
                name="session_id", # 会话 ID
                dtype=DataType.VARCHAR,
                max_length=72, # 根据实际会话ID格式调整
                description="此记忆所属的会话标识",
            ),
            FieldSchema(
                name="content", # 记忆内容
                dtype=DataType.VARCHAR,
                max_length=4096, # 限制记忆文本的最大长度，过长可能影响性能或存储
                description="记忆的文本内容 (例如对话摘要或关键信息片段)",
            ),
            FieldSchema(
                name=VECTOR_FIELD_NAME, # "vector"
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_dim, # 使用配置的嵌入维度
                description="记忆内容的嵌入向量，用于相似性搜索",
            ),
            FieldSchema(
                name="create_time", # 创建时间戳
                dtype=DataType.INT64, # Unix epoch 时间戳 (秒)
                description="记忆条目创建时的 Unix 时间戳",
            ),
        ]

        # 获取并设置集合名称
        # CONFIG_KEY_COLLECTION_NAME 是 "collection_name", DEFAULT_COLLECTION_NAME 是 "mnemosyne_default_memory"
        plugin.collection_name = plugin.config.get(CONFIG_KEY_COLLECTION_NAME, DEFAULT_COLLECTION_NAME)

        # 创建 CollectionSchema 对象
        # CONFIG_KEY_ENABLE_DYNAMIC_FIELD 是 "enable_dynamic_field"
        plugin.collection_schema = CollectionSchema(
            fields=fields,
            description=f"Mnemosyne 插件的长期记忆存储集合: {plugin.collection_name}",
            primary_field=PRIMARY_FIELD_NAME,
            enable_dynamic_field=plugin.config.get(CONFIG_KEY_ENABLE_DYNAMIC_FIELD, False), # 默认为 False
        )
        init_logger.info(f"Milvus Collection Schema '{plugin.collection_name}' 定义完成。动态字段: {plugin.collection_schema.enable_dynamic_field}")

        # 定义 Milvus 索引参数，从配置获取或使用默认值
        # CONFIG_KEY_INDEX_PARAMS 是 "index_params", DEFAULT_INDEX_PARAMS 是预定义的字典
        plugin.index_params = plugin.config.get(CONFIG_KEY_INDEX_PARAMS, DEFAULT_INDEX_PARAMS.copy())

        # 定义 Milvus 搜索参数，从配置获取或使用默认值
        # CONFIG_KEY_SEARCH_PARAMS 是 "search_params", DEFAULT_SEARCH_PARAMS 是预定义的字典
        default_search_params_copy = DEFAULT_SEARCH_PARAMS.copy()
        # 确保搜索参数的 metric_type 与索引参数一致
        default_search_params_copy["metric_type"] = plugin.index_params.get("metric_type", "L2")
        plugin.search_params = plugin.config.get(CONFIG_KEY_SEARCH_PARAMS, default_search_params_copy)

        # 定义查询时默认输出的字段列表
        # CONFIG_KEY_OUTPUT_FIELDS 是 "output_fields", DEFAULT_OUTPUT_FIELDS 是预定义的列表
        plugin.output_fields_for_query = plugin.config.get(CONFIG_KEY_OUTPUT_FIELDS, DEFAULT_OUTPUT_FIELDS[:]) # 使用切片创建副本

        init_logger.debug(f"Milvus 索引参数配置: {plugin.index_params}")
        init_logger.debug(f"Milvus 搜索参数配置: {plugin.search_params}")
        init_logger.debug(f"Milvus 查询默认输出字段: {plugin.output_fields_for_query}")
        init_logger.info("插件配置和 Milvus Collection Schema 初始化成功。")

    except Exception as e:
        init_logger.error(f"初始化插件配置和 Schema 过程中发生错误: {e}", exc_info=True)
        raise # 重新抛出异常，以便在主插件的 __init__ 方法中捕获并处理


def initialize_milvus(plugin: "Mnemosyne") -> None:
    """
    初始化 `MilvusManager` 实例。
    此函数会根据插件配置决定连接到 Milvus Lite (本地模式) 还是标准的 Milvus 服务器。
    成功连接后，会进一步调用函数设置必要的集合和索引。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。

    Raises:
        ValueError: 如果 Milvus 地址配置格式不正确。
        ConnectionError: 如果连接到 Milvus 服务失败。
        Exception: 其他在初始化 MilvusManager 或后续设置中发生的意外错误。
    """
    init_logger.debug("开始初始化 MilvusManager 连接和相关设置...")
    connect_args: Dict[str, Any] = {} # 用于收集传递给 MilvusManager 构造函数的参数
    is_lite_mode: bool = False # 标记是否为 Milvus Lite 模式

    try:
        # 1. 优先检查 Milvus Lite 的本地路径配置
        # CONFIG_KEY_MILVUS_LITE_PATH 是 "milvus_lite_path"
        lite_path: Optional[str] = plugin.config.get(CONFIG_KEY_MILVUS_LITE_PATH)

        # 2. 获取标准 Milvus 服务的地址配置
        # CONFIG_KEY_MILVUS_ADDRESS 是 "address"
        milvus_address: Optional[str] = plugin.config.get(CONFIG_KEY_MILVUS_ADDRESS)

        # 决定连接模式：Lite 优先，然后是标准服务地址，最后是默认 (通常是 Lite 的隐式路径)
        if lite_path:
            # --- 检测到显式的 Milvus Lite 配置 ---
            init_logger.info(f"检测到 Milvus Lite 配置，将使用指定本地路径: '{lite_path}'")
            connect_args["lite_path"] = lite_path
            is_lite_mode = True
            if milvus_address: # 如果同时配置了标准地址，发出警告
                init_logger.warning(
                    f"同时配置了 '{CONFIG_KEY_MILVUS_LITE_PATH}' 和 '{CONFIG_KEY_MILVUS_ADDRESS}'。"
                    f"将优先使用 Milvus Lite 路径，并忽略标准地址 ('{milvus_address}')。"
                )
        elif milvus_address:
            # --- 未配置 Lite 路径，尝试使用标准 Milvus 服务地址 ---
            init_logger.info(f"未配置 Milvus Lite 路径，将根据 '{CONFIG_KEY_MILVUS_ADDRESS}' ({milvus_address}) 配置连接标准 Milvus。")
            is_lite_mode = False
            # 判断地址是 URI (如 http://, https://, unix:)还是 host:port 格式
            if milvus_address.startswith(("http://", "https://", "unix:")):
                init_logger.debug(f"地址 '{milvus_address}' 被识别为 URI 格式。")
                connect_args["uri"] = milvus_address
            else: # 假定为 host:port 格式
                init_logger.debug(f"地址 '{milvus_address}' 将被解析为 host:port 格式。")
                try:
                    host, port_str = parse_address(milvus_address) # 使用工具函数解析
                    connect_args["host"] = host
                    connect_args["port"] = port_str # MilvusManager 内部会处理 str/int
                except ValueError as e: # 解析失败
                    raise ValueError(
                        f"解析标准 Milvus 服务地址 '{milvus_address}' (应为 host:port 或有效URI) 失败: {e}"
                    ) from e
        else:
            # --- 既没有显式 Lite 路径也没有标准地址，MilvusManager 将使用其内部默认 Lite 路径 ---
            init_logger.warning(
                f"未在配置中找到 '{CONFIG_KEY_MILVUS_LITE_PATH}' 或 '{CONFIG_KEY_MILVUS_ADDRESS}'。"
                "MilvusManager 将尝试使用默认的 Milvus Lite 设置 (通常在项目数据目录下)。"
            )
            # is_lite_mode 此时仍为 False，让 MilvusManager 自行决定是否进入其默认 Lite 模式

        # 3. 添加通用参数 (对 Lite 和 Standard 都可能有效)
        #    数据库名称 (db_name)
        #    CONFIG_KEY_MILVUS_DB_NAME 是 "db_name", DEFAULT_DB_NAME 是 "default"
        db_name: str = plugin.config.get(CONFIG_KEY_MILVUS_DB_NAME, DEFAULT_DB_NAME)
        if db_name != DEFAULT_DB_NAME: # 仅当非默认时才传递，保持参数简洁
            connect_args["db_name"] = db_name
            init_logger.info(f"将尝试连接到 Milvus 数据库: '{db_name}'。")
        else:
            init_logger.debug(f"将使用 Milvus 的默认数据库 ('{DEFAULT_DB_NAME}')。")

        #    连接别名 (connection_alias)
        #    CONFIG_KEY_MILVUS_CONNECTION_ALIAS 是 "connection_alias"
        #    DEFAULT_CONNECTION_ALIAS_PREFIX 是 "mnemosyne_"
        default_alias: str = f"{DEFAULT_CONNECTION_ALIAS_PREFIX}{plugin.collection_name}"
        alias: str = plugin.config.get(CONFIG_KEY_MILVUS_CONNECTION_ALIAS, default_alias)
        connect_args["alias"] = alias
        init_logger.debug(f"设置 Milvus 连接别名为: '{alias}'。")

        # 4. 添加仅适用于标准 Milvus 的认证和安全参数 (如果不是显式 Lite 模式)
        #    注意：如果 is_lite_mode 为 False 但 connect_args 为空 (即用户未提供任何路径/地址)，
        #    MilvusManager 仍可能自行决定使用默认 Lite 模式。这种情况下，这些参数会被 MilvusManager 内部忽略。
        if not connect_args.get("lite_path"): # 更准确地判断是否应添加标准参数
            init_logger.debug("为标准 Milvus 连接准备认证和安全设置（如果已在配置中提供）。")
            # CONFIG_KEY_MILVUS_AUTHENTICATION 是 "authentication"
            auth_config: Dict[str, Any] = plugin.config.get(CONFIG_KEY_MILVUS_AUTHENTICATION, {})

            added_auth_params_log: List[str] = [] # 用于日志记录添加的参数
            # CONFIG_KEY_AUTH_USER, _PASSWORD, _TOKEN, _SECURE
            for key, config_key in [
                ("user", CONFIG_KEY_AUTH_USER),
                ("password", CONFIG_KEY_AUTH_PASSWORD),
                ("token", CONFIG_KEY_AUTH_TOKEN),
                ("secure", CONFIG_KEY_AUTH_SECURE),
            ]:
                config_value = auth_config.get(config_key)
                if config_value is not None:
                    if key == 'secure': # 特别处理 'secure'，确保是布尔值
                        secure_bool = str(config_value).lower() == 'true' if isinstance(config_value, str) else bool(config_value)
                        connect_args[key] = secure_bool
                        added_auth_params_log.append(f"{key}={secure_bool}")
                    else:
                        connect_args[key] = config_value
                        # 不记录 password 和 token 的实际值到日志
                        added_auth_params_log.append(f"{key}={'******' if key in ['password', 'token'] else config_value}")

            if added_auth_params_log:
                init_logger.info(f"从配置中为标准 Milvus 连接添加了参数: {', '.join(added_auth_params_log)}。")
            else:
                init_logger.debug("未在配置中找到额外的 Milvus 认证或安全设置。")
        elif connect_args.get("lite_path"): # 如果是显式 Lite 模式
            # 检查并警告用户是否配置了不适用于 Lite 模式的参数
            auth_config = plugin.config.get(CONFIG_KEY_MILVUS_AUTHENTICATION, {})
            ignored_lite_keys = [
                k for k_conf in [CONFIG_KEY_AUTH_USER, CONFIG_KEY_AUTH_PASSWORD, CONFIG_KEY_AUTH_TOKEN, CONFIG_KEY_AUTH_SECURE]
                if (k := k_conf.split('.')[-1]) in auth_config and auth_config[k] is not None # 获取authentication下的键名
            ]
            if ignored_lite_keys:
                init_logger.warning(f"当前为 Milvus Lite 模式，配置中的以下认证/安全参数将被忽略: {ignored_lite_keys}。")

        # 5. 初始化 MilvusManager 实例
        # 从日志参数中移除敏感信息
        loggable_connect_args = {
            k: v for k, v in connect_args.items() if k not in ['password', 'token']
        }
        init_logger.info(f"准备使用以下参数初始化 MilvusManager: {loggable_connect_args}。")
        plugin.milvus_manager = MilvusManager(**connect_args)

        # 6. 检查连接状态
        if not plugin.milvus_manager or not plugin.milvus_manager.is_connected():
            # 获取实际运行模式用于日志
            actual_mode_name = "Milvus Lite" if plugin.milvus_manager and plugin.milvus_manager._is_lite else "标准 Milvus"
            raise ConnectionError(
                f"初始化 MilvusManager 或连接到 {actual_mode_name} (别名: {alias}) 失败。"
                "请检查配置、Milvus 服务状态和网络连接。"
            )

        actual_mode_name = "Milvus Lite" if plugin.milvus_manager._is_lite else "标准 Milvus"
        init_logger.info(f"成功连接到 {actual_mode_name} (别名: {alias})。")

        # 7. 设置（创建或验证）Milvus 集合和索引
        init_logger.debug("开始设置 Milvus Collection 和相关索引...")
        setup_milvus_collection_and_index(plugin) # 调用辅助函数处理
        init_logger.info("Milvus Collection 和索引设置流程已成功调用。")

        init_logger.info("Milvus 初始化流程成功完成。")

    except Exception as e:
        init_logger.error(f"Milvus 初始化或设置过程中发生严重错误: {e}", exc_info=True)
        plugin.milvus_manager = None  # 确保在初始化失败时 manager 被设为 None，避免后续操作出错
        raise # 重新抛出异常，以便主插件 __init__ 中可以捕获并决定是否中止插件加载


def setup_milvus_collection_and_index(plugin: "Mnemosyne") -> None:
    """
    确保 Milvus 集合存在、Schema 一致，并且向量字段有索引。
    如果集合不存在，则创建它。如果索引不存在，则创建它。
    最后，确保集合被加载到内存中以供搜索。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。

    Raises:
        RuntimeError: 如果创建集合或加载集合等关键步骤失败。
    """
    # 前置条件检查
    if not plugin.milvus_manager or not plugin.collection_schema:
        init_logger.critical("无法设置 Milvus 集合/索引：MilvusManager 或 CollectionSchema 未正确初始化。")
        raise RuntimeError("MilvusManager 或 CollectionSchema 未准备好，无法继续设置。")

    collection_name: str = plugin.collection_name

    # 检查集合是否存在
    if plugin.milvus_manager.has_collection(collection_name):
        init_logger.info(f"Milvus 集合 '{collection_name}' 已存在。将进行 Schema 一致性检查...")
        check_schema_consistency(plugin, collection_name, plugin.collection_schema)
        # 注意: check_schema_consistency 目前主要记录警告，不一定会阻止后续操作，除非设计如此
    else:
        # 如果集合不存在，则创建集合
        init_logger.info(f"Milvus 集合 '{collection_name}' 未找到。正在尝试创建...")
        collection_created = plugin.milvus_manager.create_collection(
            collection_name, plugin.collection_schema
        )
        if not collection_created:
            # 创建集合失败是一个严重问题，应阻止插件继续
            raise RuntimeError(f"创建 Milvus 集合 '{collection_name}' 失败。请检查日志和服务状态。")
        init_logger.info(f"成功创建 Milvus 集合 '{collection_name}'。")
        # 新创建的集合需要创建索引
        ensure_milvus_index(plugin, collection_name)

    # 再次确保索引存在（即使集合已存在也应检查一遍，以防索引被意外删除或创建失败）
    ensure_milvus_index(plugin, collection_name)

    # 确保集合已加载到内存中以供搜索
    init_logger.info(f"正在确保 Milvus 集合 '{collection_name}' 已加载到内存以备搜索...")
    if not plugin.milvus_manager.load_collection(collection_name):
        # 加载失败可能是资源问题或索引未就绪
        init_logger.error(
            f"加载 Milvus 集合 '{collection_name}' 到内存失败。搜索功能可能无法正常工作或效率低下。"
            "请检查 Milvus 服务状态、资源以及索引是否已成功构建。"
        )
        # 根据插件的容错策略，这里可以决定是否抛出异常。
        # 如果搜索是核心功能，加载失败应视为严重问题。
        # raise RuntimeError(f"加载 Milvus 集合 '{collection_name}' 失败。")
    else:
        init_logger.info(f"Milvus 集合 '{collection_name}' 已成功加载到内存。")


def ensure_milvus_index(plugin: "Mnemosyne", collection_name: str) -> None:
    """
    检查指定集合的向量字段 (`VECTOR_FIELD_NAME`) 是否已创建索引。
    如果索引不存在，则尝试根据插件配置中的 `index_params` 创建它。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。
        collection_name (str): 需要检查或创建索引的集合名称。

    Raises:
        Exception: 如果在检查或创建索引过程中发生 Milvus 通信等严重错误。
                   简单的索引创建失败（如 Milvus 返回 False）会记录错误日志，但可能不直接抛出。
    """
    if not plugin.milvus_manager: # 防御性检查
        init_logger.error("MilvusManager 未初始化，无法执行 ensure_milvus_index。")
        return

    try:
        has_vector_index: bool = False
        # 确保操作的集合确实存在，避免对不存在的集合进行操作
        if not plugin.milvus_manager.has_collection(collection_name):
            init_logger.warning(
                f"尝试为不存在的 Milvus 集合 '{collection_name}' 检查/创建索引，已跳过此操作。"
            )
            return

        # 获取集合对象以检查其上的现有索引
        collection = plugin.milvus_manager.get_collection(collection_name)
        if collection and collection.indexes: # 确保 collection 对象有效且有索引列表
            for index_obj in collection.indexes: # index_obj 是 Index 对象
                # 检查索引是否是为我们配置的向量字段 (VECTOR_FIELD_NAME) 创建的
                if index_obj.field_name == VECTOR_FIELD_NAME:
                    init_logger.info(
                        f"在 Milvus 集合 '{collection_name}' 上检测到向量字段 '{VECTOR_FIELD_NAME}' 的现有索引 (名称: {index_obj.index_name})。"
                    )
                    # 可选：更严格地检查索引类型和参数是否与配置匹配
                    # ... (此处可以添加对 index_obj.params 等的比较逻辑) ...
                    has_vector_index = True
                    break # 找到目标字段的索引即可
        elif not collection:
            init_logger.warning(
                f"无法获取 Milvus 集合 '{collection_name}' 的详细对象来验证索引信息，可能集合刚被删除或存在连接问题。"
            )
            # 此时不宜继续创建索引，因为集合状态未知

        # 如果遍历完所有索引后，仍未发现目标向量字段的索引，则尝试创建
        if not has_vector_index and collection: # 再次确认 collection 对象有效
            init_logger.warning(
                f"Milvus 集合 '{collection_name}' 的向量字段 '{VECTOR_FIELD_NAME}' 上未找到索引。正在尝试创建..."
            )
            # 使用配置好的索引参数 (plugin.index_params) 创建索引
            # CONFIG_KEY_CREATE_INDEX_TIMEOUT 是 "create_index_timeout", 默认值 600 秒
            create_timeout: Optional[float] = plugin.config.get(CONFIG_KEY_CREATE_INDEX_TIMEOUT) # 类型可能是 Optional[int] 或 float
            if create_timeout is not None:
                create_timeout = float(create_timeout)

            index_success: bool = plugin.milvus_manager.create_index(
                collection_name=collection_name,
                field_name=VECTOR_FIELD_NAME,
                index_params=plugin.index_params,
                timeout=create_timeout, # 使用从配置获取的超时或 None
            )
            if not index_success:
                init_logger.error(
                    f"为 Milvus 集合 '{collection_name}' 的字段 '{VECTOR_FIELD_NAME}' 创建索引失败。"
                    "搜索性能将受到严重影响。请检查 Milvus 服务日志以获取详细原因。"
                )
                # 根据插件策略，这里可以决定是否抛出异常，如果索引是绝对必要的
            else:
                init_logger.info(
                    f"已成功为 Milvus 集合 '{collection_name}' 的字段 '{VECTOR_FIELD_NAME}' 发送索引创建请求。"
                    "索引将在后台构建。请注意，构建完成可能需要一些时间。"
                )
                # 创建索引后，通常不需要立即等待其构建完成，除非后续操作强依赖
                # MilvusManager.create_index 内部已包含 wait_for_index_building_complete

    except Exception as e:
        init_logger.error(f"检查或创建 Milvus 集合 '{collection_name}' 的索引时发生意外错误: {e}", exc_info=True)
        # 重新抛出异常，这可能会阻止插件正常启动，但能暴露问题
        raise


def initialize_components(plugin: "Mnemosyne") -> None:
    """
    初始化插件的非 Milvus 核心组件，例如消息计数器、上下文管理器以及嵌入服务 API。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。

    Raises:
        ValueError: 如果嵌入服务配置不完整或不支持。
        ConnectionError: 如果嵌入服务连接测试失败 (如果实现了 test_connection)。
        Exception: 其他在组件初始化过程中发生的意外错误。
    """
    init_logger.debug("开始初始化插件的其他核心组件...")

    # 1. 初始化消息计数器 (MessageCounter) 和对话上下文管理器 (ConversationContextManager)
    try:
        # MessageCounter 必须首先初始化，因为 ConversationContextManager 依赖它
        plugin.msg_counter = MessageCounter() # 使用默认数据库文件路径
        init_logger.info("消息计数器 (MessageCounter) 初始化成功。")

        plugin.context_manager = ConversationContextManager(message_counter=plugin.msg_counter)
        init_logger.info("会话上下文管理器 (ConversationContextManager) 初始化成功 (已启用上次总结时间持久化)。")
    except Exception as e:
        init_logger.error(f"初始化消息计数器或上下文管理器失败: {e}", exc_info=True)
        raise # 这些是核心组件，失败则插件无法工作

    # 2. 初始化嵌入服务 API (Embedding API)
    try:
        # 首先尝试通过 AstrBot 的嵌入服务适配器插件获取服务
        try:
            # EMBEDDING_ADAPTER_STAR_NAME 是 "astrbot_plugin_embedding_adapter"
            embedding_adapter_star = plugin.context.get_registered_star(EMBEDDING_ADAPTER_STAR_NAME)
            if embedding_adapter_star and hasattr(embedding_adapter_star.star_cls, 'get_dim') and hasattr(embedding_adapter_star.star_cls, 'get_model_name'):
                plugin.ebd = embedding_adapter_star.star_cls
                dim = plugin.ebd.get_dim() # type: ignore
                model_name = plugin.ebd.get_model_name() # type: ignore
                if dim is not None and model_name is not None:
                    # 更新插件配置中的维度和集合名（如果通过适配器获取）
                    plugin.config[CONFIG_KEY_EMBEDDING_DIM] = dim
                    plugin.config[CONFIG_KEY_COLLECTION_NAME] = "ea_" + re.sub(r'[^a-zA-Z0-9]', '_', model_name)
                    init_logger.info(f"已通过嵌入服务适配器 '{EMBEDDING_ADAPTER_STAR_NAME}' 成功加载嵌入服务: {model_name} (维度: {dim})。")
                else: # 适配器有效但未返回维度或模型名
                    init_logger.warning(f"嵌入服务适配器 '{EMBEDDING_ADAPTER_STAR_NAME}' 未能提供有效的维度和模型名称。将尝试配置化加载。")
                    plugin.ebd = None # 重置为 None，以便后续逻辑处理
            else: # 未找到适配器或适配器不符合预期接口
                init_logger.debug(f"未找到兼容的嵌入服务适配器插件 '{EMBEDDING_ADAPTER_STAR_NAME}'。将尝试配置化加载。")
                plugin.ebd = None
        except Exception as e_adapter: # 捕获加载适配器过程中的任何异常
            init_logger.warning(f"尝试加载嵌入服务适配器 '{EMBEDDING_ADAPTER_STAR_NAME}' 时失败: {e_adapter}", exc_info=True)
            plugin.ebd = None # 确保 ebd 为 None

        # 如果通过适配器未能加载嵌入服务 (plugin.ebd 仍为 None)，则根据插件配置自行初始化
        if plugin.ebd is None:
            init_logger.info("将根据插件自身配置初始化嵌入服务...")
            # CONFIG_KEY_EMBEDDING_MODEL, CONFIG_KEY_EMBEDDING_KEY 是必须的
            required_embedding_keys: List[str] = [CONFIG_KEY_EMBEDDING_MODEL, CONFIG_KEY_EMBEDDING_KEY]
            missing_keys: List[str] = [key for key in required_embedding_keys if not plugin.config.get(key)]
            if missing_keys: # 如果缺少关键配置
                raise ValueError(f"无法初始化嵌入服务：插件配置中缺少以下必需项: {', '.join(missing_keys)}。")

            # CONFIG_KEY_EMBEDDING_SERVICE, DEFAULT_EMBEDDING_SERVICE ("openai")
            embedding_service_name: str = plugin.config.get(CONFIG_KEY_EMBEDDING_SERVICE, DEFAULT_EMBEDDING_SERVICE).lower()

            if embedding_service_name == "gemini":
                plugin.ebd = GeminiEmbeddingAPI(
                    model=plugin.config.get(CONFIG_KEY_EMBEDDING_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL),
                    api_key=plugin.config.get(CONFIG_KEY_EMBEDDING_KEY),
                )
                init_logger.info(f"已选择并初始化 Gemini ({plugin.ebd.model}) 作为嵌入服务提供商。")
            elif embedding_service_name == "openai": # 默认为 "openai"
                plugin.ebd = OpenAIEmbeddingAPI(
                    model=plugin.config.get(CONFIG_KEY_EMBEDDING_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL),
                    api_key=plugin.config.get(CONFIG_KEY_EMBEDDING_KEY),
                    base_url=plugin.config.get(CONFIG_KEY_EMBEDDING_URL), # 可选，用于 OpenAI 兼容 API
                )
                init_logger.info(f"已选择并初始化 OpenAI ({plugin.ebd.model}) 作为嵌入服务提供商。")
            else: # 不支持的嵌入服务类型
                raise ValueError(f"配置中指定的嵌入服务提供商 '{embedding_service_name}' 不被支持。请选择 'openai' 或 'gemini'。")

        # 对初始化完成的 embedding 服务进行连接测试 (如果服务类提供了 test_connection 方法)
        if hasattr(plugin.ebd, 'test_connection') and callable(plugin.ebd.test_connection):
            try:
                plugin.ebd.test_connection() # type: ignore
                init_logger.info("嵌入服务 API 初始化成功，连接测试通过。")
            except ConnectionError as conn_err: # 捕获特定的 ConnectionError
                init_logger.error(f"嵌入服务 API 连接测试失败: {conn_err}", exc_info=True)
                # 决定是否允许插件在 Embedding API 连接失败时继续运行
                # raise # 如果嵌入是绝对核心功能，可以选择重新抛出 ConnectionError
                init_logger.warning("嵌入服务连接测试失败。插件将继续运行，但嵌入相关功能将不可用。")
                plugin.ebd = None  # 明确设为 None 表示嵌入服务不可用
            except Exception as test_err: # 捕获 test_connection 中其他可能的异常
                 init_logger.error(f"嵌入服务 API 连接测试时发生未知错误: {test_err}", exc_info=True)
                 plugin.ebd = None
        else: # 如果没有 test_connection 方法
            init_logger.warning(
                f"嵌入服务 API 类 '{type(plugin.ebd).__name__}' 没有提供 'test_connection' 方法，跳过连接测试。"
                "请确保API配置正确。"
            )

    except Exception as e_embed_init: # 捕获整个嵌入服务初始化块的异常
        init_logger.error(f"初始化嵌入服务 API 失败: {e_embed_init}", exc_info=True)
        plugin.ebd = None  # 确保初始化失败时 ebd 属性为 None
        raise # 嵌入是核心功能，其初始化失败应阻止插件正常运行或至少给出明确指示

    init_logger.info("插件其他核心组件初始化完成。")


def check_schema_consistency(
    plugin: "Mnemosyne", collection_name: str, expected_schema: CollectionSchema
) -> bool:
    """
    检查现有 Milvus 集合的 Schema 是否与插件预期的 Schema 一致。
    此函数主要比较字段名称、数据类型、主键设置和 auto_id 等。
    如果发现不一致，会记录警告信息，但通常不阻止插件运行（除非不一致性非常严重）。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。
        collection_name (str): 需要检查 Schema 一致性的集合名称。
        expected_schema (CollectionSchema): 插件期望的 Milvus CollectionSchema 对象。

    Returns:
        bool: 如果 Schema 基本一致或集合不存在（无需检查）则返回 True。
              如果检测到显著的不一致或检查过程中发生错误，则返回 False。
    """
    # 如果 Milvus 管理器未初始化或指定集合不存在，则无需（也无法）检查一致性
    if not plugin.milvus_manager or not plugin.milvus_manager.has_collection(collection_name):
        init_logger.debug(f"Milvus 集合 '{collection_name}' 不存在或 MilvusManager 未初始化，无需进行 Schema 一致性检查。")
        return True # 没有可供比较的现有集合，认为“一致”于预期（即期望它被创建）

    try:
        # 获取实际存在的集合对象
        collection = plugin.milvus_manager.get_collection(collection_name)
        if not collection: # 获取集合对象失败
            init_logger.error(f"无法获取 Milvus 集合 '{collection_name}' 的详细信息以检查 Schema。")
            return False # 视为不一致或检查失败

        actual_schema: CollectionSchema = collection.schema # 获取实际的 Schema
        # 将期望的和实际的字段转换为以字段名为键的字典，方便查找和比较
        expected_fields_dict: Dict[str, FieldSchema] = {f.name: f for f in expected_schema.fields}
        actual_fields_dict: Dict[str, FieldSchema] = {f.name: f for f in actual_schema.fields}

        is_consistent: bool = True # 初始假定 Schema是一致的
        warnings_list: List[str] = [] # 用于收集不一致的警告信息

        # 1. 检查期望的字段是否存在于实际 Schema 中，并比较其属性
        for field_name, expected_field_schema in expected_fields_dict.items():
            actual_field_schema = actual_fields_dict.get(field_name)
            if not actual_field_schema: # 期望的字段在实际 Schema 中缺失
                warnings_list.append(f"配置中期望的字段 '{field_name}' 在 Milvus 集合的实际 Schema 中缺失。")
                is_consistent = False
                continue # 跳过对此字段的后续属性比较

            # 比较字段的数据类型
            if actual_field_schema.dtype != expected_field_schema.dtype:
                # 特殊处理向量类型，需要额外比较维度 (dim)
                is_expected_vector = expected_field_schema.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]
                is_actual_vector = actual_field_schema.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]

                if is_expected_vector and is_actual_vector: # 如果两者都是向量类型，则比较维度
                    expected_dim = expected_field_schema.params.get("dim")
                    actual_dim = actual_field_schema.params.get("dim")
                    if expected_dim != actual_dim:
                        warnings_list.append(
                            f"字段 '{field_name}' 的向量维度不匹配 (预期: {expected_dim}, 实际: {actual_dim})。"
                        )
                        is_consistent = False
                # 特殊处理 VARCHAR 类型，比较最大长度 (max_length)
                elif expected_field_schema.dtype == DataType.VARCHAR and actual_field_schema.dtype == DataType.VARCHAR:
                    expected_max_len = expected_field_schema.params.get("max_length")
                    actual_max_len = actual_field_schema.params.get("max_length")
                    if expected_max_len is not None and actual_max_len is not None and actual_max_len < expected_max_len:
                        warnings_list.append(
                            f"字段 '{field_name}' (VARCHAR) 的实际最大长度 ({actual_max_len}) 小于预期 ({expected_max_len})，可能导致数据截断。"
                        )
                        is_consistent = False # 这通常是较严重的问题
                    elif expected_max_len is not None and actual_max_len is not None and actual_max_len > expected_max_len:
                        warnings_list.append(
                            f"字段 '{field_name}' (VARCHAR) 的实际最大长度 ({actual_max_len}) 大于预期 ({expected_max_len})。这通常是可接受的。"
                        )
                        # is_consistent 保持不变，因为通常长度更长是兼容的
                else: # 其他数据类型不匹配
                    warnings_list.append(
                        f"字段 '{field_name}' 的数据类型不匹配 (预期: {expected_field_schema.dtype}, 实际: {actual_field_schema.dtype})。"
                    )
                    is_consistent = False

            # 比较主键 (is_primary) 属性
            if actual_field_schema.is_primary != expected_field_schema.is_primary:
                warnings_list.append(f"字段 '{field_name}' 的主键 (is_primary) 状态与配置不匹配。")
                is_consistent = False
            # 比较主键的 auto_id 属性 (仅当字段是主键时此比较才有意义)
            if expected_field_schema.is_primary and (actual_field_schema.auto_id != expected_field_schema.auto_id):
                warnings_list.append(f"主键字段 '{field_name}' 的 auto_id 属性与配置不匹配。")
                is_consistent = False

        # 2. 检查实际 Schema 中是否存在配置中未定义的字段
        #    (仅当未启用动态字段时，这算作一个警告)
        if not plugin.collection_schema.enable_dynamic_field: # enable_dynamic_field 来自插件配置
            for field_name in actual_fields_dict:
                if field_name not in expected_fields_dict:
                    warnings_list.append(
                        f"Milvus 集合的实际 Schema 中发现未在插件配置中定义的字段 '{field_name}' (当前未启用动态字段)。"
                    )
                    # is_consistent = False # 是否将此视为严重不一致取决于策略，通常仅为警告

        # 根据检查结果记录日志
        if not is_consistent:
            full_warning_message = (
                f"Milvus 集合 '{collection_name}' 的 Schema 与当前插件配置存在一处或多处潜在不一致：\n - "
                + "\n - ".join(warnings_list)
                + "\n请仔细检查您的 Milvus 集合结构或插件配置。"
                "Schema 不一致可能导致运行时错误、数据处理问题或搜索结果不准确。"
            )
            init_logger.warning(full_warning_message)
        else:
            init_logger.info(f"Milvus 集合 '{collection_name}' 的 Schema 与当前插件配置基本一致。")

        return is_consistent # 返回最终的一致性状态

    except Exception as e:
        init_logger.error(
            f"检查 Milvus 集合 '{collection_name}' Schema 一致性时发生意外错误: {e}", exc_info=True
        )
        return False # 发生检查错误时，保守地认为 Schema 不一致
