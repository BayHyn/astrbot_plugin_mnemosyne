# -*- coding: utf-8 -*-
"""
Mnemosyne 插件的常量定义
"""

# --- Milvus 相关常量 ---
DEFAULT_COLLECTION_NAME = "mnemosyne_default_memory"  # 默认 Milvus 集合名称
DEFAULT_EMBEDDING_DIM = 768  # 默认嵌入向量维度, 常见开源模型如 m3e-small
PRIMARY_FIELD_NAME = "memory_id"  # 主键字段名
VECTOR_FIELD_NAME = "vector"  # 向量字段名
DEFAULT_TOP_K = 5  # 默认搜索返回结果数量
DEFAULT_MILVUS_PORT = 19530 # Milvus 默认端口号 (core/tools.py)
DEFAULT_MILVUS_TIMEOUT = 10.0  # Milvus 操作默认超时时间 (秒)
DEFAULT_INDEX_PARAMS = { # 默认索引参数
    "metric_type": "L2",
    "index_type": "AUTOINDEX", # AUTOINDEX 通常能为小规模数据提供不错的性能
    "params": {}, # AUTOINDEX 通常不需要额外参数，其他类型如 IVF_FLAT 可能需要 {"nlist": 1024}
}
DEFAULT_SEARCH_PARAMS = { # 默认搜索参数, metric_type 应与索引一致
    "metric_type": "L2", # 确保与索引的 metric_type 匹配
    "params": {"nprobe": 10}, # 示例参数, AUTOINDEX 通常忽略此参数，IVF_* 等类型需要
}


# --- RAG/记忆处理 相关常量 ---
DEFAULT_PERSONA_ON_NONE = "default_persona"  # 当人格信息缺失时使用的默认人格标识
DEFAULT_OUTPUT_FIELDS = ["content", "create_time", "session_id", "personality_id", PRIMARY_FIELD_NAME] # 查询时默认输出的字段,确保主键在内
ROLE_USER = "user" # 用户角色标识
ROLE_ASSISTANT = "assistant" # 助手角色标识
ROLE_SYSTEM = "system" # 系统角色标识 (用于记忆注入)
MNEMOSYNE_TAG_REGEX_PATTERN = r"<Mnemosyne>.*?</Mnemosyne>" # 用于匹配记忆标签的正则表达式 (core/tools.py)


# --- 总结任务 相关常量 ---
# 默认后台总结检查任务的配置
DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS = 300  # 默认检查间隔 (5分钟)
DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS = 1800 # 默认时间阈值 (30分钟)
DEFAULT_LONG_MEMORY_PROMPT = "请将以下多轮对话历史总结为一段简洁、客观、包含关键信息的长期记忆条目:" # 默认的总结提示词 (core/summarization_service.py)


# --- 日志和插件信息 ---
LOG_NAME = "Mnemosyne" # 插件主日志记录器名称
INIT_LOG_NAME = "MnemosyneInit" # 初始化过程专用日志记录器名称
MEMORY_OPS_LOG_NAME = "MnemosyneMemOps" # 记忆操作模块日志名称
SUMMARIZATION_LOG_NAME = "MnemosyneSummarizer" # 总结服务模块日志名称
CONTEXT_MANAGER_LOG_NAME = "MnemosyneCtxMgr" # 上下文管理器日志名称 (memory_manager/context_manager.py)
MESSAGE_COUNTER_LOG_NAME = "MnemosyneMsgCount" # 消息计数器日志名称 (memory_manager/message_counter.py)
MILVUS_MANAGER_LOG_NAME = "MnemosyneMilvusMgr" # Milvus管理器日志名称
TOOLS_LOG_NAME = "MnemosyneTools" # 工具模块日志名称 (虽然 tools.py 当前未使用 logger)
EMBEDDING_API_LOG_NAME = "MnemosyneEmbeddingAPI" # Embedding API 模块日志名称 (memory_manager/embedding.py)


# --- Context Manager 字典键名 ---
CTX_HISTORY = "history" # 对话历史记录列表的键名
CTX_LAST_SUMMARY_TIME = "last_summary_time" # 上次总结时间戳的键名
CTX_EVENT = "event" # AstrMessageEvent 对象的键名
CTX_MESSAGE_ROLE = "role" # 消息角色 (user, assistant, system) 的键名
CTX_MESSAGE_CONTENT = "content" # 消息内容的键名
CTX_MESSAGE_TIMESTAMP = "timestamp" # 消息时间戳的键名

# --- Message Counter (SQLite) 相关常量 ---
DB_DEFAULT_DIR_NAME = "mnemosyne_data" # 默认数据库存放目录名 (memory_manager/message_counter.py)
DB_DEFAULT_MESSAGE_COUNTS_FILENAME = "message_counters.db" # 默认消息计数数据库文件名 (memory_manager/message_counter.py)
DB_TABLE_MESSAGE_COUNTS = "message_counts" # 消息计数表名
DB_TABLE_SESSION_SUMMARY_TIMES = "session_summary_times" # 上次总结时间表名
DB_COLUMN_SESSION_ID = "session_id" # 会话ID列名
DB_COLUMN_COUNT = "count" # 消息计数值列名
DB_COLUMN_LAST_SUMMARY_TIMESTAMP = "last_summary_timestamp" # 上次总结时间戳列名


# --- 配置键名 (部分常用或易变的) ---
# Milvus 连接配置
CONFIG_KEY_MILVUS_LITE_PATH = "milvus_lite_path"
CONFIG_KEY_MILVUS_ADDRESS = "address" # 标准 Milvus 服务地址 (host:port 或 URI)
CONFIG_KEY_MILVUS_DB_NAME = "db_name"
CONFIG_KEY_MILVUS_CONNECTION_ALIAS = "connection_alias"
CONFIG_KEY_MILVUS_AUTHENTICATION = "authentication" # Milvus 认证配置的父键
CONFIG_KEY_AUTH_USER = "user" # Milvus 用户名
CONFIG_KEY_AUTH_PASSWORD = "password" # Milvus 密码
CONFIG_KEY_AUTH_TOKEN = "token" # Milvus token
CONFIG_KEY_AUTH_SECURE = "secure" # Milvus 是否启用 TLS/SSL

# Milvus Schema 和索引配置
CONFIG_KEY_COLLECTION_NAME = "collection_name"
CONFIG_KEY_EMBEDDING_DIM = "embedding_dim"
CONFIG_KEY_ENABLE_DYNAMIC_FIELD = "enable_dynamic_field"
CONFIG_KEY_INDEX_PARAMS = "index_params"
CONFIG_KEY_SEARCH_PARAMS = "search_params"
CONFIG_KEY_OUTPUT_FIELDS = "output_fields"
CONFIG_KEY_CREATE_INDEX_TIMEOUT = "create_index_timeout" # 创建索引的超时时间
CONFIG_KEY_MILVUS_FLUSH_AFTER_INSERT = "milvus_flush_after_insert" # main.py, milvus_manager.py

# Embedding 服务配置
CONFIG_KEY_EMBEDDING_SERVICE = "embedding_service"
CONFIG_KEY_EMBEDDING_MODEL = "embedding_model"
CONFIG_KEY_EMBEDDING_KEY = "embedding_key"
CONFIG_KEY_EMBEDDING_URL = "embedding_url"

# 总结任务配置
CONFIG_KEY_SUMMARY_CHECK_TASK = "summary_check_task"
CONFIG_KEY_LLM_PROVIDERS = "LLM_providers" # main.py 中用于获取 LLM provider 的配置键
CONFIG_KEY_SUMMARY_CHECK_INTERVAL = "SUMMARY_CHECK_INTERVAL_SECONDS" # main.py
CONFIG_KEY_SUMMARY_TIME_THRESHOLD = "SUMMARY_TIME_THRESHOLD_SECONDS" # main.py
CONFIG_KEY_NUM_PAIRS = "num_pairs" # 插件自身配置，但与 AstrBot 配置相关联
CONFIG_KEY_LONG_MEMORY_PROMPT = "long_memory_prompt" # summarization_service.py
CONFIG_KEY_SUMMARY_LLM_CONFIG = "summary_llm_config" # summarization_service.py
CONFIG_KEY_DEFAULT_PERSONA_ON_NONE = "default_persona_id_on_none" # summarization_service.py

# RAG / 记忆注入配置
CONFIG_KEY_USE_PERSONALITY_FILTERING = "use_personality_filtering" # memory_operations.py
CONFIG_KEY_MILVUS_SEARCH_TIMEOUT = "milvus_search_timeout" # memory_operations.py
CONFIG_KEY_TOP_K = "top_k" # memory_operations.py
CONFIG_KEY_LONG_MEMORY_PREFIX = "long_memory_prefix" # memory_operations.py
CONFIG_KEY_LONG_MEMORY_SUFFIX = "long_memory_suffix" # memory_operations.py
CONFIG_KEY_MEMORY_ENTRY_FORMAT = "memory_entry_format" # memory_operations.py
CONFIG_KEY_MEMORY_INJECTION_METHOD = "memory_injection_method" # memory_operations.py
CONFIG_KEY_CONTEXTS_MEMORY_LEN = "contexts_memory_len" # 插件自身配置

# AstrBot 平台配置 (用于 main.py reset_session_memory_cmd)
CONFIG_KEY_PLATFORM_SETTINGS = "platform_settings"
CONFIG_KEY_UNIQUE_SESSION = "unique_session"

# --- 嵌入服务相关默认值 ---
DEFAULT_EMBEDDING_SERVICE = "openai"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_ADAPTER_STAR_NAME = "astrbot_plugin_embedding_adapter"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1" # memory_manager/embedding.py

# --- 环境变量名 ---
ENV_VAR_OPENAI_API_KEY = "OPENAI_API_KEY" # memory_manager/embedding.py
ENV_VAR_GEMINI_API_KEY = "GEMINI_API_KEY" # memory_manager/embedding.py


# --- 命令相关 ---
MAX_TOTAL_FETCH_RECORDS = 1000 # list_records 命令一次性从 Milvus 拉取的最大记录数，用于排序找最新的记录
CONFIRM_FLAG = "--confirm" # 用于确认危险操作的命令标志
DEFAULT_LIST_RECORDS_LIMIT = 5 # list_records 命令默认显示的记录数
MAX_LIST_RECORDS_LIMIT = 50 # list_records 命令允许显示的最大记录数
CONTENT_PREVIEW_MAX_LENGTH = 200 # list_records 命令中内容预览的最大长度

# --- 其他 ---
DEFAULT_DB_NAME = "default" # Milvus 默认数据库名称
DEFAULT_CONNECTION_ALIAS_PREFIX = "mnemosyne_" # Milvus 连接别名前缀
DEFAULT_ADDRESS_PROTOCOL = "http://" # 解析地址时默认添加的协议 (core/tools.py)

# AstrBot 配置相关 (用于 initialize_config_check)
CONFIG_KEY_PROVIDER_SETTINGS = "provider_settings"
CONFIG_KEY_MAX_CONTEXT_LENGTH = "max_context_length"

# 记忆注入方法名
INJECTION_METHOD_USER_PROMPT = "user_prompt"
INJECTION_METHOD_SYSTEM_PROMPT = "system_prompt"
INJECTION_METHOD_INSERT_SYSTEM_PROMPT = "insert_system_prompt"
