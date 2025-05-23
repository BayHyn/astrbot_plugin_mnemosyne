# -*- coding: utf-8 -*-
"""
Mnemosyne 插件核心记忆操作逻辑。
包括 RAG (Retrieval Augmented Generation) 查询、LLM 响应处理、记忆注入等。
记忆总结和存储相关逻辑已移至 `summarization_service.py`。
"""

import time # 用于时间相关操作，例如生成时间戳（如果需要）
import asyncio # 用于异步操作，例如运行 executor 中的同步函数
from datetime import datetime # 用于格式化时间戳
from typing import TYPE_CHECKING, List, Dict, Optional, Any, cast # 类型注解

# AstrBot API 导入
from astrbot.api.provider import LLMResponse, ProviderRequest # LLM 提供者相关
from astrbot.api.event import AstrMessageEvent # 消息事件对象
from pymilvus.exceptions import MilvusException # Milvus 特定异常
from pymilvus.orm.search import Hit, Hits # Milvus 搜索结果类型 (用于更精确的类型提示)

# 插件内部模块导入
from .tools import ( # 导入工具函数，用于清理上下文中的记忆标签
    remove_mnemosyne_tags,
    remove_system_mnemosyne_tags,
    remove_system_content,
)
from . import summarization_service # 导入总结服务模块
from .constants import ( # 导入常量
    VECTOR_FIELD_NAME, DEFAULT_TOP_K, DEFAULT_MILVUS_TIMEOUT, DEFAULT_PERSONA_ON_NONE,
    PRIMARY_FIELD_NAME, # 在 _process_milvus_hits 中用于日志和可能的补充
    MEMORY_OPS_LOG_NAME, ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM,
    CONFIG_KEY_USE_PERSONALITY_FILTERING, CONFIG_KEY_TOP_K, CONFIG_KEY_MILVUS_SEARCH_TIMEOUT,
    CONFIG_KEY_LONG_MEMORY_PREFIX, CONFIG_KEY_LONG_MEMORY_SUFFIX, CONFIG_KEY_MEMORY_ENTRY_FORMAT,
    CONFIG_KEY_MEMORY_INJECTION_METHOD, CONFIG_KEY_CONTEXTS_MEMORY_LEN,
    INJECTION_METHOD_USER_PROMPT, INJECTION_METHOD_SYSTEM_PROMPT, INJECTION_METHOD_INSERT_SYSTEM_PROMPT
)

# 类型提示，避免循环导入
if TYPE_CHECKING:
    from ..main import Mnemosyne # 引用主插件类 Mnemosyne

# 获取日志记录器
from astrbot.core.log import LogManager
logger = LogManager.GetLogger(log_name=MEMORY_OPS_LOG_NAME) # 使用常量中定义的日志名称


async def handle_query_memory(
    plugin: "Mnemosyne", event: AstrMessageEvent, req: ProviderRequest
) -> None:
    """
    处理 LLM 请求前的 RAG (Retrieval Augmented Generation) 检索逻辑。
    此函数负责：
    1. 检查 RAG 执行的前提条件（如 Milvus 和 Embedding 服务是否可用）。
    2. 获取当前会话和人格（Persona）信息。
    3. 初始化或更新会话上下文。
    4. 将用户的新消息添加到上下文中并增加消息计数。
    5. 对用户查询进行向量化。
    6. 执行 Milvus 向量搜索以检索相关记忆。
    7. 将检索到的记忆格式化并注入到 LLM 的请求中。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例，包含配置和各种管理器。
        event (AstrMessageEvent): 触发此操作的消息事件对象。
        req (ProviderRequest): 即将发送给 LLM 服务提供者的请求对象，将被修改以注入记忆。
    """
    # --- 1. 前置条件检查 ---
    if not await _check_rag_prerequisites(plugin):
        logger.debug("RAG 前提条件未满足，跳过记忆查询。")
        return

    try:
        # --- 2. 获取会话和人格信息 ---
        persona_id: Optional[str] = await _get_persona_id(plugin, event)
        session_id: Optional[str] = await plugin.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        if not session_id: # 理论上 AstrBot 应保证 session_id 存在
            logger.error("无法获取当前会话 ID (session_id)，记忆查询中止。")
            return

        # --- 3. 初始化或更新会话上下文 ---
        # 如果是新会话，从 ProviderRequest 初始化上下文历史
        if session_id not in plugin.context_manager.conversations: # type: ignore
            logger.info(f"会话 {session_id} 为新会话，使用 ProviderRequest 中的上下文进行初始化。")
            plugin.context_manager.init_conv(session_id, req.contexts, event) # type: ignore

        # --- 4. 清理旧记忆标签并添加新用户消息 ---
        clean_contexts(plugin, req) # 从 req.contexts 中移除之前可能注入的记忆标签

        # 将当前用户消息添加到插件管理的上下文中
        # ROLE_USER 常量代表 "user"
        plugin.context_manager.add_message(session_id, ROLE_USER, req.prompt, event) # type: ignore
        # 增加此会话的消息计数
        plugin.msg_counter.increment_counter(session_id) # type: ignore
        logger.debug(f"用户消息已添加到会话 {session_id} 的上下文中并更新了消息计数。")

        # --- 5. RAG 搜索 ---
        detailed_results: Optional[List[Dict[str, Any]]] = [] # 初始化为空列表
        try:
            # 5.1. 向量化用户查询
            query_embeddings: Optional[List[List[float]]]
            try:
                # 在 executor 中运行同步的 embedding 获取方法
                if not plugin.ebd: # 双重检查 ebd 是否有效
                    logger.error("Embedding 服务在查询时无效，无法向量化用户输入。")
                    return
                query_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, # 使用默认线程池
                    lambda: plugin.ebd.get_embeddings([req.prompt]), # type: ignore
                )
            except Exception as e_embed: # 捕获 embedding 过程中的任何异常
                logger.error(f"执行用户查询向量化时出错: {e_embed}", exc_info=True)
                query_embeddings = None # 标记向量化失败

            if not query_embeddings or not query_embeddings[0]:
                logger.error("无法获取用户查询的 Embedding 向量，RAG 检索中止。")
                return
            query_vector: List[float] = query_embeddings[0] # 获取第一个（也是唯一的）查询向量

            # 5.2. 执行 Milvus 向量搜索
            detailed_results = await _perform_milvus_search(
                plugin, query_vector, session_id, persona_id
            )

            # 5.3. 格式化结果并注入到 LLM 请求中
            if detailed_results: # 如果找到了相关记忆
                _format_and_inject_memory(plugin, detailed_results, req)
            else:
                logger.info(f"会话 {session_id} 的 RAG 搜索未找到相关记忆。")

        except Exception as e_rag: # 捕获 RAG 搜索流程中的其他错误
            logger.error(f"处理长期记忆 RAG 查询时发生内部错误: {e_rag}", exc_info=True)
            return # 发生错误则不继续

    except Exception as e_main: # 捕获整个 handle_query_memory 流程的顶层错误
        logger.error(f"处理 LLM 请求前的记忆查询流程遭遇意外失败: {e_main}", exc_info=True)


async def handle_on_llm_resp(
    plugin: "Mnemosyne", event: AstrMessageEvent, resp: LLMResponse
) -> None:
    """
    处理 LLM 响应后的逻辑。
    主要职责：
    1. 校验响应角色（应为助手角色）。
    2. 获取会话和人格信息。
    3. 将 LLM 的响应添加到插件管理的上下文中并更新消息计数。
    4. 调用总结服务检查是否需要触发记忆总结。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件的实例。
        event (AstrMessageEvent): 触发此操作的消息事件对象。
        resp (LLMResponse): 从 LLM 服务提供者返回的响应对象。
    """
    # 仅处理来自助手 (assistant) 角色的响应
    # ROLE_ASSISTANT 常量代表 "assistant"
    if resp.role != ROLE_ASSISTANT:
        logger.debug(f"LLM 响应角色为 '{resp.role}' (非 '{ROLE_ASSISTANT}')，不进行记忆处理。")
        return

    try:
        session_id: Optional[str] = await plugin.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        if not session_id:
            logger.error("无法获取当前 session_id，无法记录 LLM 响应到 Mnemosyne。")
            return

        persona_id: Optional[str] = await _get_persona_id(plugin, event)

        # 将 LLM 响应添加到上下文并更新计数器
        # (注意：add_message 现在需要 event 参数，如果会话是新创建的)
        # 由于这是 LLM 响应，理论上会话已通过 handle_query_memory 初始化，event 对象应已存在于 context_manager 中
        plugin.context_manager.add_message(session_id, ROLE_ASSISTANT, resp.completion_text) # type: ignore
        plugin.msg_counter.increment_counter(session_id) # type: ignore
        logger.debug(f"LLM 响应已添加到会话 {session_id} 的上下文中并更新了消息计数。")
        logger.debug(f"LLM 返回的文本内容片段：'{resp.completion_text[:100]}...'")


        # 调用总结服务，判断是否需要基于当前上下文和计数器状态触发总结
        # summarization_service._check_and_trigger_summary 内部会处理计数器重置和总结时间更新
        await summarization_service._check_and_trigger_summary( # type: ignore
            plugin, # 传递插件实例
            session_id, # 当前会话 ID
            plugin.context_manager.get_history(session_id), # type: ignore # 获取当前会话的完整历史
            persona_id, # 当前人格 ID
        )

    except Exception as e:
        logger.error(f"处理 LLM 响应后的记忆记录或总结触发失败: {e}", exc_info=True)


async def _check_rag_prerequisites(plugin: "Mnemosyne") -> bool:
    """
    检查 RAG (Retrieval Augmented Generation) 查询的前提条件是否满足。
    确保 Milvus 服务、Embedding API 和消息计数器都已正确初始化。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。

    Returns:
        bool: 如果所有前提条件均满足，则返回 `True`；否则返回 `False`。
    """
    # 检查 Milvus 管理器及其连接状态
    if not plugin.milvus_manager or not plugin.milvus_manager.is_connected():
        logger.error("Milvus 服务未初始化或未连接，无法执行 RAG 查询。")
        return False
    # 检查 Embedding API 是否已初始化
    if not plugin.ebd:
        logger.error("Embedding API 未初始化，无法执行 RAG 查询（无法向量化）。")
        return False
    # 检查消息计数器是否已初始化 (虽然主要用于总结，但检查完整性)
    if not plugin.msg_counter:
        logger.error("消息计数器 (MessageCounter) 未初始化，RAG 流程可能受影响（例如总结触发）。")
        return False
    logger.debug("RAG 前提条件检查通过。")
    return True


async def _get_persona_id(
    plugin: "Mnemosyne", event: AstrMessageEvent
) -> Optional[str]:
    """
    获取当前会话的人格 (Persona) ID。
    会尝试从当前会话的上下文中获取，如果未设置，则尝试获取 AstrBot 的全局默认人格。
    如果两者均未配置，且插件配置了 `use_personality_filtering`，则可能使用 `DEFAULT_PERSONA_ON_NONE`。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
        event (AstrMessageEvent): 当前消息事件对象。

    Returns:
        Optional[str]: 人格 ID 字符串。如果无法确定人格或不使用人格过滤，则可能为 `None`。
    """
    session_id: Optional[str] = await plugin.context.conversation_manager.get_curr_conversation_id(
        event.unified_msg_origin
    )
    # 从 AstrBot 的会话管理器获取当前会话对象
    conversation = await plugin.context.conversation_manager.get_conversation(
        event.unified_msg_origin, session_id
    )
    persona_id: Optional[str] = conversation.persona_id if conversation else None # 获取会话特定的人格ID

    # 如果会话没有人格ID (或为特殊标记 "[%None]")，则尝试获取全局默认人格
    if not persona_id or persona_id == "[%None]":
        default_persona_info: Optional[Dict[str, Any]] = plugin.context.provider_manager.selected_default_persona
        persona_id = default_persona_info["name"] if default_persona_info and "name" in default_persona_info else None
        log_prefix = f"会话 {session_id}: " if session_id else "当前会话: "
        if persona_id:
            logger.info(f"{log_prefix}未设置特定人格，将使用全局默认人格: '{persona_id}'。")
        else: # 全局默认人格也未配置
            # CONFIG_KEY_USE_PERSONALITY_FILTERING 是 "use_personality_filtering"
            # DEFAULT_PERSONA_ON_NONE 是 "default_persona"
            if plugin.config.get(CONFIG_KEY_USE_PERSONALITY_FILTERING, False):
                persona_id = DEFAULT_PERSONA_ON_NONE # 使用预设的占位符人格ID
                logger.warning(
                    f"{log_prefix}及全局均未配置人格。由于启用了人格过滤，将使用占位符 '{persona_id}' 进行记忆操作。"
                )
            else: # 未启用人格过滤，且无明确人格ID
                logger.info(
                    f"{log_prefix}及全局均未配置人格，且未启用人格过滤。记忆操作将不区分人格。"
                )
                persona_id = None # 明确设为 None
    return persona_id


# _check_and_trigger_summary 和其他总结相关函数已移至 core.summarization_service.py
# （占位注释，实际代码已删除）


async def _perform_milvus_search(
    plugin: "Mnemosyne",
    query_vector: List[float],
    session_id: Optional[str],
    persona_id: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    """
    在 Milvus 中执行向量相似性搜索。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
        query_vector (List[float]): 用于查询的嵌入向量。
        session_id (Optional[str]): 当前会话的 ID，用于构建过滤表达式。
        persona_id (Optional[str]): 当前人格的 ID，用于构建过滤表达式（如果启用了人格过滤）。

    Returns:
        Optional[List[Dict[str, Any]]]: 包含搜索结果实体字典的列表。
                                         如果搜索无结果或发生错误，则返回 `None`。
    """
    # 构建 Milvus 搜索的过滤表达式 (filter expression)
    filters: List[str] = [f"{PRIMARY_FIELD_NAME} > 0"] # 基础过滤器，确保 memory_id 有效 (通常 auto_id 从1开始)

    # 如果有会话 ID，则添加会话过滤条件
    if session_id:
        filters.append(f'session_id == "{session_id}"') # 注意 Milvus 表达式中字符串需要双引号
    else: # 理论上 session_id 在 RAG 流程中应该存在
        logger.warning("执行 Milvus 搜索时无法获取当前 session_id，将不按特定会话过滤记忆！")

    # 如果启用了人格过滤 (use_personality_filtering) 且当前有人格 ID，则添加人格过滤条件
    # CONFIG_KEY_USE_PERSONALITY_FILTERING 是 "use_personality_filtering"
    use_personality_filtering: bool = plugin.config.get(CONFIG_KEY_USE_PERSONALITY_FILTERING, False)
    if use_personality_filtering and persona_id:
        filters.append(f'personality_id == "{persona_id}"')
        logger.debug(f"Milvus 搜索将使用人格 '{persona_id}' 进行过滤。")
    elif use_personality_filtering: # 启用了人格过滤但当前无人格
        logger.debug("启用了人格过滤，但当前无有效人格 ID，本次搜索不按人格进行过滤。")

    # 组合所有过滤条件
    search_expression: str = " and ".join(filters) if filters else ""

    collection_name: str = plugin.collection_name # 当前使用的集合名
    # CONFIG_KEY_TOP_K 是 "top_k", DEFAULT_TOP_K 是 5
    top_k: int = plugin.config.get(CONFIG_KEY_TOP_K, DEFAULT_TOP_K)
    # CONFIG_KEY_MILVUS_SEARCH_TIMEOUT 是 "milvus_search_timeout", DEFAULT_MILVUS_TIMEOUT 是 10.0
    timeout_seconds: float = plugin.config.get(CONFIG_KEY_MILVUS_SEARCH_TIMEOUT, DEFAULT_MILVUS_TIMEOUT)

    logger.info(
        f"开始在 Milvus 集合 '{collection_name}' 中搜索相关记忆 "
        f"(TopK: {top_k}, Filter: '{search_expression or '无过滤'}', Timeout: {timeout_seconds}s)"
    )

    try:
        # 使用 asyncio.wait_for 来为 Milvus 的同步搜索操作设置超时
        # MilvusManager.search 方法是同步的，因此需要在 executor 中运行以避免阻塞事件循环
        if not plugin.milvus_manager: # 防御性检查
             logger.error("MilvusManager 未初始化，无法执行搜索。")
             return None

        search_results_raw = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, # 使用默认线程池
                lambda: plugin.milvus_manager.search( # type: ignore
                    collection_name=collection_name,
                    query_vectors=[query_vector], # search 方法期望向量列表
                    vector_field=VECTOR_FIELD_NAME, # "vector"
                    search_params=plugin.search_params, # 从插件实例获取搜索参数
                    limit=top_k,
                    expression=search_expression,
                    output_fields=plugin.output_fields_for_query, # 从插件实例获取输出字段
                ),
            ),
            timeout=timeout_seconds, # 应用超时
        )
        # search_results_raw 的类型是 List[Hits]
        # Hits 是一个类列表对象，包含多个 Hit 对象
        # 每个 Hit 对象代表一个搜索命中的结果

    except asyncio.TimeoutError: # 特别捕获超时错误
        logger.error(f"Milvus 搜索操作因超时 ({timeout_seconds} 秒) 而取消。")
        return None
    except MilvusException as me: # 捕获 Milvus 特有的异常
        logger.error(f"Milvus 搜索操作失败: {me}", exc_info=True)
        return None
    except Exception as e: # 捕获其他潜在的未知错误
        logger.error(f"执行 Milvus 搜索时发生未知错误: {e}", exc_info=True)
        return None

    # 处理搜索结果
    if not search_results_raw or not search_results_raw[0]: # 检查是否有结果，以及第一个查询是否有命中
        logger.info("Milvus 向量搜索未找到任何相关记忆。")
        return None
    else:
        # search_results_raw[0] 是对应第一个（也是唯一一个）查询向量的 Hits 对象
        hits: Hits = search_results_raw[0]
        # 调用辅助函数处理 Hits 对象并提取详细实体数据
        detailed_results: List[Dict[str, Any]] = _process_milvus_hits(hits)
        logger.info(f"Milvus 搜索成功，处理了 {len(hits)} 个原始命中，提取了 {len(detailed_results)} 条有效记忆。")
        return detailed_results


def _process_milvus_hits(hits: Hits) -> List[Dict[str, Any]]:
    """
    处理 Milvus `Hits` 对象，提取每个 `Hit` 中的实体数据。
    `Hits` 对象表现得像一个列表，其中每个元素是一个 `Hit` 对象。
    每个 `Hit` 对象有一个 `entity` 属性，它通常需要调用 `.to_dict()` 来获取标准字典。

    Args:
        hits (Hits): 从 Milvus `search` 方法返回的 `Hits` 对象。

    Returns:
        List[Dict[str, Any]]: 一个包含从每个 `Hit` 中提取的实体数据的字典列表。
                              如果 `hits` 为空或处理过程中发生错误，可能返回空列表。
    """
    detailed_results: List[Dict[str, Any]] = []
    if not hits: # 如果没有命中结果
        return detailed_results

    try:
        num_hits = len(hits)
        logger.debug(f"开始处理 {num_hits} 条 Milvus 原始命中结果...")
        for i in range(num_hits):
            try:
                hit: Hit = hits[i] # 获取单个 Hit 对象
                # Hit 对象有一个 entity 属性，它是一个 Entity 对象
                # Entity 对象有 to_dict() 方法，但其返回的字典可能包含 'entity' 键，实际数据在该键下
                if hit and hasattr(hit, "entity") and hit.entity:
                    # Milvus 的 Hit.entity.to_dict() 返回的可能是 {'entity': {...actual_fields...}}
                    # 或者直接是 {...actual_fields...}，取决于版本或具体情况，做兼容处理
                    entity_raw_dict = hit.entity.to_dict()
                    entity_data: Optional[Dict[str, Any]]
                    if 'entity' in entity_raw_dict and isinstance(entity_raw_dict['entity'], dict):
                        entity_data = entity_raw_dict['entity']
                    else: # 假设 to_dict() 直接返回了字段字典
                        entity_data = entity_raw_dict

                    if entity_data:
                        # 确保主键字段存在于实体数据中，这对于调试和某些操作很重要
                        if PRIMARY_FIELD_NAME not in entity_data and hasattr(hit, 'id'):
                            entity_data[PRIMARY_FIELD_NAME] = hit.id # 从 Hit 对象补充 id
                        detailed_results.append(entity_data)
                    else:
                        logger.warning(f"处理 Milvus 命中结果索引 {i} 时：实体数据为空或无效。原始Hit ID: {hit.id if hasattr(hit, 'id') else 'N/A'}")
                else:
                    logger.debug(f"处理 Milvus 命中结果索引 {i} 时：Hit 对象或其 entity 属性无效，已跳过。")
            except Exception as e_hit: # 处理单个 Hit 对象时发生的错误
                logger.error(f"处理 Milvus 单个命中结果 (索引 {i}) 时发生错误: {e_hit}", exc_info=True)
                # 选择继续处理下一个 Hit，而不是中断整个流程

    except Exception as e_hits_processing: # 处理整个 Hits 集合时发生的更严重错误
        logger.error(f"处理 Milvus Hits 集合时发生严重错误: {e_hits_processing}", exc_info=True)
        # 此时 detailed_results 可能不完整或为空

    logger.debug(f"成功从 Milvus Hits 中提取并处理了 {len(detailed_results)} 条记忆实体。")
    return detailed_results


def _format_and_inject_memory(
    plugin: "Mnemosyne", detailed_results: List[Dict[str, Any]], req: ProviderRequest
) -> None:
    """
    将从 Milvus 检索到的详细记忆结果格式化为字符串，并根据配置注入到 LLM 的请求中。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例，用于访问配置。
        detailed_results (List[Dict[str, Any]]): 包含从 Milvus 获取的记忆实体的字典列表。
        req (ProviderRequest): 即将发送给 LLM 的请求对象，将被修改。
    """
    if not detailed_results: # 如果没有详细结果，则不执行任何操作
        logger.info("没有可供格式化和注入的长期记忆。")
        return

    # 从配置中获取记忆片段的前缀、后缀和单条记忆的格式化字符串
    # CONFIG_KEY_LONG_MEMORY_PREFIX 等是常量
    long_memory_prefix: str = plugin.config.get(CONFIG_KEY_LONG_MEMORY_PREFIX, "<Mnemosyne> 长期记忆片段：")
    long_memory_suffix: str = plugin.config.get(CONFIG_KEY_LONG_MEMORY_SUFFIX, "</Mnemosyne>")
    memory_entry_format_template: str = plugin.config.get(CONFIG_KEY_MEMORY_ENTRY_FORMAT, "- [{time}] {content}")

    # 构建注入的长期记忆字符串
    formatted_memories_str: str = f"{long_memory_prefix}\n" # 添加前缀和换行

    for idx, result_entity in enumerate(detailed_results):
        content: str = result_entity.get("content", "内容缺失") # 获取记忆内容
        timestamp_val: Optional[float] = result_entity.get("create_time") # 获取创建时间戳
        time_str: str
        try:
            # 将 Unix 时间戳格式化为易读的日期时间字符串
            time_str = datetime.fromtimestamp(cast(float, timestamp_val)).strftime("%Y-%m-%d %H:%M") if timestamp_val else "未知时间"
        except (TypeError, ValueError, OSError) as e_time: # 处理无效时间戳
            logger.warning(f"格式化记忆条目 (ID: {result_entity.get(PRIMARY_FIELD_NAME, 'N/A')}) 的时间戳 '{timestamp_val}' 时出错: {e_time}")
            time_str = f"时间戳({timestamp_val})" if timestamp_val is not None else "未知时间"

        # 使用配置的格式化模板格式化单条记忆
        formatted_memories_str += memory_entry_format_template.format(time=time_str, content=content) + "\n"

    formatted_memories_str += long_memory_suffix # 添加后缀
    logger.info(f"已成功格式化 {len(detailed_results)} 条长期记忆，准备注入到提示中。")
    logger.debug(f"格式化后的记忆内容预览:\n{formatted_memories_str[:500]}...") # 记录部分内容用于调试

    # 根据配置的注入方法 (memory_injection_method) 将格式化记忆注入到请求中
    # CONFIG_KEY_MEMORY_INJECTION_METHOD 是 "memory_injection_method"
    # INJECTION_METHOD_USER_PROMPT, _SYSTEM_PROMPT, _INSERT_SYSTEM_PROMPT 是常量
    injection_method: str = plugin.config.get(CONFIG_KEY_MEMORY_INJECTION_METHOD, INJECTION_METHOD_USER_PROMPT)

    # 在注入前，先清理上下文中可能已存在的旧记忆标签
    clean_contexts(plugin, req)

    if injection_method == INJECTION_METHOD_USER_PROMPT: # "user_prompt"
        # 将记忆追加到用户提示 (prompt) 的最前面
        req.prompt = formatted_memories_str + "\n" + req.prompt
        logger.debug("长期记忆已注入到用户提示 (user_prompt) 的开头。")
    elif injection_method == INJECTION_METHOD_SYSTEM_PROMPT: # "system_prompt"
        # 将记忆追加到系统提示 (system_prompt)
        req.system_prompt = (req.system_prompt or "") + "\n" + formatted_memories_str # 确保 system_prompt 非 None
        logger.debug("长期记忆已追加到系统提示 (system_prompt)。")
    elif injection_method == INJECTION_METHOD_INSERT_SYSTEM_PROMPT: # "insert_system_prompt"
        # 将记忆作为一条新的系统角色消息插入到上下文历史 (contexts) 中
        # ROLE_SYSTEM 是 "system"
        req.contexts.append({"role": ROLE_SYSTEM, "content": formatted_memories_str})
        logger.debug("长期记忆已作为一条系统消息插入到上下文历史 (contexts) 中。")
    else: # 未知或不支持的注入方法
        logger.warning(
            f"配置了未知的记忆注入方法 '{injection_method}'，将默认采用追加到用户提示 (user_prompt) 的方式。"
        )
        req.prompt = formatted_memories_str + "\n" + req.prompt


def clean_contexts(plugin: "Mnemosyne", req: ProviderRequest) -> None:
    """
    根据配置的记忆注入方法，从 ProviderRequest 的上下文中移除之前由本插件注入的长期记忆标签。
    这是为了防止在多次交互或重新生成响应时，旧的记忆标签被重复处理或累积。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例，用于访问配置。
        req (ProviderRequest): 需要清理上下文的 ProviderRequest 对象。
    """
    # CONFIG_KEY_MEMORY_INJECTION_METHOD, INJECTION_METHOD_USER_PROMPT 等是常量
    injection_method: str = plugin.config.get(CONFIG_KEY_MEMORY_INJECTION_METHOD, INJECTION_METHOD_USER_PROMPT)
    # CONFIG_KEY_CONTEXTS_MEMORY_LEN 是 "contexts_memory_len"
    contexts_memory_len: int = plugin.config.get(CONFIG_KEY_CONTEXTS_MEMORY_LEN, 0) # 获取配置的上下文记忆长度

    # 根据注入方法选择相应的清理工具函数
    if injection_method == INJECTION_METHOD_USER_PROMPT:
        # 如果记忆是注入到用户提示中，则清理 req.contexts (通常是历史对话)
        req.contexts = remove_mnemosyne_tags(req.contexts, contexts_memory_len)
        logger.debug("已尝试从 req.contexts 中移除用户提示型记忆标签。")
    elif injection_method == INJECTION_METHOD_SYSTEM_PROMPT:
        # 如果记忆是注入到系统提示中，则清理 req.system_prompt
        req.system_prompt = remove_system_mnemosyne_tags(req.system_prompt or "", contexts_memory_len)
        logger.debug("已尝试从 req.system_prompt 中移除系统提示型记忆标签。")
    elif injection_method == INJECTION_METHOD_INSERT_SYSTEM_PROMPT:
        # 如果记忆是作为系统消息插入的，则从 req.contexts 中移除这类系统消息
        req.contexts = remove_system_content(req.contexts, contexts_memory_len)
        logger.debug("已尝试从 req.contexts 中移除作为系统消息注入的记忆内容。")
    # 如果是未知方法，在 _format_and_inject_memory 中已按 user_prompt 处理，此处无需额外操作
    return


# Summarization functions (_check_summary_prerequisites, _get_summary_llm_response,
# _extract_summary_text, _store_summary_to_milvus, handle_summary_long_memory,
# _periodic_summarization_check) have been moved to core.summarization_service.py
# （占位注释，实际代码已删除）
