# -*- coding: utf-8 -*-
"""
Mnemosyne 插件核心记忆总结服务。
此模块包含与记忆内容总结相关的所有功能，包括：
- 检查总结的前提条件。
- 调用 LLM 服务进行文本总结。
- 从 LLM 响应中提取总结内容。
- 将总结后的记忆及其向量存储到 Milvus 数据库。
- 定期检查会话是否需要进行总结。
"""

import time # 用于获取当前时间戳
import asyncio # 用于异步任务管理
from datetime import datetime # (当前未直接使用，但通常用于时间格式化)
from typing import TYPE_CHECKING, List, Dict, Optional, Any, cast # 类型注解

# AstrBot API 和 Milvus 异常导入
from astrbot.api.provider import LLMResponse # LLM 响应对象
from pymilvus.exceptions import MilvusException # Milvus 特定异常

# 插件内部模块导入
from .tools import format_context_to_string # 工具函数：格式化上下文历史为字符串
from .constants import ( # 导入常量
    VECTOR_FIELD_NAME, DEFAULT_PERSONA_ON_NONE, SUMMARIZATION_LOG_NAME, ROLE_SYSTEM,
    CONFIG_KEY_LONG_MEMORY_PROMPT, DEFAULT_LONG_MEMORY_PROMPT, CONFIG_KEY_SUMMARY_LLM_CONFIG,
    CONFIG_KEY_DEFAULT_PERSONA_ON_NONE, CONFIG_KEY_NUM_PAIRS, CONFIG_KEY_MILVUS_FLUSH_AFTER_INSERT
)

# 类型提示，避免循环导入
if TYPE_CHECKING:
    from ..main import Mnemosyne # 引用主插件类 Mnemosyne
    from astrbot.api.provider import Provider # LLM Provider 类型

# 获取日志记录器
from astrbot.core.log import LogManager
logger = LogManager.GetLogger(log_name=SUMMARIZATION_LOG_NAME) # 使用常量中定义的日志名称


async def _check_summary_prerequisites(plugin: "Mnemosyne", memory_text: str) -> bool:
    """
    检查执行记忆总结的前提条件是否满足。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
        memory_text (str): 待总结的记忆文本。

    Returns:
        bool: 如果所有前提条件（Milvus可用、Embedding API可用、文本非空）均满足，则返回 `True`，否则返回 `False`。
    """
    # 检查 Milvus 管理器及其连接状态
    if not plugin.milvus_manager or not plugin.milvus_manager.is_connected():
        logger.error("Milvus 服务不可用（未初始化或未连接），无法存储总结后的长期记忆。")
        return False
    # 检查 Embedding 服务是否已初始化
    if not plugin.ebd:
        logger.error("Embedding API 服务未初始化，无法向量化总结后的记忆。")
        return False
    # 检查待总结的文本是否有效（非空且不只包含空白字符）
    if not memory_text or not memory_text.strip():
        logger.warning("尝试总结空的或仅包含空白字符的记忆文本，操作已跳过。")
        return False
    logger.debug("记忆总结前提条件检查通过。")
    return True


async def _get_summary_llm_response(
    plugin: "Mnemosyne", memory_text: str
) -> Optional[LLMResponse]:
    """
    请求配置的 LLM 服务提供者对给定的记忆文本进行总结。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
        memory_text (str): 需要总结的原始记忆文本（通常是多轮对话历史）。

    Returns:
        Optional[LLMResponse]: LLM 服务返回的响应对象。如果请求失败或无法获取 LLM Provider，则返回 `None`。
    """
    llm_provider: Optional["Provider"] = plugin.provider # 获取插件实例中缓存的 LLM Provider
    try:
        # 如果插件实例中没有缓存 Provider，尝试从 AstrBot 上下文获取当前正在使用的 Provider
        if not llm_provider:
            llm_provider = plugin.context.get_using_provider()
            if not llm_provider: # 如果仍无法获取
                logger.error("无法获取用于总结记忆的 LLM Provider 实例。请检查插件和 AstrBot 配置。")
                return None
    except Exception as e: # 捕获获取 Provider 过程中可能发生的任何异常
        logger.error(f"获取 LLM Provider 实例时发生错误: {e}", exc_info=True)
        return None

    # 从插件配置中获取用于总结的提示词和 LLM 特定配置
    # CONFIG_KEY_LONG_MEMORY_PROMPT, DEFAULT_LONG_MEMORY_PROMPT
    long_memory_prompt_template: str = plugin.config.get(CONFIG_KEY_LONG_MEMORY_PROMPT, DEFAULT_LONG_MEMORY_PROMPT)
    # CONFIG_KEY_SUMMARY_LLM_CONFIG
    summary_llm_config: Dict[str, Any] = plugin.config.get(CONFIG_KEY_SUMMARY_LLM_CONFIG, {})

    logger.debug(
        f"准备请求 LLM 进行记忆总结。提示 (前50字符): '{long_memory_prompt_template[:50]}...', "
        f"待总结内容长度: {len(memory_text)}, LLM配置: {summary_llm_config}"
    )

    try:
        # 调用 LLM Provider 的 text_chat 方法进行总结
        # ROLE_SYSTEM 常量代表 "system"
        llm_response: LLMResponse = await llm_provider.text_chat(
            prompt=memory_text, # 待总结的文本作为用户输入
            contexts=[{"role": ROLE_SYSTEM, "content": long_memory_prompt_template}], # 总结提示词作为系统消息
            **summary_llm_config, # 应用任何特定于总结的 LLM 参数
        )
        logger.debug(f"LLM 成功返回总结响应。原始响应数据: {llm_response}")
        return llm_response
    except Exception as e: # 捕获 LLM 请求过程中的任何异常
        logger.error(f"请求 LLM 进行记忆总结失败: {e}", exc_info=True)
        return None


def _extract_summary_text(
    plugin: "Mnemosyne", llm_response: LLMResponse
) -> Optional[str]:
    """
    从 LLM 的响应对象中提取总结后的文本内容并进行基本校验。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例 (当前未使用，但保留以便未来扩展)。
        llm_response (LLMResponse): 从 LLM 服务返回的响应对象。

    Returns:
        Optional[str]: 提取并处理（去除首尾空白）后的总结文本。如果响应无效或文本为空，则返回 `None`。
    """
    completion_text: Optional[str] = None
    # LLMResponse 对象可能直接包含 completion_text，或者是一个包含该字段的字典
    if isinstance(llm_response, LLMResponse):
        completion_text = llm_response.completion_text
    elif isinstance(llm_response, dict): # 兼容字典形式的响应 (不太可能，但做防御)
        completion_text = llm_response.get("completion_text")
    else: # 未知响应类型
        logger.error(f"LLM 总结服务返回了未知类型的数据: {type(llm_response)}。无法提取总结文本。")
        return None

    # 校验提取的文本是否有效
    if not completion_text or not completion_text.strip():
        logger.error(f"LLM 总结响应无效或内容为空。原始响应: {llm_response}")
        return None

    summary_text: str = completion_text.strip() #去除首尾空白字符
    logger.info(f"LLM 成功生成并提取记忆总结，文本长度: {len(summary_text)}。")
    logger.debug(f"提取的总结内容 (前100字符): '{summary_text[:100]}...'")
    return summary_text


async def _store_summary_to_milvus(
    plugin: "Mnemosyne",
    persona_id: Optional[str],
    session_id: str,
    summary_text: str,
    embedding_vector: List[float],
) -> None:
    """
    将总结后的记忆文本及其对应的嵌入向量存储到 Milvus 数据库中。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
        persona_id (Optional[str]): 与此记忆关联的人格 ID。如果为 None，则使用配置的默认值。
        session_id (str): 此记忆所属的会话 ID。
        summary_text (str): 总结后的记忆文本内容。
        embedding_vector (List[float]): 总结文本的嵌入向量。
    """
    collection_name: str = plugin.collection_name # 获取当前使用的集合名称
    current_timestamp: int = int(time.time()) # 获取当前时间的 Unix 时间戳 (秒)

    # 确定用于存储的有效人格 ID
    # CONFIG_KEY_DEFAULT_PERSONA_ON_NONE, DEFAULT_PERSONA_ON_NONE
    effective_persona_id: str = (
        persona_id
        if persona_id
        else plugin.config.get(CONFIG_KEY_DEFAULT_PERSONA_ON_NONE, DEFAULT_PERSONA_ON_NONE)
    )

    # 准备要插入 Milvus 的数据记录
    data_to_insert: List[Dict[str, Any]] = [
        {
            "personality_id": effective_persona_id,
            "session_id": session_id,
            "content": summary_text,
            VECTOR_FIELD_NAME: embedding_vector, # VECTOR_FIELD_NAME 是 "vector"
            "create_time": current_timestamp,
        }
    ]

    logger.info(
        f"准备向 Milvus 集合 '{collection_name}' 插入 1 条总结记忆。"
        f"(Persona: {effective_persona_id}, Session: {session_id[:8]}...)" # 日志中截断较长的 session_id
    )

    loop = asyncio.get_event_loop() # 获取当前事件循环
    mutation_result: Optional[Any] = None # 初始化 Milvus 操作结果
    try:
        # 在 executor 中运行同步的 Milvus insert 方法
        if not plugin.milvus_manager: # 防御性检查
            logger.error("MilvusManager 未初始化，无法存储总结。")
            return
        mutation_result = await loop.run_in_executor(
            None, # 使用默认线程池
            lambda: plugin.milvus_manager.insert( # type: ignore
                collection_name=collection_name, data=data_to_insert
            ),
        )
    except MilvusException as me: # 捕获 Milvus 特定异常
        logger.error(f"向 Milvus 插入总结记忆时发生 MilvusException: {me}", exc_info=True)
        return # 插入失败，不继续后续操作
    except Exception as e: # 捕获其他潜在异常
        logger.error(f"向 Milvus 插入总结记忆时发生未知错误: {e}", exc_info=True)
        return # 插入失败

    # 检查插入操作的结果
    if mutation_result and mutation_result.insert_count > 0:
        inserted_ids = mutation_result.primary_keys
        logger.info(f"成功将总结记忆插入到 Milvus。插入记录的 ID(s): {inserted_ids}。")

        # 根据配置决定是否在插入后执行 flush 操作
        # CONFIG_KEY_MILVUS_FLUSH_AFTER_INSERT
        try:
            if plugin.config.get(CONFIG_KEY_MILVUS_FLUSH_AFTER_INSERT, False):
                logger.debug(
                    f"根据配置，正在刷新 (Flush) Milvus 集合 '{collection_name}' 以确保记忆立即可用..."
                )
                await loop.run_in_executor(
                    None,
                    lambda: plugin.milvus_manager.flush([collection_name]), # type: ignore
                )
                logger.debug(f"Milvus 集合 '{collection_name}' 刷新完成。")
            else: # 如果配置为不刷新
                logger.debug(f"根据配置，跳过在插入总结后刷新 Milvus 集合 '{collection_name}'。记忆可能稍后可见。")
        except MilvusException as me_flush: # 捕获 flush 过程中的 Milvus 异常
            logger.error(
                f"刷新 Milvus 集合 '{collection_name}' 时发生 MilvusException: {me_flush}",
                exc_info=True,
            )
        except Exception as flush_err: # 捕获 flush 过程中的其他异常
            logger.error(
                f"刷新 Milvus 集合 '{collection_name}' 时发生未知错误: {flush_err}",
                exc_info=True,
            )
    else: # 插入操作未成功或未插入任何记录
        logger.error(
            f"插入总结记忆到 Milvus 失败或未插入任何记录。MutationResult: {mutation_result}。"
            f"LLM 回复内容 (前100字符): '{summary_text[:100]}...'"
        )


async def handle_summary_long_memory(
    plugin: "Mnemosyne", persona_id: Optional[str], session_id: str, memory_text: str
) -> None:
    """
    处理长文本记忆的总结流程。
    这是一个后台任务，负责：
    1. 检查总结前提条件。
    2. 请求 LLM 进行文本总结。
    3. 提取总结文本。
    4. 获取总结文本的 Embedding 向量。
    5. 将总结文本和向量存储到 Milvus。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
        persona_id (Optional[str]): 当前人格 ID。
        session_id (str): 当前会话 ID。
        memory_text (str): 需要总结的原始对话历史文本。
    """
    # 1. 检查总结前提条件
    if not await _check_summary_prerequisites(plugin, memory_text):
        logger.debug("总结前提条件未满足，取消本次总结任务。")
        return

    try:
        # 2. 请求 LLM 进行总结
        llm_response: Optional[LLMResponse] = await _get_summary_llm_response(plugin, memory_text)
        if not llm_response: # 如果获取 LLM 响应失败
            logger.error("未能从 LLM 获取有效总结响应，总结任务中止。")
            return

        # 3. 从 LLM 响应中提取总结文本
        summary_text: Optional[str] = _extract_summary_text(plugin, llm_response)
        if not summary_text: # 如果提取总结文本失败
            logger.error("未能从 LLM 响应中提取有效总结文本，总结任务中止。")
            return

        # 4. 获取总结文本的 Embedding 向量
        embedding_vectors: Optional[List[List[float]]]
        try:
            # 在 executor 中运行同步的 embedding 获取
            if not plugin.ebd: # 双重检查 ebd 服务是否有效
                logger.error("Embedding 服务在总结时无效，无法向量化总结文本。")
                return
            embedding_vectors = await asyncio.get_event_loop().run_in_executor(
                None, # 使用默认线程池
                lambda: plugin.ebd.get_embeddings([summary_text]), # type: ignore
            )
        except Exception as e_embed: # 捕获向量化过程中的任何异常
            logger.error(
                f"获取总结文本 '{summary_text[:100]}...' 的 Embedding 时出错: {e_embed}",
                exc_info=True,
            )
            return # 获取 Embedding 失败，中止任务

        if not embedding_vectors or not embedding_vectors[0]: # 如果未能获取向量
            logger.error(f"无法为总结文本 '{summary_text[:100]}...' 获取 Embedding 向量，总结任务中止。")
            return
        embedding_vector: List[float] = embedding_vectors[0] # 获取第一个（也是唯一的）向量

        # 5. 将总结文本和向量存储到 Milvus
        await _store_summary_to_milvus(
            plugin, persona_id, session_id, summary_text, embedding_vector
        )
        logger.info(f"会话 {session_id} 的记忆总结、向量化和存储流程已成功完成。")

    except Exception as e_main_summary: # 捕获整个总结和存储流程中的其他严重错误
        logger.error(f"处理记忆总结和存储 (会话 {session_id}) 过程中发生严重错误: {e_main_summary}", exc_info=True)


async def _check_and_trigger_summary(
    plugin: "Mnemosyne",
    session_id: str,
    context_history: List[Dict[str, Any]], # 使用更精确的类型提示
    persona_id: Optional[str],
) -> None:
    """
    检查当前会话是否满足基于消息轮次的总结条件，如果满足，则异步创建并启动一个总结任务。
    此函数通常在处理完一次 LLM 响应后被调用。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
        session_id (str): 当前会话的 ID。
        context_history (List[Dict[str, Any]]): 当前会话的完整上下文历史记录。
        persona_id (Optional[str]): 当前人格的 ID。
    """
    # 确保插件组件已初始化
    if not plugin.msg_counter or not plugin.context_manager:
        logger.error("消息计数器或上下文管理器未初始化，无法执行基于轮次的总结检查。")
        return

    # CONFIG_KEY_NUM_PAIRS 是 "num_pairs"
    num_pairs_threshold: int = plugin.config.get(CONFIG_KEY_NUM_PAIRS, 10) # 默认10轮对话对

    # 调整计数器（如果需要）并检查是否达到总结阈值
    if plugin.msg_counter.adjust_counter_if_necessary(
        session_id, context_history
    ) and plugin.msg_counter.get_counter(session_id) >= num_pairs_threshold:

        logger.info(f"会话 {session_id} 已达到 {plugin.msg_counter.get_counter(session_id)}/{num_pairs_threshold} 轮对话，满足计数器总结条件。开始准备总结...")
        # 从上下文中格式化需要总结的内容
        # format_context_to_string 工具函数负责从历史记录中提取指定轮次的内容
        history_contents_to_summarize: str = format_context_to_string(
            context_history, num_pairs_threshold # 仅总结达到阈值的部分
        )
        
        # 异步创建总结任务，不在当前协程中等待其完成
        asyncio.create_task(
            handle_summary_long_memory(plugin, persona_id, session_id, history_contents_to_summarize)
        )
        logger.info(f"会话 {session_id} 的后台记忆总结任务已成功提交执行。")

        # 重置此会话的消息计数器，并更新内存中（及数据库中）的上次总结时间戳
        plugin.msg_counter.reset_counter(session_id)
        plugin.context_manager.update_summary_time(session_id) # 此方法内部会持久化时间
        logger.debug(f"会话 {session_id} 的消息计数器已重置，上次总结时间已更新。")
    else:
        logger.debug(f"会话 {session_id} 当前消息轮次 {plugin.msg_counter.get_counter(session_id)} 未达到总结阈值 {num_pairs_threshold}。")


async def _periodic_summarization_check(plugin: "Mnemosyne") -> None:
    """
    [后台任务] 定期检查所有活动会话，如果某个会话自上次总结以来已超过配置的时间阈值，
    并且该会话中有新的消息（即消息计数器 > 0），则触发一次强制总结。

    Args:
        plugin ("Mnemosyne"): Mnemosyne 插件实例。
    """
    # 动态导入 _get_persona_id 以避免可能的循环导入问题，
    # 尤其是在 _get_persona_id 逻辑复杂或未来可能移动的情况下。
    # 理想情况下，应在模块级别解决依赖，但此处作为一种策略。
    try:
        from .memory_operations import _get_persona_id as get_persona_id_for_periodic_check
    except ImportError:
        logger.critical(
            "无法从 .memory_operations 导入 _get_persona_id 函数，定期总结任务无法确定人格ID，将无法运行。"
        )
        return # 关键依赖缺失，任务无法执行

    logger.info(
        f"启动定期总结检查后台任务。检查间隔: {plugin.summary_check_interval} 秒, "
        f"总结时间阈值: {plugin.summary_time_threshold} 秒。"
    )
    while True: # 无限循环，构成后台任务的主体
        try:
            await asyncio.sleep(plugin.summary_check_interval) # 等待指定的检查间隔

            # 如果上下文管理器未初始化或时间阈值被设为无穷大（禁用），则跳过本次检查周期
            if not plugin.context_manager or plugin.summary_time_threshold == float("inf"):
                logger.debug("定期总结检查：上下文管理器未就绪或基于时间的总结已禁用，跳过本次检查。")
                continue

            current_time: float = time.time() # 获取当前时间戳
            # 获取所有当前活动会话的ID列表进行检查
            # plugin.context_manager.conversations 是一个字典，键为 session_id
            session_ids_to_check: List[str] = list(plugin.context_manager.conversations.keys()) # type: ignore
            logger.debug(f"定期总结检查：开始检查 {len(session_ids_to_check)} 个活动会话的总结超时...")

            for session_id in session_ids_to_check:
                try:
                    # 获取会话的完整上下文信息
                    session_data: Optional[Dict[str, Any]] = plugin.context_manager.get_session_context(session_id) # type: ignore
                    if not session_data: # 如果会话在获取 keys 后被移除，则跳过
                        logger.debug(f"定期总结检查：会话 {session_id} 在处理前被移除，跳过。")
                        continue
                    
                    # 获取此会话当前未总结的消息计数
                    message_count_for_summary: int = plugin.msg_counter.get_counter(session_id) # type: ignore
                    # 计算自上次总结以来的时间
                    time_since_last_summary: float = current_time - session_data.get("last_summary_time", current_time) # 若无上次时间，则视为刚发生

                    # 触发条件：(1) 会话中有新的消息，并且 (2) 自上次总结的时间已超过配置的阈值
                    if message_count_for_summary > 0 and \
                       time_since_last_summary > plugin.summary_time_threshold:
                        
                        logger.info(
                            f"会话 {session_id} 满足时间强制总结条件。"
                            f"距离上次总结已 {time_since_last_summary:.0f} 秒 (阈值: {plugin.summary_time_threshold} 秒)，"
                            f"且有 {message_count_for_summary} 条未总结消息。触发强制总结。"
                        )
                        
                        # 格式化需要总结的对话历史（所有未总结的消息）
                        history_to_summarize: str = format_context_to_string(
                            session_data.get("history", []), # 获取对话历史
                            message_count_for_summary # 总结所有待处理消息
                        )
                        
                        # 获取与此会话关联的人格 ID
                        persona_id: Optional[str] = None
                        event_for_persona: Optional[AstrMessageEvent] = session_data.get("event")
                        if not event_for_persona: # 如果上下文中没有存储 event 对象
                            logger.warning(f"会话 {session_id} 在定期总结时缺少 'event' 对象，无法确定人格 ID。将使用 None。")
                        else: #尝试获取人格ID
                            try:
                                persona_id = await get_persona_id_for_periodic_check(plugin, event_for_persona)
                            except Exception as e_pid:
                                logger.error(f"定期总结：获取会话 {session_id} 的人格ID时出错: {e_pid}", exc_info=True)
                                # 发生错误时，人格ID将保持为 None

                        # 异步创建并启动总结任务
                        asyncio.create_task(
                            handle_summary_long_memory(
                                plugin, persona_id, session_id, history_to_summarize
                            )
                        )
                        logger.info(f"会话 {session_id} 的后台强制总结任务已成功提交执行。")

                        # 重置消息计数器并更新上次总结时间（在任务提交后立即执行）
                        plugin.msg_counter.reset_counter(session_id) # type: ignore
                        plugin.context_manager.update_summary_time(session_id) # type: ignore
                        logger.debug(f"会话 {session_id} (强制总结后) 的消息计数器已重置，上次总结时间已更新。")
                    else: # 未满足时间强制总结条件
                        logger.debug(f"会话 {session_id} 未满足时间强制总结条件。消息数: {message_count_for_summary}, "
                                     f"距上次总结: {time_since_last_summary:.0f}s / {plugin.summary_time_threshold}s.")

                except KeyError: # 如果在处理过程中 session_data 结构不完整或会话被删除
                    logger.debug(f"定期总结检查：处理会话 {session_id} 时发生 KeyError (可能已被移除或结构不完整)，跳过。")
                except Exception as e_session_check: # 捕获检查单个会话时发生的其他错误
                    logger.error(
                        f"定期总结检查：检查或总结会话 {session_id} 时发生错误: {e_session_check}", exc_info=True
                    )

        except asyncio.CancelledError: # 捕获任务被取消的异常
            logger.info("定期总结检查后台任务已被取消。正在退出任务循环。")
            break # 退出 while True 循环，结束任务
        except Exception as e_main_loop: # 捕获主循环中发生的其他意外错误
            logger.error(f"定期总结检查后台任务的主循环发生严重错误: {e_main_loop}", exc_info=True)
            # 为避免因连续快速失败而刷屏日志，在重新进入下一个检查周期前稍作等待
            await asyncio.sleep(plugin.summary_check_interval / 2 if plugin.summary_check_interval > 10 else 5)
