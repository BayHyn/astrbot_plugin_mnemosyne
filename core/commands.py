# -*- coding: utf-8 -*-
"""
Mnemosyne 插件的命令处理函数实现。
这些函数是具体命令的后端逻辑，由 `main.py` 中的命令处理方法调用。
(注意：命令权限等装饰器已在 `main.py` 中应用)
"""

from typing import TYPE_CHECKING, Optional, AsyncGenerator, List, Dict, Any
from datetime import datetime

# 导入 AstrBot API 和类型
from astrbot.api.event import AstrMessageEvent

# 导入必要的模块和常量
from .constants import (
    PRIMARY_FIELD_NAME,
    MAX_TOTAL_FETCH_RECORDS,
    CONFIRM_FLAG, # "--confirm"
    DEFAULT_OUTPUT_FIELDS, # 用于 list_records
    CONTENT_PREVIEW_MAX_LENGTH,
    MAX_LIST_RECORDS_LIMIT,
    DEFAULT_LIST_RECORDS_LIMIT # 虽然主函数有默认值，这里校验时也可用
)

# 类型提示，避免循环导入
if TYPE_CHECKING:
    from ..main import Mnemosyne # 用于类型注解 self


async def list_collections_cmd_impl(self: "Mnemosyne", event: AstrMessageEvent) -> AsyncGenerator[None, AstrMessageEvent]:
    """
    [命令实现] 列出当前 Milvus 实例中的所有集合。

    Args:
        self ("Mnemosyne"): Mnemosyne 插件实例。
        event (AstrMessageEvent): 命令触发的消息事件对象。

    Yields:
        AstrMessageEvent: 包含命令结果 (集合列表或错误信息) 的纯文本消息事件。
    """
    # 检查 Milvus 管理器是否已初始化并连接
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或连接失败，无法列出集合。")
        return
    try:
        # 获取集合列表
        collections: Optional[List[str]] = self.milvus_manager.list_collections()

        if collections is None: # 如果 Milvus 操作返回 None，表示获取失败
            yield event.plain_result("⚠️ 获取 Milvus 集合列表失败，请检查插件日志以获取详细信息。")
            return

        if not collections: # 如果列表为空
            response_text = "当前 Milvus 实例中没有找到任何集合。"
        else:
            # 构建响应文本
            response_text = "当前 Milvus 实例中的集合列表：\n" + "\n".join(
                [f"📚 {col_name}" for col_name in collections] # 使用表情符号增强可读性
            )
            # 检查当前插件配置的集合是否存在于列表中
            if self.collection_name in collections:
                response_text += f"\n\nℹ️ 当前插件正在使用的集合: `{self.collection_name}`"
            else:
                response_text += (
                    f"\n\n⚠️ 警告：当前插件配置使用的集合 `{self.collection_name}` 不在上述列表中！插件可能无法正常工作。"
                )
        yield event.plain_result(response_text)

    except Exception as e:
        # 捕获任何在获取过程中发生的其他异常
        self.logger.error(f"执行 'memory list' 命令时发生意外错误: {str(e)}", exc_info=True)
        yield event.plain_result(f"⚠️ 获取集合列表时发生内部错误: {str(e)}")


async def delete_collection_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    collection_name: str,
    confirm: Optional[str] = None,
) -> AsyncGenerator[None, AstrMessageEvent]:
    """
    [命令实现] 删除指定的 Milvus 集合及其所有数据。
    此操作具有危险性，需要用户二次确认。

    Args:
        self ("Mnemosyne"): Mnemosyne 插件实例。
        event (AstrMessageEvent): 命令触发的消息事件对象。
        collection_name (str): 要删除的 Milvus 集合的名称。
        confirm (Optional[str]): 确认参数，必须为 `CONFIRM_FLAG` ("--confirm") 才执行删除。

    Yields:
        AstrMessageEvent: 包含操作结果 (成功信息、警告或错误信息) 的纯文本消息事件。
    """
    # 检查 Milvus 服务状态
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或连接失败，无法删除集合。")
        return

    # 检查要删除的集合是否是当前插件正在使用的集合
    is_current_collection: bool = (collection_name == self.collection_name)
    warning_msg: str = ""
    if is_current_collection:
        warning_msg = (
            f"\n\n🔥🔥🔥 **严重警告**：您正在尝试删除当前插件正在使用的集合 (`{collection_name}`)！"
            "此操作将导致插件核心功能立即失效，直到集合被手动重建或插件配置更改为指向其他有效集合！ 🔥🔥🔥"
        )

    # 检查确认标志
    if confirm != CONFIRM_FLAG: # CONFIRM_FLAG 是 "--confirm"
        # 提示用户进行确认
        confirmation_request_msg = (
            f"⚠️ **操作确认** ⚠️\n"
            f"您请求删除 Milvus 集合 `{collection_name}` 及其包含的所有数据。\n"
            f"**此操作不可撤销，数据将永久丢失！**\n"
            f"{warning_msg}\n\n" # 如果是当前集合，显示严重警告
            f"如果您确定要继续，请再次执行命令并添加 `{CONFIRM_FLAG}` 参数，例如：\n"
            f"`/memory drop_collection {collection_name} {CONFIRM_FLAG}`"
        )
        yield event.plain_result(confirmation_request_msg)
        return

    try:
        sender_id: str = event.get_sender_id() # 获取操作者ID，用于日志记录
        self.logger.warning(
            f"管理员 {sender_id} 请求删除 Milvus 集合: `{collection_name}` (已确认执行)。"
        )
        if is_current_collection:
            self.logger.critical( # 使用更高级别的日志记录严重操作
                f"管理员 {sender_id} 正在删除当前插件使用的核心集合 `{collection_name}`！相关功能将立即中断。"
            )

        # 执行删除操作
        success: bool = self.milvus_manager.drop_collection(collection_name)

        if success:
            response_msg = f"✅ 已成功删除 Milvus 集合 `{collection_name}`。"
            if is_current_collection:
                response_msg += "\n‼️ **重要提示**：插件当前使用的集合已被删除，相关记忆功能将无法使用，请尽快处理！"
            yield event.plain_result(response_msg)
            self.logger.info(f"管理员 {sender_id} 成功删除了 Milvus 集合: `{collection_name}`。")
        else:
            # Milvus 返回删除失败，但未抛出异常 (可能由 MilvusManager 内部处理)
            yield event.plain_result(
                f"⚠️ 删除集合 `{collection_name}` 的请求已发送，但 Milvus 返回操作失败。请检查 Milvus 服务端日志以获取详细信息。"
            )

    except Exception as e:
        self.logger.error(
            f"执行 'memory drop_collection {collection_name}' 命令时发生严重错误: {str(e)}",
            exc_info=True, # 记录完整异常堆栈
        )
        yield event.plain_result(f"⚠️ 删除集合 `{collection_name}` 时发生严重内部错误: {str(e)}")


async def list_records_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    collection_name: Optional[str] = None,
    limit: int = DEFAULT_LIST_RECORDS_LIMIT, # 使用常量作为默认值
) -> AsyncGenerator[None, AstrMessageEvent]:
    """
    [命令实现] 查询指定集合的记忆记录，按创建时间倒序显示最新的记录。

    Args:
        self ("Mnemosyne"): Mnemosyne 插件实例。
        event (AstrMessageEvent): 命令触发的消息事件对象。
        collection_name (Optional[str]): 要查询的集合名称。如果为 None，则查询当前插件配置的集合。
        limit (int): 要显示的记录数量。默认为 `DEFAULT_LIST_RECORDS_LIMIT`。

    Yields:
        AstrMessageEvent: 包含查询结果 (记录列表或错误/提示信息) 的纯文本消息事件。
    """
    # 检查 Milvus 服务状态
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或连接失败，无法查询记录。")
        return

    # 确定目标集合名称
    target_collection: str = collection_name or self.collection_name

    # 验证用户输入的 limit 参数
    if not (0 < limit <= MAX_LIST_RECORDS_LIMIT): # MAX_LIST_RECORDS_LIMIT 是 50
        yield event.plain_result(f"⚠️ 显示数量 (limit) 必须在 1 到 {MAX_LIST_RECORDS_LIMIT} 之间。")
        return

    try:
        # 检查目标集合是否存在
        if not self.milvus_manager.has_collection(target_collection):
            yield event.plain_result(f"⚠️ 集合 `{target_collection}` 不存在。")
            return

        # 获取当前会话 ID，用于可能的过滤
        # 注意：此命令默认显示所有会话的记录，除非将来进行修改以支持按当前会话过滤
        # session_id_filter: Optional[str] = await self.context.conversation_manager.get_curr_conversation_id(
        #     event.unified_msg_origin
        # )
        # 目前，我们查询所有记录，然后按时间排序。如果需要按特定会话过滤，表达式需要修改。
        # expr = f'{PRIMARY_FIELD_NAME} >= 0' # 查询所有实体
        # 如果要按当前会话过滤，则：
        current_session_id: Optional[str] = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
        expr_parts = [f"{PRIMARY_FIELD_NAME} >= 0"] # 基础表达式，确保获取所有记录
        query_description = f"集合 `{target_collection}`"

        # 暂时不默认按当前会话过滤，因为命令设计是列出记录，可能是管理员查看所有
        # if current_session_id:
        # expr_parts.append(f'session_id == "{current_session_id}"')
        # query_description += f" (当前会话: {current_session_id})"

        expr = " and ".join(expr_parts)
        self.logger.info(
            f"查询 {query_description} 中的所有相关记录 (Milvus 拉取上限 {MAX_TOTAL_FETCH_RECORDS} 条) 以便排序。"
        )

        # 定义需要从 Milvus 获取的字段，使用常量 DEFAULT_OUTPUT_FIELDS
        # DEFAULT_OUTPUT_FIELDS 应该包含 "content", "create_time", "session_id", "personality_id", PRIMARY_FIELD_NAME
        output_fields_to_fetch: List[str] = DEFAULT_OUTPUT_FIELDS

        self.logger.debug(
            f"准备查询 Milvus: 集合='{target_collection}', 表达式='{expr}', 输出字段={output_fields_to_fetch}, Milvus限制={MAX_TOTAL_FETCH_RECORDS}"
        )

        # 从 Milvus 获取记录，使用较大的上限 (MAX_TOTAL_FETCH_RECORDS) 以便后续排序
        fetched_records: Optional[List[Dict[str, Any]]] = self.milvus_manager.query(
            collection_name=target_collection,
            expression=expr,
            output_fields=output_fields_to_fetch,
            limit=MAX_TOTAL_FETCH_RECORDS, # 拉取足够多的数据进行准确排序
        )

        # 检查查询结果是否有效
        if fetched_records is None: # MilvusManager.query 在失败时返回 None
            self.logger.error(
                f"查询集合 `{target_collection}` 失败，MilvusManager.query 返回 None。"
            )
            yield event.plain_result(
                f"⚠️ 查询集合 `{target_collection}` 的记录失败，请检查插件日志。"
            )
            return

        if not fetched_records: # 列表为空
            yield event.plain_result(
                f"ℹ️ 集合 `{target_collection}` 中没有找到任何记忆记录。"
            )
            return

        # 提示用户获取到的记录数量，如果达到了 Milvus 拉取上限
        if len(fetched_records) >= MAX_TOTAL_FETCH_RECORDS:
            self.logger.warning(
                f"查询到的记录数量已达到 Milvus 拉取上限 ({MAX_TOTAL_FETCH_RECORDS})。"
                "如果总记录数超过此上限，排序结果可能仅基于部分最新数据。"
            )
            yield event.plain_result(
                f"ℹ️ 注意：已获取 {MAX_TOTAL_FETCH_RECORDS} 条记录进行排序。如果总记录数远超此数量，显示的“最新”记录可能并非全局最新。"
            )

        self.logger.debug(f"成功从 Milvus 获取到 {len(fetched_records)} 条原始记录用于排序。")

        # 在获取到的记录中按 `create_time` 降序排序，以找到最新的记录
        try:
            # 使用 lambda 获取 'create_time'，如果不存在或为 None，则默认为 0 (早期记录在前)
            fetched_records.sort(
                key=lambda record: record.get("create_time", 0) or 0, reverse=True
            )
            self.logger.debug(
                f"已将获取到的 {len(fetched_records)} 条记录按 `create_time` 降序排序。"
            )
        except Exception as sort_e:
            self.logger.warning(
                f"对查询结果进行排序时出错: {sort_e}。记录显示顺序可能不按时间排序。", exc_info=True
            )
            # 如果排序失败，仍然继续处理，但顺序可能不理想

        # 从排序后的结果中取出用户请求的 `limit` 数量的记录
        records_to_display: List[Dict[str, Any]] = fetched_records[:limit]

        # 准备响应消息
        total_fetched_count: int = len(fetched_records)
        display_count: int = len(records_to_display)
        response_lines: List[str] = [
            f"📜 集合 `{target_collection}` 的最新记忆记录 (共分析 {total_fetched_count} 条, 显示最新的 {display_count} 条):"
        ]

        # 格式化每条记录以供显示
        for i, record_data in enumerate(records_to_display, start=1):
            create_timestamp: Optional[float] = record_data.get("create_time")
            time_str: str
            try:
                time_str = (
                    datetime.fromtimestamp(create_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    if create_timestamp is not None
                    else "未知时间"
                )
            except (TypeError, ValueError, OSError) as time_e:
                self.logger.warning(
                    f"记录 ID `{record_data.get(PRIMARY_FIELD_NAME, '未知ID')}` 的时间戳 '{create_timestamp}' 无效或解析错误: {time_e}"
                )
                time_str = f"无效时间戳({create_timestamp})" if create_timestamp is not None else "未知时间"

            content: str = record_data.get("content", "内容不可用")
            # CONTENT_PREVIEW_MAX_LENGTH 是 200
            content_preview: str = content[:CONTENT_PREVIEW_MAX_LENGTH] + ("..." if len(content) > CONTENT_PREVIEW_MAX_LENGTH else "")
            record_session_id: str = record_data.get("session_id", "未知会话")
            persona_id_val: str = record_data.get("personality_id", "未知人格") # Renamed to avoid conflict
            pk_val: Any = record_data.get(PRIMARY_FIELD_NAME, "未知ID")

            response_lines.append(
                f"#{i} [ID: {pk_val}]\n"
                f"  📅 时间: {time_str}\n"
                f"  👤 人格: {persona_id_val}\n"
                f"  💬 会话: {record_session_id}\n"
                f"  📝 内容: {content_preview}"
            )

        yield event.plain_result("\n\n".join(response_lines))

    except Exception as e:
        self.logger.error(
            f"执行 'memory list_records' 命令时发生意外错误 (集合: {target_collection}): {str(e)}",
            exc_info=True,
        )
        yield event.plain_result(f"⚠️ 查询记忆记录时发生内部错误，请联系管理员 ({type(e).__name__})。")


async def delete_session_memory_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    session_id: str, # 会话ID，现在是必须的
    confirm: Optional[str] = None,
) -> AsyncGenerator[None, AstrMessageEvent]:
    """
    [命令实现] 删除指定会话 ID 相关的所有记忆信息。
    此操作具有危险性，需要用户二次确认。

    Args:
        self ("Mnemosyne"): Mnemosyne 插件实例。
        event (AstrMessageEvent): 命令触发的消息事件对象。
        session_id (str): 要删除其记忆的会话 ID。
        confirm (Optional[str]): 确认参数，必须为 `CONFIRM_FLAG` ("--confirm") 才执行删除。

    Yields:
        AstrMessageEvent: 包含操作结果 (成功信息、警告或错误信息) 的纯文本消息事件。
    """
    # 检查 Milvus 服务状态
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或连接失败，无法删除会话记忆。")
        return

    # 校验 session_id 是否提供
    if not session_id or not session_id.strip():
        yield event.plain_result("⚠️ 请提供要删除记忆的具体会话 ID (session_id)。")
        return

    # 清理 session_id 输入 (去除可能的引号和首尾空格)
    session_id_to_delete: str = session_id.strip().strip('"`')

    # 检查确认标志
    if confirm != CONFIRM_FLAG: # CONFIRM_FLAG 是 "--confirm"
        confirmation_request_msg = (
            f"⚠️ **操作确认** ⚠️\n"
            f"您请求删除会话 ID `{session_id_to_delete}` 在当前插件使用的集合 (`{self.collection_name}`) 中的所有记忆信息！\n"
            f"**此操作不可撤销，相关会话的记忆将永久丢失！**\n\n"
            f"要确认删除，请再次执行命令并添加 `{CONFIRM_FLAG}` 参数，例如：\n"
            f'`/memory delete_session_memory "{session_id_to_delete}" {CONFIRM_FLAG}`'
        )
        yield event.plain_result(confirmation_request_msg)
        return

    try:
        target_collection_name: str = self.collection_name # 操作目标是当前插件配置的集合
        # 构建用于 Milvus 删除操作的表达式
        delete_expression: str = f'session_id == "{session_id_to_delete}"'
        sender_id: str = event.get_sender_id()

        self.logger.warning(
            f"用户 {sender_id} 请求删除会话 `{session_id_to_delete}` 的所有记忆 "
            f"(目标集合: `{target_collection_name}`, Milvus表达式: `{delete_expression}`) (已确认执行)。"
        )

        # 执行删除操作
        mutation_result: Optional[Any] = self.milvus_manager.delete(
            collection_name=target_collection_name, expression=delete_expression
        )

        if mutation_result:
            # Milvus 的 delete 操作返回的 delete_count 可能不总是准确反映实际删除数量，
            # 特别是对于复杂表达式或在数据未完全 flush 的情况下。
            # 它更多表示匹配表达式并被标记为删除的实体数量。
            deleted_count_info: str = (
                str(mutation_result.delete_count)
                if hasattr(mutation_result, "delete_count") and mutation_result.delete_count is not None
                else "未知数量"
            )
            self.logger.info(
                f"已向 Milvus 发送删除会话 `{session_id_to_delete}` 记忆的请求。操作影响的实体数 (可能不精确): {deleted_count_info}。"
            )

            # 为了确保删除操作立即生效并对后续查询可见，执行 flush 操作
            try:
                self.logger.info(
                    f"正在刷新 (Flush) 集合 `{target_collection_name}` 以确保删除操作完全应用..."
                )
                self.milvus_manager.flush([target_collection_name])
                self.logger.info(f"集合 `{target_collection_name}` 刷新完成。删除操作已生效。")
                yield event.plain_result(
                    f"✅ 已成功删除会话 ID `{session_id_to_delete}` 的所有记忆信息。"
                )
            except Exception as flush_err:
                self.logger.error(
                    f"刷新集合 `{target_collection_name}` 以应用删除时发生错误: {flush_err}",
                    exc_info=True,
                )
                yield event.plain_result(
                    f"⚠️ 已发送删除请求 (影响实体数: {deleted_count_info})，但在刷新集合使更改生效时遇到错误: {str(flush_err)}。\n"
                    "删除的记忆可能在一段时间后才完全不可见。"
                )
        else:
            # milvus_manager.delete 返回 None 表示操作失败
            yield event.plain_result(
                f"⚠️ 删除会话 ID `{session_id_to_delete}` 的记忆请求失败。请检查插件日志和 Milvus 服务状态。"
            )

    except Exception as e:
        self.logger.error(
            f"执行 'memory delete_session_memory' 命令时发生严重错误 (Session ID: `{session_id_to_delete}`): {str(e)}",
            exc_info=True,
        )
        yield event.plain_result(f"⚠️ 删除会话记忆时发生严重内部错误: {str(e)}")


async def get_session_id_cmd_impl(self: "Mnemosyne", event: AstrMessageEvent) -> AsyncGenerator[None, AstrMessageEvent]:
    """
    [命令实现] 获取当前与用户对话的会话 ID。

    Args:
        self ("Mnemosyne"): Mnemosyne 插件实例。
        event (AstrMessageEvent): 命令触发的消息事件对象。

    Yields:
        AstrMessageEvent: 包含当前会话 ID 或提示信息的纯文本消息事件。
    """
    try:
        # 从 AstrBot 的会话管理器获取当前会话 ID
        current_session_id: Optional[str] = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin # 使用统一消息来源确保跨平台兼容性
        )

        if current_session_id:
            yield event.plain_result(f"当前会话 ID: `{current_session_id}`")
        else:
            # 如果无法获取会话 ID (例如，在没有会话上下文的情况下调用此命令)
            yield event.plain_result(
                "🤔 无法获取当前会话 ID。可能您还没有开始与我对话，或者当前上下文不涉及特定会话。"
            )
            self.logger.warning(
                f"用户 {event.get_sender_id()} 在来源 {event.unified_msg_origin} 尝试获取 session_id，但未能成功获取。"
            )
    except Exception as e:
        self.logger.error(
            f"执行 'memory get_session_id' 命令时发生意外错误: {str(e)}", exc_info=True
        )
        yield event.plain_result(f"⚠️ 获取当前会话 ID 时发生内部错误: {str(e)}")
