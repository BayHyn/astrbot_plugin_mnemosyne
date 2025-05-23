# -*- coding: utf-8 -*-
"""
Mnemosyne 插件的工具函数集合。
这些函数提供了一些通用的辅助功能，例如地址解析、标签移除、上下文格式化等。
"""

from urllib.parse import urlparse # 用于解析 URL
from astrbot.api.event import AstrMessageEvent # AstrBot 消息事件类型
import functools # 用于创建装饰器
import re # 正则表达式操作
from typing import List, Dict, Set, Union, Tuple, Callable, Any # 类型注解

# 从常量模块导入所需常量
from .constants import (
    DEFAULT_MILVUS_PORT, MNEMOSYNE_TAG_REGEX_PATTERN, ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM,
    DEFAULT_ADDRESS_PROTOCOL
)

# 注意：此文件目前没有自己的日志记录器。如果需要，可以添加：
# from astrbot.core.log import LogManager
# logger = LogManager.GetLogger(log_name=TOOLS_LOG_NAME) # TOOLS_LOG_NAME 需在 constants.py 定义


def parse_address(address: str) -> Tuple[Optional[str], int]:
    """
    解析给定的网络地址字符串，提取主机名和端口号。
    如果地址字符串中没有指定协议 (如 "http://")，则默认添加 "http://" 进行解析。
    如果未指定端口号，则使用 Milvus 的默认端口 `DEFAULT_MILVUS_PORT` (19530)。

    Args:
        address (str): 要解析的地址字符串，可以是 "host:port" 或 "protocol://host:port" 格式。

    Returns:
        Tuple[Optional[str], int]: 一个包含主机名 (str 或 None) 和端口号 (int) 的元组。
                                    如果无法解析主机名，则主机名部分可能为 None。
    """
    # 如果地址没有明确的协议前缀，则添加默认协议 (DEFAULT_ADDRESS_PROTOCOL) 以便 urlparse 正确工作
    if not (address.startswith("http://") or address.startswith("https://")):
        processed_address: str = DEFAULT_ADDRESS_PROTOCOL + address
    else:
        processed_address = address

    parsed_url = urlparse(processed_address) # 解析处理后的地址
    host: Optional[str] = parsed_url.hostname # 提取主机名
    # 提取端口号，如果不存在则使用默认端口 (DEFAULT_MILVUS_PORT)
    port: int = parsed_url.port if parsed_url.port is not None else DEFAULT_MILVUS_PORT
    return host, port


def content_to_str(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    [装饰器] 一个简单的装饰器示例，用于将目标函数的所有位置参数和关键字参数转换为字符串，
    并在调用实际函数前打印这些转换后的参数。主要用于调试。

    Args:
        func (Callable[..., Any]): 被装饰的函数。

    Returns:
        Callable[..., Any]: 包装后的函数。
    """
    @functools.wraps(func) # 保留被装饰函数的元信息 (如名称、文档字符串)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 将所有位置参数转换为字符串
        str_args: List[str] = [str(arg) for arg in args]
        # 将所有关键字参数的值转换为字符串
        str_kwargs: Dict[str, str] = {k: str(v) for k, v in kwargs.items()}
        # 打印转换后的参数信息 (用于调试)
        print(
            f"函数 '{func.__name__}' 被调用，参数 (已转换为字符串): args={str_args}, kwargs={str_kwargs}"
        )
        # 调用原始函数，并传入转换后的参数
        return func(*str_args, **str_kwargs)
    return wrapper


def remove_mnemosyne_tags(
    contents: List[Dict[str, Any]], # 更改为 Any 以处理潜在的 list 类型的 content
    contexts_memory_len: int = 0
) -> List[Dict[str, Any]]:
    """
    使用正则表达式移除用户消息内容 (`role`="user") 中的 Mnemosyne 记忆标签 (`<Mnemosyne>...</Mnemosyne>`)。
    可以配置保留最新的 N 个记忆标签对。

    Args:
        contents (List[Dict[str, Any]]): 包含聊天记录的列表。
                                            每个元素是一个字典，通常包含 "role" 和 "content" 键。
                                            "content" 键的值可能是字符串或（根据TODO）列表。
        contexts_memory_len (int, optional): 需要保留的最新的 `<Mnemosyne>` 标签对数量。
                                             如果 `<= 0`，则移除所有匹配的标签对。默认为 0。

    Returns:
        List[Dict[str, Any]]: 清理或部分清理了 `<Mnemosyne>` 标签对的聊天记录列表。
    """
    # MNEMOSYNE_TAG_REGEX_PATTERN 是 r"<Mnemosyne>.*?</Mnemosyne>"
    # re.DOTALL 使得 '.' 可以匹配包括换行符在内的任意字符
    compiled_regex = re.compile(MNEMOSYNE_TAG_REGEX_PATTERN, re.DOTALL)
    cleaned_contents: List[Dict[str, Any]] = [] # 用于存储处理后的聊天记录

    if contexts_memory_len <= 0:
        # --- 情况1: 移除所有 Mnemosyne 标签 ---
        for content_item in contents:
            # 仅处理用户角色 (ROLE_USER) 的消息内容
            if isinstance(content_item, dict) and content_item.get("role") == ROLE_USER:
                original_text: Any = content_item.get("content", "") # 获取原始文本内容
                # 确保内容是字符串类型再进行正则替换
                if isinstance(original_text, str):
                    cleaned_text: str = compiled_regex.sub("", original_text) # 替换为空字符串
                    cleaned_contents.append({"role": ROLE_USER, "content": cleaned_text})
                else: # 如果内容不是字符串 (例如，可能是列表或其他类型)，则原样保留
                    cleaned_contents.append({"role": ROLE_USER, "content": original_text}) # 保留原始类型
            else: # 非用户角色的消息，或格式不符的消息，直接保留
                cleaned_contents.append(content_item)
    else:
        # --- 情况2: 保留最新的 N 个 Mnemosyne 标签 ---
        # 步骤1: 从所有用户消息中提取出所有的 Mnemosyne 标签块
        all_mnemosyne_blocks: List[str] = []
        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == ROLE_USER:
                original_text = content_item.get("content", "")
                if isinstance(original_text, str): # 仅处理字符串内容
                    found_blocks: List[str] = compiled_regex.findall(original_text)
                    all_mnemosyne_blocks.extend(found_blocks)

        # 步骤2: 确定哪些标签块需要被保留 (最新的 N 个)
        blocks_to_keep: Set[str] = set(all_mnemosyne_blocks[-contexts_memory_len:])

        # 步骤3: 定义一个替换函数，用于 re.sub
        #         如果匹配到的块在 blocks_to_keep 集合中，则保留它，否则替换为空字符串
        def replace_logic(match: re.Match) -> str:
            block: str = match.group(0) # 获取匹配到的整个标签块
            return block if block in blocks_to_keep else ""

        # 步骤4: 再次遍历聊天记录，对用户消息内容应用上述替换逻辑
        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == ROLE_USER:
                original_text = content_item.get("content", "")
                if isinstance(original_text, str): # 仅处理字符串内容
                    # 仅当文本中可能包含标签时才执行替换，以提高效率
                    if compiled_regex.search(original_text):
                        cleaned_text = compiled_regex.sub(replace_logic, original_text)
                        cleaned_contents.append({"role": ROLE_USER, "content": cleaned_text})
                    else: # 如果文本中没有标签，则直接添加原始内容
                        cleaned_contents.append(content_item)
                else: # 非字符串内容，原样保留
                     cleaned_contents.append({"role": ROLE_USER, "content": original_text})
            else: # 非用户角色的消息，直接保留
                cleaned_contents.append(content_item)
    return cleaned_contents


def remove_system_mnemosyne_tags(text: str, contexts_memory_len: int = 0) -> str:
    """
    使用正则表达式移除系统提示字符串中的 Mnemosyne 记忆标签 (`<Mnemosyne>...</Mnemosyne>`)。
    可以配置保留最新的 N 个记忆标签对。

    Args:
        text (str): 系统提示字符串。
        contexts_memory_len (int, optional): 需要保留的最新的 `<Mnemosyne>` 标签对数量。
                                             如果 `<= 0`，则移除所有匹配的标签对。默认为 0。

    Returns:
        str: 清理或部分清理了 `<Mnemosyne>` 标签对的系统提示字符串。
             如果输入 `text` 不是字符串，则原样返回。
    """
    if not isinstance(text, str): # 类型检查，确保输入是字符串
        return text

    # MNEMOSYNE_TAG_REGEX_PATTERN 是 r"<Mnemosyne>.*?</Mnemosyne>"
    compiled_regex = re.compile(MNEMOSYNE_TAG_REGEX_PATTERN, re.DOTALL)
    cleaned_text: str

    if contexts_memory_len <= 0:
        # --- 情况1: 移除所有 Mnemosyne 标签 ---
        cleaned_text = compiled_regex.sub("", text) # 替换为空字符串
    else:
        # --- 情况2: 保留最新的 N 个 Mnemosyne 标签 ---
        all_mnemosyne_blocks: List[str] = compiled_regex.findall(text) # 提取所有标签块
        blocks_to_keep: Set[str] = set(all_mnemosyne_blocks[-contexts_memory_len:]) # 确定要保留的块

        # 定义替换逻辑
        def replace_logic(match: re.Match) -> str:
            block: str = match.group(0)
            return block if block in blocks_to_keep else ""

        # 应用替换逻辑 (仅当文本中可能包含标签时)
        if compiled_regex.search(text):
            cleaned_text = compiled_regex.sub(replace_logic, text)
        else: # 如果文本中没有标签，则无需处理
            cleaned_text = text
    return cleaned_text


def remove_system_content(
    contents: List[Dict[str, str]], contexts_memory_len: int = 0
) -> List[Dict[str, str]]:
    """
    从 LLM 的上下文历史记录中移除较旧的系统角色消息 (`role`="system")。
    此函数会保留指定数量 (`contexts_memory_len`) 的最新系统消息，并维持整体消息顺序。

    Args:
        contents (List[Dict[str, str]]): 包含聊天记录的列表。
                                           每个元素是一个字典，应包含 "role" 和 "content" 键。
        contexts_memory_len (int, optional): 需要保留的最新的系统消息的数量。
                                             必须是非负整数。如果 `< 0`，则视为 0。默认为 0 (移除所有系统消息)。

    Returns:
        List[Dict[str, str]]: 处理后的聊天记录列表，其中较旧的系统消息已被移除。
    """
    if not isinstance(contents, list): # 防御性检查，确保输入是列表
        return []
    if contexts_memory_len < 0: # 确保 contexts_memory_len 不为负
        contexts_memory_len = 0

    # 1. 找到所有系统角色消息 (`role`="system") 的索引位置
    # ROLE_SYSTEM 是 "system"
    system_message_indices: List[int] = [
        i for i, msg_item in enumerate(contents)
        if isinstance(msg_item, dict) and msg_item.get("role") == ROLE_SYSTEM
    ]

    # 2. 根据需要保留的数量，确定哪些系统消息需要被移除
    indices_to_remove: Set[int] = set() # 使用集合存储待移除消息的索引，以提高查找效率
    num_system_messages_found: int = len(system_message_indices)

    if num_system_messages_found > contexts_memory_len:
        # 计算需要移除的旧系统消息的数量
        num_to_remove: int = num_system_messages_found - contexts_memory_len
        # 将最早的 (num_to_remove) 条系统消息的索引添加到待移除集合中
        indices_to_remove.update(system_message_indices[:num_to_remove])

    # 3. 构建新的聊天记录列表，跳过那些被标记为待移除的系统消息
    cleaned_contents: List[Dict[str, str]] = [
        msg_item for i, msg_item in enumerate(contents)
        if i not in indices_to_remove # 只保留不在待移除集合中的消息
    ]
    return cleaned_contents


def format_context_to_string(
    context_history: List[Union[Dict[str, str], str]], length: int = 10
) -> str:
    """
    从给定的上下文历史记录中提取最新的 `length` 条用户 (`user`) 和助手 (`assistant`) 的对话消息，
    并将它们的内容格式化为一个单一的、用换行符分隔的字符串。
    其他角色的消息（如 `system`）将被忽略，并且不计入 `length` 的限制。

    Args:
        context_history (List[Union[Dict[str, str], str]]): 上下文历史消息列表。
            列表中的每个元素可以是一个包含 "role" 和 "content"键的字典，或者是一个简单的字符串（将被忽略）。
        length (int, optional): 需要提取的用户或助手对话消息的总数量。默认为 10。
                                如果 `length <= 0`，则返回空字符串。

    Returns:
        str: 格式化后的字符串。包含了最新的 `length` 条用户和助手消息的内容，
             每条消息前缀以 "角色:" (例如, "user:", "assistant:")，并以换行符分隔。
             消息按其在原始历史记录中的顺序（从旧到新）排列。
    """
    if length <= 0: # 如果请求的长度无效，则返回空字符串
        return ""

    selected_message_contents: List[str] = [] # 用于存储符合条件的消息内容
    messages_counted: int = 0 # 已选取的有效消息计数

    # 从后往前遍历历史记录，以优先获取最新的消息
    for message_item in reversed(context_history):
        role: Optional[str] = None
        content: Optional[str] = None

        # 检查消息是否为符合预期的字典格式
        if isinstance(message_item, dict) and "role" in message_item and "content" in message_item:
            role = message_item.get("role")
            content = message_item.get("content")

        # 如果消息内容有效，并且角色是用户 (ROLE_USER) 或助手 (ROLE_ASSISTANT)
        if content is not None and role in [ROLE_USER, ROLE_ASSISTANT]:
            # 格式化消息并添加到列表的开头 (因为我们是反向遍历，所以用 insert(0, ...) 来维持原始顺序)
            # ROLE_USER 是 "user", ROLE_ASSISTANT 是 "assistant"
            selected_message_contents.insert(0, f"{role}: {str(content)}") # 确保 content 是字符串
            messages_counted += 1
            if messages_counted >= length: # 如果已达到所需数量，则停止处理
                break
    
    # 使用换行符将收集到的消息内容连接成一个字符串
    return "\n".join(selected_message_contents)


def is_group_chat(event: AstrMessageEvent) -> bool:
    """
    判断给定的 `AstrMessageEvent` 是否表示一个群聊消息。

    Args:
        event (AstrMessageEvent): AstrBot 的消息事件对象。

    Returns:
        bool: 如果消息来自群聊 (即 `get_group_id()` 返回非空字符串)，则返回 `True`；
              否则 (例如私聊或频道消息，取决于具体平台实现) 返回 `False`。
    """
    # AstrMessageEvent.get_group_id() 通常在群聊时返回群ID，私聊时返回空字符串或特定值
    return event.get_group_id() != ""
