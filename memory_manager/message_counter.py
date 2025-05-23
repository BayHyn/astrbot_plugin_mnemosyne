import sqlite3 # SQLite 数据库接口
import os # 操作系统接口，用于路径操作
from typing import Optional, List, Any # 类型注解

from astrbot.core.log import LogManager # AstrBot 日志管理器
# 从常量模块导入相关常量
from ..core.constants import (
    MESSAGE_COUNTER_LOG_NAME, DB_DEFAULT_DIR_NAME, DB_DEFAULT_MESSAGE_COUNTS_FILENAME,
    DB_TABLE_MESSAGE_COUNTS, DB_TABLE_SESSION_SUMMARY_TIMES, DB_COLUMN_SESSION_ID,
    DB_COLUMN_COUNT, DB_COLUMN_LAST_SUMMARY_TIMESTAMP
)

# 获取日志记录器，使用常量中定义的名称
logger = LogManager.GetLogger(log_name=MESSAGE_COUNTER_LOG_NAME)


class MessageCounter:
    """
    消息计数器类。
    使用 SQLite 数据库持久化存储每个会话的消息轮次计数以及上次总结的时间戳。
    """

    def __init__(self, db_file: Optional[str] = None) -> None:
        """
        初始化消息计数器。
        如果未提供 `db_file` 参数，则会自动在项目特定数据目录下生成数据库文件路径。

        Args:
            db_file (Optional[str], optional): SQLite 数据库文件的完整路径。
                                              如果为 `None`，则使用默认路径。默认为 `None`。
        """
        if db_file is None:
            # --- 自动确定数据库文件路径 ---
            # 获取当前脚本文件 (__file__) 所在的目录
            current_file_dir: str = os.path.dirname(os.path.abspath(__file__))

            # 从当前文件目录向上追溯三层，期望到达插件的根目录或项目的一个标准位置
            base_dir: str = current_file_dir
            for _ in range(3): # 向上回溯3层: memory_manager -> astrbot_plugin_mnemosyne -> (通常是 plugins 或项目根目录)
                parent_dir: str = os.path.dirname(base_dir)
                if parent_dir == base_dir: # 如果已到达文件系统根目录，则停止
                    break
                base_dir = parent_dir
            
            # DB_DEFAULT_DIR_NAME 是 "mnemosyne_data"
            # DB_DEFAULT_MESSAGE_COUNTS_FILENAME 是 "message_counters.db"
            # 在基目录下构建数据存储目录 (例如 "mnemosyne_data")
            data_storage_dir: str = os.path.join(base_dir, DB_DEFAULT_DIR_NAME)

            # 确保数据存储目录存在，如果不存在则创建它
            # exist_ok=True 表示如果目录已存在，os.makedirs 不会抛出异常
            os.makedirs(data_storage_dir, exist_ok=True)

            # 最终的数据库文件路径
            self.db_file: str = os.path.join(data_storage_dir, DB_DEFAULT_MESSAGE_COUNTS_FILENAME)
            logger.info(f"未提供数据库文件路径，将使用自动生成的默认路径: '{self.db_file}'")
        else:
            # 如果用户显式提供了 db_file，则直接使用该路径
            self.db_file = db_file
            logger.info(f"将使用用户提供的数据库文件路径: '{self.db_file}'")

        self._initialize_db() # 调用数据库初始化方法

    def _initialize_db(self) -> None:
        """
        初始化 SQLite 数据库。
        如果相关表 (`message_counts`, `session_summary_times`) 不存在，则创建它们。
        """
        conn: Optional[sqlite3.Connection] = None # 初始化数据库连接变量
        try:
            conn = sqlite3.connect(self.db_file) # 连接到 SQLite 数据库文件
            cursor: sqlite3.Cursor = conn.cursor() # 创建游标对象

            # --- 创建 message_counts 表 (如果不存在) ---
            # 用于存储每个会话的消息轮次计数
            # DB_TABLE_MESSAGE_COUNTS, DB_COLUMN_SESSION_ID, DB_COLUMN_COUNT 是常量
            create_message_counts_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE_MESSAGE_COUNTS} (
                    {DB_COLUMN_SESSION_ID} TEXT PRIMARY KEY,
                    {DB_COLUMN_COUNT} INTEGER NOT NULL DEFAULT 0
                )
            """
            cursor.execute(create_message_counts_table_sql)
            logger.debug(f"表 '{DB_TABLE_MESSAGE_COUNTS}' 初始化/验证完成。")

            # --- 创建 session_summary_times 表 (如果不存在) ---
            # 用于存储每个会话上次成功总结的时间戳
            # DB_TABLE_SESSION_SUMMARY_TIMES, DB_COLUMN_SESSION_ID, DB_COLUMN_LAST_SUMMARY_TIMESTAMP 是常量
            create_summary_times_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE_SESSION_SUMMARY_TIMES} (
                    {DB_COLUMN_SESSION_ID} TEXT PRIMARY KEY,
                    {DB_COLUMN_LAST_SUMMARY_TIMESTAMP} REAL NOT NULL
                )
            """
            cursor.execute(create_summary_times_table_sql)
            logger.debug(f"表 '{DB_TABLE_SESSION_SUMMARY_TIMES}' 初始化/验证完成。")

            conn.commit() # 提交事务，确保表结构更改生效
            logger.info(f"SQLite 数据库及相关表在 '{self.db_file}' 初始化成功。")
        except sqlite3.Error as e: # 捕获 SQLite 操作相关的错误
            logger.error(f"初始化 SQLite 数据库 '{self.db_file}' 时发生错误: {e}", exc_info=True)
            if conn:
                conn.rollback()  # 如果发生错误，回滚任何部分更改
        finally: # 确保数据库连接在操作完成或发生异常后都能关闭
            if conn:
                conn.close()

    # --- 上次总结时间戳 (session_summary_times table) 相关方法 ---

    def get_last_summary_time(self, session_id: str) -> Optional[float]:
        """
        从数据库检索指定会话 ID 的上次总结时间戳。

        Args:
            session_id (str): 需要查询的会话 ID。

        Returns:
            Optional[float]: 上次总结的 Unix 时间戳 (秒)。如果未找到记录或发生数据库错误，则返回 `None`。
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            # DB_TABLE_SESSION_SUMMARY_TIMES, DB_COLUMN_LAST_SUMMARY_TIMESTAMP, DB_COLUMN_SESSION_ID
            select_sql = f"SELECT {DB_COLUMN_LAST_SUMMARY_TIMESTAMP} FROM {DB_TABLE_SESSION_SUMMARY_TIMES} WHERE {DB_COLUMN_SESSION_ID} = ?"
            cursor.execute(select_sql, (session_id,))
            result: Optional[tuple] = cursor.fetchone() # 获取查询结果的第一行

            if result: # 如果查询到结果
                logger.debug(f"从数据库获取到会话 {session_id} 的上次总结时间戳: {result[0]}。")
                return float(result[0]) # 时间戳以 REAL 类型存储，转换为 float
            else: # 未找到该会话的记录
                logger.debug(f"数据库中未找到会话 {session_id} 的上次总结时间戳记录。")
                return None
        except sqlite3.Error as e:
            logger.error(f"获取会话 {session_id} 的上次总结时间戳时发生数据库错误: {e}", exc_info=True)
            return None # 发生错误时返回 None
        finally:
            if conn:
                conn.close()

    def update_last_summary_time(self, session_id: str, summary_time: float) -> None:
        """
        向数据库中插入或更新指定会话 ID 的上次总结时间戳。

        Args:
            session_id (str): 需要更新的会话 ID。
            summary_time (float): 新的总结时间戳 (Unix 时间戳，秒)。
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            # DB_TABLE_SESSION_SUMMARY_TIMES, DB_COLUMN_SESSION_ID, DB_COLUMN_LAST_SUMMARY_TIMESTAMP
            # 使用 INSERT OR REPLACE 语句，如果记录已存在则更新，否则插入新记录
            insert_or_replace_sql = f"""
                INSERT OR REPLACE INTO {DB_TABLE_SESSION_SUMMARY_TIMES} 
                ({DB_COLUMN_SESSION_ID}, {DB_COLUMN_LAST_SUMMARY_TIMESTAMP}) VALUES (?, ?)
            """
            cursor.execute(insert_or_replace_sql, (session_id, summary_time))
            conn.commit() # 提交事务
            logger.info(f"已成功将会话 {session_id} 的上次总结时间戳更新/插入为: {summary_time}。")
        except sqlite3.Error as e:
            logger.error(f"更新/插入会话 {session_id} 的上次总结时间戳时发生数据库错误: {e}", exc_info=True)
            if conn:
                conn.rollback() # 回滚事务
        finally:
            if conn:
                conn.close()

    # --- 消息轮次计数 (message_counts table) 相关方法 ---

    def reset_counter(self, session_id: str) -> None:
        """
        重置指定会话 ID 的消息计数器为 0。
        如果该会话尚不存在于计数表中，则会创建一条新记录。

        Args:
            session_id (str): 需要重置计数器的会话 ID。
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            # DB_TABLE_MESSAGE_COUNTS, DB_COLUMN_SESSION_ID, DB_COLUMN_COUNT
            # 使用 INSERT OR REPLACE 将计数设置为0，如果记录不存在则创建
            reset_sql = f"INSERT OR REPLACE INTO {DB_TABLE_MESSAGE_COUNTS} ({DB_COLUMN_SESSION_ID}, {DB_COLUMN_COUNT}) VALUES (?, ?)"
            cursor.execute(reset_sql, (session_id, 0))
            conn.commit()
            logger.debug(f"会话 {session_id} 的消息计数器已成功重置为 0。")
        except sqlite3.Error as e:
            logger.error(f"重置会话 {session_id} 的消息计数器时发生数据库错误: {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def increment_counter(self, session_id: str) -> None:
        """
        为指定会话 ID 的消息计数器增加 1。
        如果该会话尚不存在于计数表中，则会先创建一条初始值为 0 的记录，然后再增加。

        Args:
            session_id (str): 需要增加计数器的会话 ID。
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            # DB_TABLE_MESSAGE_COUNTS, DB_COLUMN_SESSION_ID, DB_COLUMN_COUNT
            # 步骤1: 确保记录存在，如果不存在则插入初始值为0的记录 (INSERT OR IGNORE)
            ensure_record_sql = f"INSERT OR IGNORE INTO {DB_TABLE_MESSAGE_COUNTS} ({DB_COLUMN_SESSION_ID}, {DB_COLUMN_COUNT}) VALUES (?, 0)"
            cursor.execute(ensure_record_sql, (session_id,))
            
            # 步骤2: 将计数器值加 1
            increment_sql = f"UPDATE {DB_TABLE_MESSAGE_COUNTS} SET {DB_COLUMN_COUNT} = {DB_COLUMN_COUNT} + 1 WHERE {DB_COLUMN_SESSION_ID} = ?"
            cursor.execute(increment_sql, (session_id,))
            conn.commit()
            logger.debug(f"会话 {session_id} 的消息计数器已成功增加 1。当前值: {self.get_counter(session_id)}。") # 日志中可选择性显示当前值
        except sqlite3.Error as e:
            logger.error(f"为会话 {session_id} 增加消息计数器时发生数据库错误: {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def get_counter(self, session_id: str) -> int:
        """
        获取指定会话 ID 的当前消息计数器值。

        Args:
            session_id (str): 需要查询计数器的会话 ID。

        Returns:
            int: 该会话 ID 对应的消息计数器值。如果会话 ID 不存在于表中或发生错误，则返回 0。
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            # DB_TABLE_MESSAGE_COUNTS, DB_COLUMN_COUNT, DB_COLUMN_SESSION_ID
            select_sql = f"SELECT {DB_COLUMN_COUNT} FROM {DB_TABLE_MESSAGE_COUNTS} WHERE {DB_COLUMN_SESSION_ID} = ?"
            cursor.execute(select_sql, (session_id,))
            result: Optional[tuple] = cursor.fetchone()

            if result: # 如果找到记录
                return int(result[0]) # 计数值存储为 INTEGER
            else: # 未找到记录
                logger.debug(f"数据库中未找到会话 {session_id} 的消息计数记录，返回默认值 0。")
                return 0
        except sqlite3.Error as e:
            logger.error(f"获取会话 {session_id} 的消息计数器时发生数据库错误: {e}", exc_info=True)
            return 0 # 发生错误时，保守返回 0
        finally:
            if conn:
                conn.close()

    def adjust_counter_if_necessary(self, session_id: str, context_history: List[Any]) -> bool:
        """
        检查给定会话的上下文历史记录长度是否小于其在数据库中存储的消息计数。
        如果小于（表示可能发生了数据不一致，例如外部修改了历史记录），则将数据库中的计数器调整为历史记录的实际长度。

        Args:
            session_id (str): 需要检查和调整计数器的会话 ID。
            context_history (List[Any]): 该会话的当前上下文历史记录列表。列表长度用于比较。

        Returns:
            bool: 如果计数器未调整（即历史长度不小于计数器值）或调整成功，则返回 `True`。
                  如果调整过程中发生数据库错误，则返回 `False`。
        """
        current_database_counter: int = self.get_counter(session_id)
        current_history_length: int = len(context_history)

        if current_history_length < current_database_counter:
            logger.warning(
                f"检测到不一致：会话 {session_id} 的上下文历史记录长度 ({current_history_length}) "
                f"小于数据库中记录的消息计数 ({current_database_counter})。将尝试调整数据库计数。"
            )
            conn: Optional[sqlite3.Connection] = None
            try:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                # DB_TABLE_MESSAGE_COUNTS, DB_COLUMN_COUNT, DB_COLUMN_SESSION_ID
                update_sql = f"UPDATE {DB_TABLE_MESSAGE_COUNTS} SET {DB_COLUMN_COUNT} = ? WHERE {DB_COLUMN_SESSION_ID} = ?"
                cursor.execute(update_sql, (current_history_length, session_id))
                conn.commit()
                logger.info(f"会话 {session_id} 的消息计数器已成功从 {current_database_counter} 调整为 {current_history_length}。")
                return True # 即使调整了，也视为操作成功（或已纠正）
            except sqlite3.Error as e:
                logger.error(f"调整会话 {session_id} 的消息计数器时发生数据库错误: {e}", exc_info=True)
                if conn:
                    conn.rollback()
                return False  # 调整失败，返回 False 以指示问题
            finally:
                if conn:
                    conn.close()
        else: # 历史长度大于或等于数据库计数，无需调整
            logger.debug(
                f"会话 {session_id} 的上下文历史记录长度 ({current_history_length}) "
                f"与消息计数器 ({current_database_counter}) 一致或更长，无需调整。"
            )
            return True
