# -*- coding: utf-8 -*-
"""
向量数据库工厂类
支持创建和管理不同类型的向量数据库实例
"""

import asyncio
from typing import Dict, Any, Optional, List
from astrbot.api import logger

from ...core.constants import VECTOR_FIELD_NAME
from ..vector_db_base import VectorDatabase
from .milvus_manager import MilvusManager
from .faiss_manager import FaissManager


class VectorDatabaseFactory:
    """
    向量数据库工厂类
    负责根据配置创建合适的向量数据库实例
    """

    @staticmethod
    def create_database(
        db_type: str, config: Dict[str, Any]
    ) -> Optional[VectorDatabase]:
        """
        创建向量数据库实例

        Args:
            db_type: 数据库类型 ("milvus" 或 "faiss")
            config: 数据库配置
            logger: 日志记录器

        Returns:
            VectorDatabase 实例或 None（如果创建失败）
        """

        try:
            db_type_lower = db_type.lower()

            if db_type_lower == "milvus":
                return VectorDatabaseFactory._create_milvus_database(config, logger)
            elif db_type_lower == "faiss":
                return VectorDatabaseFactory._create_faiss_database(config, logger)
            else:
                logger.error(f"Unsupported database type: {db_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to create {db_type} database: {e}", exc_info=True)
            return None

    @staticmethod
    def _create_milvus_database(
        config: Dict[str, Any], logger
    ) -> Optional[MilvusManager]:
        """创建 Milvus 数据库实例"""
        try:
            # 提取 Milvus 配置参数
            connect_args = {}

            # Milvus Lite 路径
            lite_path = config.get("milvus_lite_path", "")
            if lite_path:
                connect_args["lite_path"] = lite_path
                logger.info(f"Using Milvus Lite with path: {lite_path}")
            else:
                # 标准 Milvus 配置
                address = config.get("address", "")
                if address:
                    if address.startswith(("http://", "https://", "unix:")):
                        connect_args["uri"] = address
                    else:
                        # 解析 host:port
                        if ":" in address:
                            host, port = address.rsplit(":", 1)
                            connect_args["host"] = host
                            connect_args["port"] = port
                        else:
                            connect_args["host"] = address
                            connect_args["port"] = "19530"

                # 认证配置
                auth_config = config.get("authentication", {})
                if auth_config:
                    for key in ["user", "password", "token", "secure"]:
                        if key in auth_config and auth_config[key] is not None:
                            if key == "secure":
                                connect_args[key] = bool(auth_config[key])
                            else:
                                connect_args[key] = auth_config[key]

            # 数据库名称
            db_name = config.get("db_name", "default")
            if db_name != "default":
                connect_args["db_name"] = db_name

            # 连接别名
            collection_name = config.get("collection_name", "default")
            connect_args["alias"] = config.get(
                "connection_alias", f"mnemosyne_{collection_name}"
            )

            # 创建 MilvusManager 实例
            milvus_manager = MilvusManager(**connect_args)

            logger.info("Successfully created Milvus database instance")
            return milvus_manager

        except Exception as e:
            logger.error(f"Failed to create Milvus database: {e}", exc_info=True)
            return None

    @staticmethod
    def _create_faiss_database(
        config: Dict[str, Any], logger
    ) -> Optional[FaissManager]:
        """创建 FAISS 数据库实例"""
        try:
            # 提取 FAISS 配置参数
            faiss_config = config.get("faiss_config", {})
            data_path = faiss_config.get("faiss_data_path", "faiss_data")
            index_type = faiss_config.get("faiss_index_type", "IndexFlatL2")
            nlist = faiss_config.get("faiss_nlist", 100)

            # 创建 FaissManager 实例
            faiss_manager = FaissManager(
                data_path=data_path, index_type=index_type, nlist=nlist
            )

            logger.info(
                f"Successfully created FAISS database instance with path: {data_path}"
            )
            return faiss_manager

        except Exception as e:
            logger.error(f"Failed to create FAISS database: {e}", exc_info=True)
            return None

    @staticmethod
    def get_supported_databases() -> list[str]:
        """获取支持的数据库类型列表"""
        return ["milvus", "faiss"]

    @staticmethod
    def validate_config(db_type: str, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        验证数据库配置

        Args:
            db_type: 数据库类型
            config: 配置字典

        Returns:
            (is_valid, error_message) 元组
        """
        try:
            db_type_lower = db_type.lower()

            if db_type_lower == "milvus":
                return VectorDatabaseFactory._validate_milvus_config(config)
            elif db_type_lower == "faiss":
                return VectorDatabaseFactory._validate_faiss_config(config)
            else:
                return False, f"Unsupported database type: {db_type}"

        except Exception as e:
            return False, f"Config validation error: {e}"

    @staticmethod
    def _validate_milvus_config(config: Dict[str, Any]) -> tuple[bool, str]:
        """验证 Milvus 配置"""
        # 检查是否配置了 Lite 路径或标准地址
        lite_path = config.get("milvus_lite_path", "")
        address = config.get("address", "")

        if not lite_path and not address:
            return (
                False,
                "Either 'milvus_lite_path' or 'address' must be configured for Milvus",
            )

        # 如果配置了认证，检查必要字段
        auth_config = config.get("authentication", {})
        if auth_config:
            if auth_config.get("user") and not auth_config.get("password"):
                return False, "Password is required when user is specified"

        return True, ""

    @staticmethod
    def _validate_faiss_config(config: Dict[str, Any]) -> tuple[bool, str]:
        """验证 FAISS 配置"""
        faiss_config = config.get("faiss_config", {})

        # 检查数据路径
        data_path = faiss_config.get("faiss_data_path", "faiss_data")
        if not isinstance(data_path, str) or not data_path.strip():
            return False, "Invalid 'faiss_data_path' configuration"

        # 检查索引类型
        index_type = faiss_config.get("faiss_index_type", "IndexFlatL2")
        supported_types = [
            "IndexFlatL2",
            "IndexFlatIP",
            "IndexIVFFlat",
            "IndexHNSWFlat",
        ]
        if index_type not in supported_types:
            return (
                False,
                f"Unsupported FAISS index type: {index_type}. Supported types: {supported_types}",
            )

        # 检查 nlist 参数（仅对 IVF 索引有效）
        if index_type == "IndexIVFFlat":
            nlist = faiss_config.get("faiss_nlist", 100)
            if not isinstance(nlist, int) or nlist <= 0:
                return False, "Invalid 'faiss_nlist' configuration for IndexIVFFlat"

        return True, ""

    @staticmethod
    def get_default_config(db_type: str) -> Dict[str, Any]:
        """
        获取指定数据库类型的默认配置

        Args:
            db_type: 数据库类型

        Returns:
            默认配置字典
        """
        db_type_lower = db_type.lower()

        if db_type_lower == "milvus":
            return {
                "milvus_lite_path": "",
                "address": "",
                "authentication": {
                    "user": "",
                    "password": "",
                    "token": "",
                    "secure": False,
                },
                "db_name": "default",
                "connection_alias": "",
                "collection_name": "mnemosyne_default",
            }
        elif db_type_lower == "faiss":
            return {
                "faiss_config": {
                    "faiss_data_path": "faiss_data",
                    "faiss_index_type": "IndexFlatL2",
                    "faiss_nlist": 100,
                }
            }
        else:
            return {}

    @staticmethod
    async def migrate_data_async(
        source_db: VectorDatabase,
        target_db: VectorDatabase,
        collection_name: str,
        batch_size: int = 1000,
        progress_callback=None,
    ) -> bool:
        """
        在不同数据库之间迁移数据

        Args:
            source_db: 源数据库
            target_db: 目标数据库
            collection_name: 集合名称
            batch_size: 批处理大小

        Returns:
            迁移是否成功
        """

        try:
            loop = asyncio.get_running_loop()

            # 迁移前兼容性检查
            logger.info("开始迁移前兼容性检查...")
            compatibility_check = await loop.run_in_executor(
                None,
                VectorDatabaseFactory._check_migration_compatibility,
                source_db,
                target_db,
                collection_name,
            )

            if not compatibility_check["compatible"]:
                logger.error(f"迁移兼容性检查失败: {compatibility_check['error']}")
                return False

            logger.info("兼容性检查通过，开始数据迁移...")

            # 检查源集合是否存在
            has_source_collection = await loop.run_in_executor(
                None, source_db.has_collection, collection_name
            )
            if not has_source_collection:
                logger.error(f"Source collection '{collection_name}' does not exist")
                return False

            # 获取源集合的所有数据
            all_data = await loop.run_in_executor(
                None,
                source_db.query,
                collection_name,
                "memory_id >= 0",
                ["*"],
                None,
            )

            if not all_data:
                logger.info(f"No data found in source collection '{collection_name}'")
                return True

            # 获取源集合的完整 schema 信息
            source_stats = await loop.run_in_executor(
                None, source_db.get_collection_stats, collection_name
            )
            vector_dim = source_stats.get("vector_dim", 1024)

            # 根据目标数据库类型创建适当的schema
            target_db_type = target_db.get_database_type().value

            if target_db_type == "milvus":
                from pymilvus import CollectionSchema, FieldSchema, DataType

                fields = [
                    FieldSchema(
                        name="memory_id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=False,
                    ),
                    FieldSchema(
                        name="session_id", dtype=DataType.VARCHAR, max_length=255
                    ),
                    FieldSchema(
                        name="personality_id", dtype=DataType.VARCHAR, max_length=255
                    ),
                    FieldSchema(
                        name="content", dtype=DataType.VARCHAR, max_length=65535
                    ),
                    FieldSchema(name="create_time", dtype=DataType.INT64),
                    FieldSchema(
                        name=VECTOR_FIELD_NAME,
                        dtype=DataType.FLOAT_VECTOR,
                        dim=vector_dim,
                    ),
                ]
                schema = CollectionSchema(
                    fields=fields, description="Mnemosyne memory collection"
                )
            else:
                schema = {"vector_dim": vector_dim, "fields": []}

            # 在目标数据库中创建集合
            created_collection = await loop.run_in_executor(
                None, target_db.create_collection, collection_name, schema
            )
            if not created_collection:
                logger.error(f"Failed to create target collection '{collection_name}'")
                return False

            logger.info(f"Successfully created target collection '{collection_name}'")

            # 为Milvus创建索引
            if target_db_type == "milvus":
                logger.info("Creating index for Milvus collection...")
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                }
                created_index = await loop.run_in_executor(
                    None,
                    target_db.create_index,
                    collection_name,
                    VECTOR_FIELD_NAME,
                    index_params,
                )
                if not created_index:
                    logger.warning(
                        "Failed to create index, but continuing with migration"
                    )
                else:
                    logger.info("Successfully created index for vector field")

            # 分批迁移数据
            total_records = len(all_data)
            migrated_count = 0
            failed_count = 0
            batch_count = (total_records + batch_size - 1) // batch_size

            logger.info(
                f"开始迁移数据: 总记录数={total_records}, 批次大小={batch_size}, 总批次数={batch_count}"
            )
            logger.info(
                f"源数据库: {source_db.get_database_type().value} -> 目标数据库: {target_db_type}"
            )

            for i in range(0, total_records, batch_size):
                batch_num = i // batch_size + 1
                batch_data = all_data[i : i + batch_size]
                current_batch_size = len(batch_data)

                logger.info(
                    f"处理批次 {batch_num}/{batch_count}: 记录数={current_batch_size}"
                )

                # 数据格式验证和转换
                try:
                    processed_batch = (
                        VectorDatabaseFactory._process_migration_data(
                            batch_data, target_db_type, vector_dim
                        )
                    )

                    if len(processed_batch) < current_batch_size:
                        skipped = current_batch_size - len(processed_batch)
                        failed_count += skipped
                        logger.warning(
                            f"批次 {batch_num}: 跳过了 {skipped} 条有问题的记录"
                        )

                except Exception as e:
                    logger.error(f"批次 {batch_num} 数据处理失败: {e}")
                    failed_count += current_batch_size
                    continue

                # 插入数据到目标数据库
                try:
                    insert_result = await loop.run_in_executor(
                        None, target_db.insert, collection_name, processed_batch
                    )
                    if insert_result:
                        migrated_count += len(processed_batch)
                        progress_percent = (migrated_count / total_records) * 100
                        logger.info(
                            f"批次 {batch_num}/{batch_count} 迁移成功: {len(processed_batch)} 条记录"
                        )
                        logger.info(
                            f"总进度: {migrated_count}/{total_records} ({progress_percent:.1f}%)"
                        )

                        if progress_callback:
                            try:
                                await progress_callback(
                                    {
                                        "batch_num": batch_num,
                                        "batch_count": batch_count,
                                        "migrated_count": migrated_count,
                                        "total_records": total_records,
                                        "progress_percent": progress_percent,
                                        "current_batch_size": len(processed_batch),
                                    }
                                )
                            except Exception as e:
                                logger.warning(f"进度回调执行失败: {e}")
                    else:
                        logger.error(f"批次 {batch_num} 插入失败: 无法插入到目标数据库")
                        failed_count += len(processed_batch)

                        if not VectorDatabaseFactory._should_continue_migration(
                            failed_count, total_records
                        ):
                            logger.error("失败记录过多，终止迁移")
                            return False

                except Exception as e:
                    logger.error(f"批次 {batch_num} 插入异常: {e}", exc_info=True)
                    failed_count += len(processed_batch)
                    continue

                await asyncio.sleep(0.01)

            # 迁移完成统计
            success_rate = (
                (migrated_count / total_records) * 100 if total_records > 0 else 0
            )
            logger.info("=" * 60)
            logger.info("数据迁移完成统计:")
            logger.info(f"  源数据库: {source_db.get_database_type().value}")
            logger.info(f"  目标数据库: {target_db_type}")
            logger.info(f"  总记录数: {total_records}")
            logger.info(f"  成功迁移: {migrated_count}")
            logger.info(f"  失败记录: {failed_count}")
            logger.info(f"  成功率: {success_rate:.1f}%")
            logger.info("=" * 60)

            # 验证迁移结果
            if migrated_count > 0:
                try:
                    target_stats = await loop.run_in_executor(
                        None, target_db.get_collection_stats, collection_name
                    )
                    target_count = target_stats.get("row_count", 0)
                    logger.info(f"目标数据库验证: 集合中实际记录数 = {target_count}")

                    if target_count != migrated_count:
                        logger.warning(
                            f"记录数不匹配: 预期={migrated_count}, 实际={target_count}"
                        )
                except Exception as e:
                    logger.warning(f"无法验证目标数据库记录数: {e}")

            if migrated_count == 0 and total_records > 0:
                logger.error("迁移失败: 没有成功迁移任何记录")
                return False
            elif failed_count > 0:
                logger.warning(f"迁移部分成功: {failed_count} 条记录失败")
                return success_rate >= 90
            else:
                logger.info("迁移完全成功: 所有记录都已成功迁移")
                return True

        except Exception as e:
            logger.error(f"Data migration failed: {e}", exc_info=True)
            return False

    @staticmethod
    def _should_continue_migration(failed_count: int, total_records: int) -> bool:
        """判断是否应该继续迁移"""
        if total_records == 0:
            return False

        failure_rate = (failed_count / total_records) * 100
        # 如果失败率超过50%，停止迁移
        return failure_rate < 50.0

    @staticmethod
    def _process_migration_data(
        batch_data: List[Dict], target_db_type: str, vector_dim: int
    ) -> List[Dict]:
        """
        处理迁移数据，确保格式兼容

        Args:
            batch_data: 原始数据批次
            target_db_type: 目标数据库类型
            vector_dim: 向量维度

        Returns:
            处理后的数据批次
        """

        processed_data = []
        skipped_count = 0

        for i, record in enumerate(batch_data):
            try:
                # 基础数据验证
                if not isinstance(record, dict):
                    raise ValueError(f"Record {i} is not a dictionary: {type(record)}")

                # 确保必需字段存在
                memory_id = record.get("memory_id")
                if memory_id is None:
                    raise ValueError(f"Record {i} missing required field: memory_id")

                # 向量数据验证
                vector_data = None
                for field_name in ["embedding", "vector"]:  # 依次尝试 'embedding' 和 'vector'
                    if field_name in record:
                        vector_data = record.get(field_name)
                        break

                if vector_data is None:
                    raise ValueError(
                        f"Record {i} missing required field: embedding or vector"
                    )

                # 处理不同数据库的向量格式差异
                if target_db_type == "milvus":
                    # Milvus要求向量是list格式
                    if isinstance(vector_data, str):
                        # 可能是JSON字符串格式
                        import json

                        try:
                            vector_data = json.loads(vector_data)
                        except json.JSONDecodeError:
                            raise ValueError(f"Record {i}: Invalid vector JSON string")
                    elif hasattr(vector_data, "tolist"):
                        # numpy数组转换
                        vector_data = vector_data.tolist()
                elif target_db_type == "faiss":
                    # FAISS可以处理多种格式，但统一转换为list
                    if hasattr(vector_data, "tolist"):
                        vector_data = vector_data.tolist()
                    elif isinstance(vector_data, str):
                        import json

                        try:
                            vector_data = json.loads(vector_data)
                        except json.JSONDecodeError:
                            raise ValueError(f"Record {i}: Invalid vector JSON string")

                processed_record = {
                    "memory_id": int(memory_id),
                    "session_id": str(record.get("session_id", "")),
                    "personality_id": str(record.get("personality_id", "")),
                    "content": str(record.get("content", "")),
                    "create_time": int(record.get("create_time", 0)),
                    VECTOR_FIELD_NAME: vector_data,
                }

                # 数据类型验证
                if not isinstance(processed_record[VECTOR_FIELD_NAME], list):
                    raise ValueError(
                        f"Record {i}: Vector must be a list, got {type(processed_record[VECTOR_FIELD_NAME])}"
                    )

                if len(processed_record[VECTOR_FIELD_NAME]) != vector_dim:
                    raise ValueError(
                        f"Record {i}: Vector dimension mismatch: expected {vector_dim}, got {len(processed_record[VECTOR_FIELD_NAME])}"
                    )

                # 确保向量是浮点数
                try:
                    processed_record[VECTOR_FIELD_NAME] = [
                        float(x) for x in processed_record[VECTOR_FIELD_NAME]
                    ]
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Record {i}: Invalid vector values: {e}")

                # 特定数据库的格式要求
                if target_db_type == "milvus":
                    # Milvus字符串长度限制
                    processed_record["session_id"] = processed_record["session_id"][
                        :255
                    ]
                    processed_record["personality_id"] = processed_record[
                        "personality_id"
                    ][:255]
                    processed_record["content"] = processed_record["content"][:65535]

                    # Milvus要求memory_id为正整数
                    if processed_record["memory_id"] < 0:
                        processed_record["memory_id"] = abs(
                            processed_record["memory_id"]
                        )

                elif target_db_type == "faiss":
                    # FAISS对字符串长度没有严格限制，但为了一致性进行合理截断
                    if len(processed_record["content"]) > 100000:  # 100KB限制
                        processed_record["content"] = processed_record["content"][
                            :100000
                        ]
                        logger.warning(
                            f"Record {i}: Content truncated to 100KB for FAISS compatibility"
                        )

                processed_data.append(processed_record)

            except Exception as e:
                skipped_count += 1
                logger.error(
                    f"跳过记录 {i} (memory_id: {record.get('memory_id', 'unknown')}): {e}"
                )
                # 跳过有问题的记录，继续处理其他记录
                continue

        if skipped_count > 0:
            logger.warning(
                f"本批次跳过了 {skipped_count}/{len(batch_data)} 条有问题的记录"
            )

        return processed_data

    @staticmethod
    def migrate_data(
        source_db: VectorDatabase,
        target_db: VectorDatabase,
        collection_name: str,
        batch_size: int = 1000,
    ) -> bool:
        """
        同步版本的数据迁移方法（向后兼容）
        """
        import asyncio

        # 如果在异步上下文中，直接调用异步版本
        try:
            loop = asyncio.get_running_loop()
            # 在已有事件循环中，创建新任务
            return asyncio.run_coroutine_threadsafe(
                VectorDatabaseFactory.migrate_data_async(
                    source_db, target_db, collection_name, batch_size
                ),
                loop,
            ).result()
        except RuntimeError:
            # 没有运行的事件循环，创建新的
            return asyncio.run(
                VectorDatabaseFactory.migrate_data_async(
                    source_db, target_db, collection_name, batch_size
                )
            )

    @staticmethod
    def _check_migration_compatibility(
        source_db: VectorDatabase, target_db: VectorDatabase, collection_name: str
    ) -> Dict[str, Any]:
        """
        检查迁移兼容性

        Returns:
            包含兼容性检查结果的字典
        """
        result = {"compatible": True, "error": "", "warnings": []}

        try:
            # 检查源数据库连接
            if not source_db.is_connected():
                result["compatible"] = False
                result["error"] = "源数据库未连接"
                return result

            # 检查目标数据库连接
            if not target_db.is_connected():
                result["compatible"] = False
                result["error"] = "目标数据库未连接"
                return result

            # 检查源集合是否存在
            if not source_db.has_collection(collection_name):
                result["compatible"] = False
                result["error"] = f"源集合 '{collection_name}' 不存在"
                return result

            # 获取源数据库类型和统计信息
            source_type = source_db.get_database_type().value
            target_type = target_db.get_database_type().value
            source_stats = source_db.get_collection_stats(collection_name)

            logger.info(f"源数据库类型: {source_type}")
            logger.info(f"目标数据库类型: {target_type}")
            logger.info(f"源集合统计: {source_stats}")

            # 检查向量维度
            vector_dim = source_stats.get("vector_dim")
            if not vector_dim:
                result["warnings"].append("无法获取源集合的向量维度信息")
            else:
                logger.info(f"向量维度: {vector_dim}")

            # 检查记录数量
            record_count = source_stats.get("row_count", 0)
            logger.info(f"源集合记录数: {record_count}")

            if record_count == 0:
                result["warnings"].append("源集合为空，无数据需要迁移")

            # 特定迁移路径的兼容性检查
            migration_path = f"{source_type}_to_{target_type}"

            if migration_path == "milvus_to_faiss":
                # Milvus到FAISS的特殊检查
                logger.info("检查 Milvus -> FAISS 迁移兼容性...")

                # 检查Milvus集合是否已加载
                try:
                    if hasattr(source_db, "load_collection"):
                        source_db.load_collection(collection_name)
                        logger.info("Milvus集合已加载")
                except Exception as e:
                    result["warnings"].append(f"无法加载Milvus集合: {e}")

                # 检查FAISS目标路径
                if hasattr(target_db, "data_path"):
                    import os

                    target_path = target_db.data_path
                    if os.path.exists(target_path):
                        result["warnings"].append(f"FAISS目标路径已存在: {target_path}")

            elif migration_path == "faiss_to_milvus":
                # FAISS到Milvus的特殊检查
                logger.info("检查 FAISS -> Milvus 迁移兼容性...")

                # 检查Milvus版本兼容性
                try:
                    import pymilvus

                    milvus_version = pymilvus.__version__
                    logger.info(f"PyMilvus版本: {milvus_version}")

                    # 检查是否支持所需的数据类型
                    from pymilvus import DataType

                    required_types = [  # noqa: F841
                        DataType.INT64,
                        DataType.VARCHAR,
                        DataType.FLOAT_VECTOR,
                    ]
                    logger.info("Milvus数据类型支持检查通过")

                except ImportError:
                    result["compatible"] = False
                    result["error"] = "PyMilvus未安装，无法迁移到Milvus"
                    return result
                except Exception as e:
                    result["warnings"].append(f"Milvus兼容性检查警告: {e}")

                # 检查目标集合是否已存在
                if target_db.has_collection(collection_name):
                    result["warnings"].append(
                        f"目标Milvus集合 '{collection_name}' 已存在，将被覆盖"
                    )

            # 内存使用估算
            if record_count > 0 and vector_dim:
                estimated_memory_mb = (record_count * vector_dim * 4) / (
                    1024 * 1024
                )  # 4字节/浮点数
                logger.info(f"估算内存使用: {estimated_memory_mb:.1f} MB")

                if estimated_memory_mb > 1000:  # 1GB
                    result["warnings"].append(
                        f"大数据集迁移，估算内存使用: {estimated_memory_mb:.1f} MB"
                    )

            # 汇总警告
            if result["warnings"]:
                logger.warning("兼容性检查警告:")
                for warning in result["warnings"]:
                    logger.warning(f"  - {warning}")

            return result

        except Exception as e:
            result["compatible"] = False
            result["error"] = f"兼容性检查异常: {e}"
            logger.error(f"兼容性检查失败: {e}", exc_info=True)
            return result
