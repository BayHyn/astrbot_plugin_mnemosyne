# -*- coding: utf-8 -*-
"""
Mnemosyne 插件独立 Web 服务器
提供记忆数据的可视化管理界面
"""

import os
import asyncio
import threading
import time
import secrets
import socket
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from astrbot.api import logger

if TYPE_CHECKING:
    from ..main import Mnemosyne


class MnemosyneWebServer:
    """Mnemosyne 插件独立 Web 服务器"""

    def __init__(self, plugin: "Mnemosyne"):
        self.plugin = plugin

        # 服务器配置
        self.host = plugin.config.get("web_interface", {}).get("host", "0.0.0.0")
        self.port = plugin.config.get("web_interface", {}).get("port", 8765)
        self.enabled = plugin.config.get("web_interface", {}).get("enabled", False)

        # 安全配置
        self.auth_enabled = plugin.config.get("web_interface", {}).get(
            "auth_enabled", True
        )
        self.access_token = plugin.config.get("web_interface", {}).get(
            "access_token", ""
        )

        # 如果没有配置访问令牌，生成一个随机令牌
        if self.auth_enabled and not self.access_token:
            self.access_token = self._generate_access_token()
            logger.info(f"生成了新的访问令牌: {self.access_token}")

        # HTTP Bearer 认证
        self.security = HTTPBearer(auto_error=False) if self.auth_enabled else None

        # 服务器状态
        self.app: Optional[FastAPI] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.server_process = None

        # 自动停止配置
        self.auto_stop_enabled = plugin.config.get("web_interface", {}).get(
            "auto_stop_enabled", True
        )
        self.idle_timeout_minutes = plugin.config.get("web_interface", {}).get(
            "idle_timeout_minutes", 30
        )
        self.auto_port_selection = plugin.config.get("web_interface", {}).get(
            "auto_port_selection", True
        )
        self.last_access_time = datetime.now()
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 静态文件路径
        self.static_path = os.path.join(os.path.dirname(__file__), "static")
        self.templates_path = os.path.join(os.path.dirname(__file__), "templates")

        # 确保目录存在
        os.makedirs(self.static_path, exist_ok=True)
        os.makedirs(self.templates_path, exist_ok=True)

        logger.info(
            f"Web服务器初始化完成 - Host: {self.host}, Port: {self.port}, Enabled: {self.enabled}"
        )
        if self.auto_stop_enabled:
            logger.info(f"自动停止已启用 - 空闲超时: {self.idle_timeout_minutes} 分钟")

    def _generate_access_token(self) -> str:
        """生成访问令牌"""
        return secrets.token_urlsafe(32)

    def _is_port_available(self, host: str, port: int) -> bool:
        """检查端口是否可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception:
            return False

    def _find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """查找可用端口"""
        for i in range(max_attempts):
            port = start_port + i
            if self._is_port_available(self.host, port):
                return port
        return None

    def _check_database_status(self, collection_name: str) -> dict:
        """检查数据库状态"""
        status = {
            "vector_db_exists": False,
            "is_connected": False,
            "collection_exists": False,
            "error": None,
        }

        try:
            if self.plugin.vector_db:
                status["vector_db_exists"] = True
                status["is_connected"] = self.plugin.vector_db.is_connected()

                if hasattr(self.plugin.vector_db, "has_collection"):
                    status["collection_exists"] = self.plugin.vector_db.has_collection(
                        collection_name
                    )
                else:
                    status["collection_exists"] = "unknown"
            else:
                status["error"] = "vector_db 对象为 None"

        except Exception as e:
            status["error"] = str(e)

        return status

    def _verify_token(
        self, credentials: Optional[HTTPAuthorizationCredentials] = None
    ) -> bool:
        """验证访问令牌"""
        if not self.auth_enabled:
            return True

        if not credentials:
            return False

        return credentials.credentials == self.access_token

    def _create_auth_dependency(self):
        """创建认证依赖"""

        async def get_current_user(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(
                self.security
            ),
        ):
            """获取当前用户（认证依赖）"""
            if not self.auth_enabled:
                return True

            if not credentials:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="需要认证",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if not self._verify_token(credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="无效的访问令牌",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return True

        return get_current_user

    def create_app(self) -> FastAPI:
        """创建 FastAPI 应用"""
        app = FastAPI(
            title="Mnemosyne Memory Manager",
            description="AstrBot Mnemosyne 插件记忆管理界面",
            version="0.6.0",
        )

        # 添加 CORS 中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 添加访问时间更新中间件
        @app.middleware("http")
        async def update_access_time(request: Request, call_next):
            # 更新最后访问时间
            self.last_access_time = datetime.now()
            response = await call_next(request)
            return response

        # 挂载静态文件
        if os.path.exists(self.static_path):
            app.mount("/static", StaticFiles(directory=self.static_path), name="static")

        # 注册路由
        self._register_routes(app)

        return app

    def _register_routes(self, app: FastAPI):
        """注册路由"""

        # 创建认证依赖
        auth_dependency = self._create_auth_dependency()

        @app.get("/", response_class=HTMLResponse)
        async def index():
            """主页"""
            return await self._render_template("index.html")

        @app.get("/api/auth/token")
        async def get_auth_info():
            """获取认证信息"""
            return JSONResponse(
                {"auth_enabled": self.auth_enabled, "token_required": self.auth_enabled}
            )

        @app.get("/api/status")
        async def get_status(_=Depends(auth_dependency)):
            """获取插件状态"""
            try:
                # 检查插件是否已初始化
                if not self.plugin._core_components_initialized:
                    return JSONResponse(
                        {"status": "error", "message": "插件尚未完全初始化"}
                    )

                # 获取数据库状态
                db_status = "disconnected"
                db_info = {}

                if self.plugin.vector_db and self.plugin.vector_db.is_connected():
                    db_status = "connected"
                    db_type = self.plugin._get_database_type_safe()
                    db_info = {
                        "type": db_type,
                    }
                else:
                    db_status = "disconnected"
                    db_info = {}

                # 获取嵌入服务状态
                embedding_status = "disconnected"
                embedding_info = {}

                if self.plugin.embedding_adapter:
                    embedding_status = "connected"
                    embedding_info = {
                        "service": self.plugin.embedding_adapter.service_name,
                        "model": self.plugin.embedding_adapter.get_model_name(),
                        "dimension": self.plugin.embedding_adapter.get_dim(),
                    }

                return JSONResponse(
                    {
                        "status": "success",
                        "data": {
                            "plugin_version": "0.6.0",
                            "database": {"status": db_status, "info": db_info},
                            "embedding": {
                                "status": embedding_status,
                                "info": embedding_info,
                            },
                            "config": {
                                "vector_database_type": self.plugin.config.get(
                                    "vector_database_type", "unknown"
                                ),
                                "collection_name": self.plugin.collection_name,
                            },
                        },
                    }
                )

            except Exception as e:
                logger.error(f"获取状态失败: {e}", exc_info=True)
                return JSONResponse(
                    {"status": "error", "message": f"获取状态失败: {str(e)}"}
                )

        @app.get("/api/collections/stats")
        async def get_all_collections_stats(_=Depends(auth_dependency)):
            """获取所有集合的统计信息（可能较慢）"""
            try:
                if (
                    not self.plugin.vector_db
                    or not self.plugin.vector_db.is_connected()
                ):
                    raise HTTPException(status_code=503, detail="数据库未连接")

                collections = self.plugin.vector_db.list_collections()
                collection_stats = {}

                for collection_name in collections:
                    stats = self.plugin.vector_db.get_collection_stats(collection_name)
                    collection_stats[collection_name] = stats

                return JSONResponse({"status": "success", "data": collection_stats})

            except Exception as e:
                logger.error(f"获取所有集合统计信息失败: {e}", exc_info=True)
                return JSONResponse(
                    {
                        "status": "error",
                        "message": f"获取所有集合统计信息失败: {str(e)}",
                    }
                )

        @app.get("/api/memories")
        async def get_memories(
            collection_name: str,
            page: int = 1,
            page_size: int = 20,
            search: str = None,
            session_id: str = None,
            _=Depends(auth_dependency),
        ):
            """获取记忆数据（分页）"""
            try:
                if (
                    not self.plugin.vector_db
                    or not self.plugin.vector_db.is_connected()
                ):
                    raise HTTPException(status_code=503, detail="数据库未连接")

                if not collection_name:
                    raise HTTPException(status_code=400, detail="必须提供集合名称")

                # 构建查询条件
                expression = "memory_id >= 0"  # 基础条件

                if session_id:
                    expression = f"session_id == '{session_id}'"

                # 查询数据
                # 优化：先获取总数，再分页查询
                # 统一数据获取、搜索和分页逻辑，以兼容 Milvus 和 FAISS
                # 1. 从数据库获取所有符合 expression 的记录
                all_records = self.plugin.vector_db.query(
                    collection_name=collection_name,
                    expression=expression,
                    output_fields=["*"],  # 获取所有字段用于后续处理
                )

                if all_records is None:
                    all_records = []
                    logger.warning(
                        f"查询返回 None - 集合: {collection_name}, 表达式: {expression}"
                    )
                logger.debug(f"FAISS Query returned {len(all_records)} records. Sample: {all_records[:3]}")


                # 2. 在内存中进行文本搜索 (如果需要)
                if search and search.strip():
                    search_term = search.strip().lower()
                    all_records = [
                        r
                        for r in all_records
                        if search_term in r.get("content", "").lower()
                    ]

                # 3. 获取总数并进行分页
                total_count = len(all_records)
                start_index = (page - 1) * page_size
                end_index = start_index + page_size
                page_records = all_records[start_index:end_index]
                logger.debug(f"Paging: total={total_count}, page_records={len(page_records)}. Sample: {page_records[:3]}")

                # 格式化记录
                formatted_records = []
                for record in page_records:
                    formatted_record = {
                        "memory_id": record.get("memory_id"),
                        "content": record.get("content", ""),
                        "session_id": record.get("session_id", ""),
                        "personality_id": record.get("personality_id", ""),
                        "create_time": record.get("create_time", 0),
                        "create_time_str": self._format_timestamp(
                            record.get("create_time", 0)
                        ),
                    }
                    formatted_records.append(formatted_record)

                logger.debug(f"Formatted {len(formatted_records)} records for response. Sample: {formatted_records[:3]}")

                return JSONResponse(
                    {
                        "status": "success",
                        "data": {
                            "records": formatted_records,
                            "pagination": {
                                "page": page,
                                "page_size": page_size,
                                "total_count": total_count,
                                "total_pages": (total_count + page_size - 1)
                                // page_size,
                            },
                        },
                    }
                )

            except Exception as e:
                logger.error(f"获取记忆数据失败: {e}", exc_info=True)
                return JSONResponse(
                    {"status": "error", "message": f"获取记忆数据失败: {str(e)}"}
                )

        @app.delete("/api/collections/{collection_name}/memories/{memory_id}")
        async def delete_memory(
            collection_name: str, memory_id: int, _=Depends(auth_dependency)
        ):
            """删除指定记忆"""
            try:
                if (
                    not self.plugin.vector_db
                    or not self.plugin.vector_db.is_connected()
                ):
                    raise HTTPException(status_code=503, detail="数据库未连接")

                # 删除记录
                success = self.plugin.vector_db.delete(
                    collection_name=collection_name,
                    expression=f"memory_id == {memory_id}",
                )

                if success:
                    # 在后台刷新 Milvus 集合
                    if self.plugin._get_database_type_safe().startswith("milvus"):
                        loop = asyncio.get_running_loop()
                        loop.run_in_executor(None, self.plugin.vector_db.flush, [collection_name])
                    return JSONResponse(
                        {"status": "success", "message": f"记忆 {memory_id} 已删除"}
                    )
                else:
                    return JSONResponse({"status": "error", "message": "删除失败"})

            except Exception as e:
                logger.error(f"删除记忆失败: {e}", exc_info=True)
                return JSONResponse(
                    {"status": "error", "message": f"删除记忆失败: {str(e)}"}
                )

        @app.delete("/api/collections/{collection_name}/sessions/{session_id}")
        async def delete_session_memories(
            collection_name: str, session_id: str, _=Depends(auth_dependency)
        ):
            """删除指定会话的所有记忆"""
            try:
                if (
                    not self.plugin.vector_db
                    or not self.plugin.vector_db.is_connected()
                ):
                    raise HTTPException(status_code=503, detail="数据库未连接")

                # 删除会话记录
                success = self.plugin.vector_db.delete(
                    collection_name=collection_name,
                    expression=f"session_id == '{session_id}'",
                )

                if success:
                    # 在后台刷新 Milvus 集合
                    if self.plugin._get_database_type_safe().startswith("milvus"):
                        loop = asyncio.get_running_loop()
                        loop.run_in_executor(None, self.plugin.vector_db.flush, [collection_name])
                    return JSONResponse(
                        {
                            "status": "success",
                            "message": f"会话 {session_id} 的所有记忆已删除",
                        }
                    )
                else:
                    return JSONResponse({"status": "error", "message": "删除失败"})

            except Exception as e:
                logger.error(f"删除会话记忆失败: {e}", exc_info=True)
                return JSONResponse(
                    {"status": "error", "message": f"删除会话记忆失败: {str(e)}"}
                )

        @app.get("/api/statistics")
        async def get_statistics(collection_name: str, _=Depends(auth_dependency)):
            """获取统计信息"""
            try:
                if (
                    not self.plugin.vector_db
                    or not self.plugin.vector_db.is_connected()
                ):
                    raise HTTPException(status_code=503, detail="数据库未连接")

                if not collection_name:
                    raise HTTPException(status_code=400, detail="必须提供集合名称")

                # 获取所有记录用于统计
                # 优化：直接在数据库层面进行统计
                stats_records = self.plugin.vector_db.query(
                    collection_name=collection_name,
                    expression="memory_id >= 0",
                    output_fields=["session_id", "create_time", "personality_id"],
                    limit=16384,  # Milvus 的最大 limit
                )

                # 检查查询结果
                if stats_records is None:
                    stats_records = []
                    logger.warning(f"统计查询返回 None - 集合: {collection_name}")
                    logger.warning(
                        "可能原因: 1) 数据库连接问题 2) 集合不存在 3) 集合为空"
                    )

                    # 检查数据库连接状态
                    if self.plugin.vector_db:
                        logger.info(
                            f"数据库连接状态: {self.plugin.vector_db.is_connected()}"
                        )
                    else:
                        logger.error("vector_db 对象为 None")

                # 统计数据
                total_memories = len(stats_records)
                unique_sessions = len(
                    set(record.get("session_id", "") for record in stats_records)
                )
                unique_personalities = len(
                    set(record.get("personality_id", "") for record in stats_records)
                )

                # 时间分布统计（按天）
                time_distribution = {}
                for record in stats_records:
                    create_time = record.get("create_time", 0)
                    if create_time > 0:
                        date_str = self._format_date(create_time)
                        time_distribution[date_str] = (
                            time_distribution.get(date_str, 0) + 1
                        )

                # 会话分布统计
                session_distribution = {}
                for record in stats_records:
                    session_id = record.get("session_id", "unknown")
                    session_distribution[session_id] = (
                        session_distribution.get(session_id, 0) + 1
                    )

                return JSONResponse(
                    {
                        "status": "success",
                        "data": {
                            "total_memories": total_memories,
                            "unique_sessions": unique_sessions,
                            "unique_personalities": unique_personalities,
                            "time_distribution": time_distribution,
                            "session_distribution": session_distribution,
                        },
                    }
                )

            except Exception as e:
                logger.error(f"获取统计信息失败: {e}", exc_info=True)
                return JSONResponse(
                    {"status": "error", "message": f"获取统计信息失败: {str(e)}"}
                )

    async def _render_template(self, template_name: str) -> str:
        """渲染模板"""
        template_path = os.path.join(self.templates_path, template_name)

        if not os.path.exists(template_path):
            # 如果模板不存在，返回默认页面
            return self._get_default_html()

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取模板失败: {e}")
            return self._get_default_html()

    def _get_default_html(self) -> str:
        """获取默认HTML页面"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mnemosyne Memory Manager</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>Mnemosyne Memory Manager</h1>
            <p>记忆管理界面正在加载中...</p>
            <p>如果您看到此页面，说明Web服务器已启动，但前端界面尚未完全配置。</p>
        </body>
        </html>
        """

    def start(self) -> bool:
        """启动Web服务器"""
        if self.is_running:
            logger.warning("Web服务器已在运行中")
            return True

        if not self.enabled:
            logger.info("Web界面已禁用，跳过启动")
            return False

        try:
            # 检查并生成访问令牌
            if self.auth_enabled and not self.access_token:
                self.access_token = self._generate_access_token()
                logger.info(f"生成了新的访问令牌: {self.access_token}")

            # 检查端口是否可用
            if not self._is_port_available(self.host, self.port):
                if self.auto_port_selection:
                    logger.warning(f"端口 {self.port} 已被占用，尝试查找可用端口...")
                    available_port = self._find_available_port(self.port)
                    if available_port:
                        self.port = available_port
                        logger.info(f"使用可用端口: {self.port}")
                    else:
                        logger.error(
                            f"无法找到可用端口（从 {self.port} 开始的10个端口都被占用）"
                        )
                        return False
                else:
                    logger.error(f"端口 {self.port} 已被占用，且自动端口选择已禁用")
                    return False

            self.app = self.create_app()

            # 在新线程中启动服务器
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()

            # 等待服务器启动
            max_wait = 10  # 最多等待10秒
            for _ in range(max_wait * 10):
                if self.is_running:
                    break
                time.sleep(0.1)

            if self.is_running:
                logger.info(f"Web服务器已启动: http://{self.host}:{self.port}")

                # 启动监控线程
                if self.auto_stop_enabled:
                    self._start_monitor_thread()

                return True
            else:
                logger.error("Web服务器启动超时")
                return False

        except Exception as e:
            logger.error(f"启动Web服务器失败: {e}", exc_info=True)
            return False

    def _start_monitor_thread(self):
        """启动监控线程"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return

        self.stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_idle_timeout, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Web服务器监控线程已启动")

    def _monitor_idle_timeout(self):
        """监控空闲超时"""
        while not self.stop_event.is_set():
            try:
                # 检查是否超时
                if self.is_running:
                    idle_time = datetime.now() - self.last_access_time
                    timeout_threshold = timedelta(minutes=self.idle_timeout_minutes)

                    if idle_time > timeout_threshold:
                        logger.info(
                            f"Web服务器空闲超时 ({self.idle_timeout_minutes} 分钟)，自动停止"
                        )
                        self._auto_stop()
                        break

                # 每分钟检查一次
                if self.stop_event.wait(60):
                    break

            except Exception as e:
                logger.error(f"监控线程异常: {e}", exc_info=True)
                break

        logger.info("Web服务器监控线程已停止")

    def _auto_stop(self):
        """自动停止服务器"""
        try:
            logger.info("正在自动停止Web服务器...")
            self.stop()
        except Exception as e:
            logger.error(f"自动停止Web服务器失败: {e}", exc_info=True)

    def _run_server(self):
        """在线程中运行服务器"""
        try:
            self.is_running = True

            # 创建uvicorn配置
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",  # 减少日志输出
                access_log=False,
                loop="asyncio",
            )

            # 创建服务器实例
            server = uvicorn.Server(config)
            self.server_process = server

            # 运行服务器
            server.run()

        except OSError as e:
            if e.errno == 10048:  # 端口被占用
                logger.error(f"端口 {self.port} 被占用: {e}")
            else:
                logger.error(f"Web服务器网络错误: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Web服务器运行错误: {e}", exc_info=True)
        finally:
            self.is_running = False

    def stop(self) -> bool:
        """停止Web服务器"""
        if not self.is_running:
            logger.info("Web服务器未在运行")
            return True

        try:
            # 停止监控线程
            self.stop_event.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=3)

            # 停止uvicorn服务器
            if self.server_process:
                try:
                    self.server_process.should_exit = True
                except Exception as e:
                    logger.debug(f"停止uvicorn服务器时出现异常: {e}")

            self.is_running = False

            # 等待服务器线程结束
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)
                if self.server_thread.is_alive():
                    logger.warning("服务器线程未能在5秒内停止")

            logger.info("Web服务器已停止")
            return True

        except Exception as e:
            logger.error(f"停止Web服务器失败: {e}", exc_info=True)
            return False

    def get_status(self) -> dict:
        """获取服务器状态"""
        status = {
            "enabled": self.enabled,
            "running": self.is_running,
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}" if self.is_running else None,
            "auto_stop_enabled": self.auto_stop_enabled,
            "idle_timeout_minutes": self.idle_timeout_minutes,
        }

        if self.is_running:
            idle_time = datetime.now() - self.last_access_time
            status["last_access_time"] = self.last_access_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            status["idle_minutes"] = int(idle_time.total_seconds() / 60)
            status["remaining_minutes"] = max(
                0, self.idle_timeout_minutes - status["idle_minutes"]
            )

        return status

    def _format_timestamp(self, timestamp: float) -> str:
        """格式化时间戳为可读字符串"""
        try:
            import datetime

            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError, OverflowError):
            return "未知时间"

    def _format_date(self, timestamp: float) -> str:
        """格式化时间戳为日期字符串"""
        try:
            import datetime

            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            return "未知日期"
