# Mnemosyne Web 可视化界面

## 🌟 概述

Mnemosyne Web 界面是一个独立的可视化管理系统，为 AstrBot Mnemosyne 插件提供直观的记忆数据管理功能。

## ✨ 主要特性

### 📊 数据可视化
- **实时统计**: 显示记忆总数、会话数、人格数等关键指标
- **时间分布图**: 记忆创建时间的趋势分析
- **会话分布图**: 不同会话的记忆数量分布

### 🔍 数据浏览
- **分页浏览**: 高效的分页显示，支持 10/20/50 条每页
- **智能搜索**: 基于内容的全文搜索功能
- **会话过滤**: 按会话ID筛选记忆数据
- **详细信息**: 显示记忆内容、创建时间、会话ID等

### 🗑️ 数据管理
- **单条删除**: 删除指定的记忆条目
- **批量删除**: 删除整个会话的所有记忆
- **安全确认**: 所有删除操作都需要用户确认

### 🔒 安全保护
- **访问认证**: 基于 Bearer Token 的认证机制
- **自动令牌**: 自动生成安全的访问令牌
- **可选认证**: 支持禁用认证（仅限本地使用）

## 🏗️ 技术架构

### 后端技术
- **FastAPI**: 高性能的 Python Web 框架
- **Uvicorn**: ASGI 服务器
- **异步处理**: 全异步的 API 设计
- **RESTful API**: 标准的 REST 接口

### 前端技术
- **原生 JavaScript**: 无需复杂构建工具
- **Bootstrap 5**: 现代化的 UI 框架
- **Chart.js**: 专业的图表库
- **响应式设计**: 支持各种屏幕尺寸

### 安全特性
- **CORS 支持**: 跨域资源共享配置
- **HTTP Bearer**: 标准的认证方式
- **令牌验证**: 安全的访问控制

## 📁 文件结构

```
web_interface/
├── __init__.py              # 模块初始化
├── web_server.py           # Web 服务器主文件
├── static/                 # 静态资源
│   └── js/
│       └── app.js         # 前端应用逻辑
├── templates/              # HTML 模板
│   └── index.html         # 主页面模板
└── README.md              # 本文件
```

## 🚀 快速开始

### 1. 配置启用

在 AstrBot 配置中启用 Web 界面：

```json
{
  "web_interface": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8765,
    "auth_enabled": true,
    "access_token": ""
  }
}
```

### 2. 启动服务

```bash
/memory web_start
```

### 3. 访问界面

浏览器打开: http://127.0.0.1:8765

## 🔧 API 接口

### 认证接口
- `GET /api/auth/token` - 获取认证信息

### 状态接口
- `GET /api/status` - 获取插件状态
- `GET /api/statistics` - 获取统计信息

### 数据接口
- `GET /api/collections` - 获取集合列表
- `GET /api/memories` - 获取记忆数据
- `DELETE /api/memories/{id}` - 删除记忆
- `DELETE /api/memories/session/{session_id}` - 删除会话记忆

## 🛡️ 安全建议

1. **启用认证**: 保持 `auth_enabled: true`
2. **本地访问**: 使用 `127.0.0.1` 限制访问
3. **防火墙**: 配置适当的网络安全规则
4. **定期更新**: 定期更换访问令牌

## 🔍 故障排除

### 常见问题

**端口占用**
```bash
# 检查端口占用
netstat -an | grep 8765

# 更换端口
# 在配置中修改 port 值
```

**认证失败**
```bash
# 重新生成令牌
/memory web_stop
/memory web_start
```

**数据不显示**
```bash
# 检查插件状态
/memory status

# 检查数据库连接
/memory validate_config
```

## 📊 性能优化

- **分页大小**: 根据数据量调整页面大小
- **搜索优化**: 使用具体关键词搜索
- **缓存策略**: 浏览器会自动缓存静态资源

## 🔄 更新历史

### v0.6.0 (2024-06-24)
- 首次发布 Web 可视化界面
- 支持完整的记忆数据管理
- 集成安全认证机制
- 提供 REST API 接口

## 📞 支持

如有问题或建议：
1. 查看插件文档
2. 检查 AstrBot 日志
3. 联系插件开发者

---

**Mnemosyne Web Interface** - 让记忆管理更直观、更高效！
