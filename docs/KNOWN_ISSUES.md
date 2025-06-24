# 已知问题和解决方案

## 🚨 第三方库警告

### pymilvus pkg_resources 弃用警告

**问题描述:**
```
UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
```

**原因分析:**
- `pymilvus` 库内部使用了已弃用的 `pkg_resources` 模块
- 这是 `pymilvus` 库本身的问题，不是插件代码问题
- `pkg_resources` 计划在 2025-11-30 移除

**影响程度:**
- ⚠️ **仅为警告**: 不影响功能正常使用
- 🔄 **临时性问题**: 等待 `pymilvus` 库更新修复
- 📅 **预计解决**: pymilvus 团队会在未来版本中修复

**解决方案:**

#### 方案1: 自动抑制（已实现）
插件已自动抑制此警告，无需用户操作。

#### 方案2: 手动固定版本
如果仍然看到警告，可以在虚拟环境中执行：
```bash
pip install "setuptools<81"
```

#### 方案3: 忽略警告
这是一个已知的无害警告，可以安全忽略。

**监控状态:**
- ✅ 插件已实现自动警告过滤
- 🔍 持续关注 pymilvus 库更新
- 📋 将在 pymilvus 修复后移除警告过滤器

## 🔧 其他已知问题

### 端口占用问题

**问题描述:**
```
ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8765): 通常每个套接字地址(协议/网络地址/端口)只允许使用一次。
```

**解决方案:**
- ✅ 已实现自动端口选择
- 🛠️ 使用 `/memory web_cleanup` 清理资源
- ⚙️ 在配置中启用 `auto_port_selection: true`

### 重启后令牌丢失

**问题描述:**
重启 AstrBot 后，Web 界面访问令牌不显示。

**解决方案:**
- ✅ 已修复：每次启动自动生成新令牌
- 📋 使用 `/memory web_status` 查看当前令牌
- 🔧 可在配置中设置固定令牌

### Web API 参数错误

**问题描述:**
```
加载记忆数据失败: 获取记忆数据失败: MilvusManager.query() missing 1 required positional argument: 'expression'
```

**原因分析:**
- Web界面代码中使用了错误的参数名 `filters`
- Milvus 向量数据库的 `query` 和 `delete` 方法需要 `expression` 参数

**解决方案:**
- ✅ 已修复：将所有 `filters` 参数改为 `expression`
- 🔧 影响的方法：`query()` 和 `delete()`
- 📋 修复文件：`web_interface/web_server.py`

### Web API NoneType 错误

**问题描述:**
```
加载记忆数据失败: 获取记忆数据失败: object of type 'NoneType' has no len()
```

**原因分析:**
- `vector_db.query()` 方法返回 `None`
- 代码直接对 `None` 调用 `len()` 函数导致错误
- 可能原因：数据库连接问题、集合不存在、集合为空

**解决方案:**
- ✅ 已修复：添加 `None` 值检查和处理
- 🔧 返回空列表 `[]` 替代 `None`
- 📋 增强错误日志和诊断信息
- 🛠️ 添加数据库状态检查功能

## 📊 问题报告

如果遇到新的问题，请提供以下信息：

1. **错误信息**: 完整的错误日志
2. **环境信息**: Python 版本、操作系统
3. **配置信息**: 相关的配置设置
4. **重现步骤**: 如何重现问题

## 🔄 更新日志

### v0.6.0
- ✅ 修复端口占用问题
- ✅ 修复重启后令牌丢失问题
- ✅ 添加 pymilvus 警告过滤器
- ✅ 实现自动停止功能
- ✅ 修复 Web API 参数错误（filters → expression）
- ✅ 修复 Web API NoneType 错误（添加空值检查）
- ✅ 修复 Milvus 迁移脚本问题（Schema兼容性和数据格式）
- ✅ 增强迁移进度显示和错误处理（详细日志和兼容性检查）
- ✅ 实现异步迁移功能（使用 asyncio.create_task 替代 run_in_executor）

### Milvus 迁移脚本问题

**问题描述:**
- Schema不兼容：Milvus需要CollectionSchema对象，但迁移脚本传递字典
- 数据格式问题：缺少数据类型验证和转换
- 索引缺失：迁移后集合无法正常搜索

**原因分析:**
- 迁移脚本使用简化的schema格式
- 不同数据库的数据格式要求不同
- 缺少完整的schema定义和索引创建

**解决方案:**
- ✅ 已修复：为Milvus创建正确的CollectionSchema
- ✅ 添加数据格式验证和转换逻辑
- ✅ 自动创建向量索引
- ✅ 增强错误处理和日志记录
- 🧪 提供迁移测试脚本验证功能

---

**注意**: 本文档会随着问题的解决和新问题的发现而更新。
