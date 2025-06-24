# Mnemosyne 迁移最佳实践指南

## 🎯 迁移前准备

### 1. 数据备份
```bash
# 备份FAISS数据
cp -r faiss_data faiss_data_backup_$(date +%Y%m%d)

# 备份Milvus数据
cp milvus.db milvus_backup_$(date +%Y%m%d).db

# 备份配置文件
cp config.json config_backup_$(date +%Y%m%d).json
```

### 2. 环境检查
- 确保有足够的磁盘空间（至少是原数据的2倍）
- 检查内存使用情况
- 确认网络连接稳定（如果使用远程Milvus）

### 3. 配置验证
```bash
# 验证当前配置
/memory validate_config

# 检查插件状态
/memory status
```

## 🔄 迁移流程

### 从 FAISS 迁移到 Milvus

#### 步骤1: 配置Milvus连接
```json
{
  "vector_database_type": "faiss",  // 保持当前类型
  "milvus_lite_path": "./milvus.db",  // 添加Milvus配置
  "collection_name": "mnemosyne_default"
}
```

#### 步骤2: 验证Milvus配置
```bash
/memory validate_config
```

#### 步骤3: 执行迁移
```bash
# 确认迁移
/memory migrate_to_milvus --confirm
```

#### 步骤4: 验证迁移结果
```bash
# 重启插件后检查状态
/memory status

# 检查数据完整性
/memory list_records mnemosyne_default 10
```

### 从 Milvus 迁移到 FAISS

#### 步骤1: 配置FAISS路径
```json
{
  "vector_database_type": "milvus",  // 保持当前类型
  "faiss_config": {  // 添加FAISS配置
    "faiss_data_path": "faiss_data",
    "faiss_index_type": "IndexFlatL2",
    "faiss_nlist": 100
  }
}
```

#### 步骤2: 执行迁移
```bash
/memory migrate_to_faiss --confirm
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### 1. Schema不兼容错误
**错误信息**: `TypeError: create_collection() argument must be CollectionSchema`

**解决方案**:
- 确保使用最新版本的迁移脚本
- 检查pymilvus版本兼容性
- 重新启动插件

#### 2. 数据格式错误
**错误信息**: `Vector dimension mismatch`

**解决方案**:
```bash
# 检查源数据库的向量维度
/memory status

# 确保目标数据库配置匹配
```

#### 3. 内存不足
**错误信息**: `MemoryError` 或迁移中断

**解决方案**:
- 减少批处理大小（默认1000）
- 释放其他程序占用的内存
- 分批次迁移大数据集

#### 4. 权限问题
**错误信息**: `Permission denied`

**解决方案**:
- 检查数据目录写权限
- 确保以正确用户身份运行
- 检查文件系统空间

### 迁移失败恢复

#### 1. 回滚配置
```bash
# 恢复原配置文件
cp config_backup_YYYYMMDD.json config.json

# 重启插件
```

#### 2. 数据恢复
```bash
# 恢复FAISS数据
rm -rf faiss_data
cp -r faiss_data_backup_YYYYMMDD faiss_data

# 恢复Milvus数据
cp milvus_backup_YYYYMMDD.db milvus.db
```

## 📊 性能优化

### 迁移性能调优

#### 1. 批处理大小
```python
# 大内存环境
batch_size = 5000

# 普通环境
batch_size = 1000

# 低内存环境
batch_size = 100
```

#### 2. 异步迁移优势
- ✅ 使用 `asyncio.create_task` 替代 `run_in_executor`
- ✅ 支持实时进度回调和状态更新
- ✅ 更好的并发控制和资源管理
- ✅ 可以取消长时间运行的迁移任务
- ✅ 不阻塞事件循环，响应更快

#### 3. 并发控制
- 迁移期间避免其他数据库操作
- 暂停自动备份任务
- 关闭不必要的应用程序

#### 4. 网络优化（远程Milvus）
- 使用本地网络连接
- 增加连接超时时间
- 启用数据压缩

## 🔍 数据验证

### 迁移后验证清单

#### 1. 数据完整性
```bash
# 检查记录数量
/memory status

# 抽样检查数据
/memory list_records collection_name 20
```

#### 2. 功能测试
```bash
# 测试搜索功能
# 进行正常对话，观察记忆功能

# 测试Web界面
/memory web_start
```

#### 3. 性能测试
- 观察响应时间
- 检查内存使用
- 监控磁盘I/O

## 📋 迁移检查表

### 迁移前
- [ ] 数据已备份
- [ ] 配置已验证
- [ ] 磁盘空间充足
- [ ] 权限检查完成
- [ ] 其他应用已暂停

### 迁移中
- [ ] 监控迁移进度（实时显示批次进度和百分比）
- [ ] 观察错误日志（详细的兼容性检查和数据处理日志）
- [ ] 检查系统资源（内存和磁盘使用）
- [ ] 避免中断操作（等待当前批次完成）
- [ ] 关注失败率（超过50%会自动停止）

### 迁移后
- [ ] 插件已重启
- [ ] 状态检查正常
- [ ] 数据完整性验证
- [ ] 功能测试通过
- [ ] 性能表现正常
- [ ] 清理临时文件

## 🚨 紧急情况处理

### 迁移中断
1. 不要强制终止进程
2. 等待当前批次完成
3. 检查日志确定中断原因
4. 根据情况决定重试或回滚

### 数据损坏
1. 立即停止所有操作
2. 从备份恢复数据
3. 分析损坏原因
4. 修复问题后重新迁移

### 系统崩溃
1. 重启系统
2. 检查文件系统完整性
3. 从备份恢复数据
4. 重新配置环境

## 📞 获取帮助

### 日志收集
```bash
# 收集相关日志
grep -i "migration\|milvus\|faiss" astrbot.log > migration_logs.txt

# 查看详细迁移进度
grep "批次\|进度\|兼容性" astrbot.log

# 查看错误和警告
grep -E "ERROR|WARNING.*migration" astrbot.log
```

### 迁移问题检查
```bash
# 运行迁移问题检查工具
python check_migration_issues.py

# 运行迁移功能测试
python test_migration.py
```

### 问题报告
提供以下信息：
- 迁移类型（FAISS→Milvus 或 Milvus→FAISS）
- 错误信息和日志
- 系统环境信息
- 数据规模和配置

---

**注意**: 迁移是一个重要操作，建议在测试环境中先验证流程，确保生产环境迁移的成功率。
