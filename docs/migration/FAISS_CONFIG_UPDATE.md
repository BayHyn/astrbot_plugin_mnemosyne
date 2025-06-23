# FAISS 配置结构更新指南

## 📋 概述

为了更好地组织配置结构，FAISS 相关配置项已被归类到 `faiss_config` 对象下。这样可以：

- 🗂️ **更清晰的配置结构**: 相关配置项集中管理
- 🔧 **更好的扩展性**: 便于添加新的 FAISS 配置选项
- 📝 **更易维护**: 配置逻辑更加清晰

## 🔄 配置变更对比

### 旧配置结构 (v0.6.0 之前)
```json
{
  "vector_database_type": "faiss",
  "faiss_data_path": "faiss_data",
  "faiss_index_type": "IndexFlatL2",
  "faiss_nlist": 100,
  "embedding_provider_id": "your_provider_id",
  "LLM_providers": "your_llm_provider"
}
```

### 新配置结构 (v0.6.0+)
```json
{
  "vector_database_type": "faiss",
  "faiss_config": {
    "faiss_data_path": "faiss_data",
    "faiss_index_type": "IndexFlatL2",
    "faiss_nlist": 100
  },
  "embedding_provider_id": "your_provider_id",
  "LLM_providers": "your_llm_provider"
}
```

## 🛠️ 自动迁移

插件提供了自动迁移功能，无需手动修改配置：

### 方法一：使用迁移命令（推荐）
```bash
# 查看当前配置状态
/memory status

# 自动迁移配置结构
/memory migrate_config

# 验证迁移结果
/memory validate_config
```

### 方法二：使用迁移脚本
```bash
python docs/development/migration_tool.py --config your_config.json
```

## 📊 配置项详解

### `faiss_config` 对象包含的配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `faiss_data_path` | string | `"faiss_data"` | FAISS 数据存储路径（相对于插件数据目录） |
| `faiss_index_type` | string | `"IndexFlatL2"` | FAISS 索引类型 |
| `faiss_nlist` | integer | `100` | IVF 索引的聚类中心数量 |

### 索引类型选择

| 索引类型 | 特点 | 适用场景 |
|----------|------|----------|
| `IndexFlatL2` | 精确搜索，速度较慢 | 小数据集，要求高精度 |
| `IndexFlatIP` | 内积相似度，精确搜索 | 归一化向量，内积计算 |
| `IndexIVFFlat` | 快速搜索，略有精度损失 | 大数据集，平衡速度和精度 |
| `IndexHNSWFlat` | 高性能近似搜索 | 大数据集，要求高速度 |

## 🔍 验证配置

### 检查配置是否正确
```bash
# 验证配置结构和参数
/memory validate_config

# 查看详细状态信息
/memory status
```

### 配置验证要点
- ✅ `faiss_config` 对象存在
- ✅ `faiss_data_path` 为有效路径
- ✅ `faiss_index_type` 为支持的类型
- ✅ `faiss_nlist` 为正整数（仅 IVF 索引需要）

## 🚨 常见问题

### Q1: 旧配置还能使用吗？
**A**: 可以，但建议迁移到新结构。插件会自动处理旧配置，但新功能可能需要新结构。

### Q2: 迁移会丢失数据吗？
**A**: 不会。配置迁移只是重新组织配置结构，不会影响已存储的向量数据。

### Q3: 如何回滚到旧配置？
**A**: 手动编辑配置文件，将 `faiss_config` 下的配置项移到根级别，但不推荐这样做。

### Q4: 新配置有什么优势？
**A**: 
- 更清晰的配置组织
- 更好的类型检查和验证
- 便于添加新的 FAISS 功能
- 与其他数据库配置保持一致

## 📝 手动迁移步骤

如果需要手动迁移配置：

### 步骤 1: 备份原配置
```bash
cp your_config.json your_config.json.backup
```

### 步骤 2: 修改配置结构
将以下配置项：
```json
{
  "faiss_data_path": "faiss_data",
  "faiss_index_type": "IndexFlatL2", 
  "faiss_nlist": 100
}
```

改为：
```json
{
  "faiss_config": {
    "faiss_data_path": "faiss_data",
    "faiss_index_type": "IndexFlatL2",
    "faiss_nlist": 100
  }
}
```

### 步骤 3: 验证配置
```bash
/memory validate_config
```

## 🎯 最佳实践

### 1. 使用相对路径
```json
{
  "faiss_config": {
    "faiss_data_path": "faiss_data"  // 推荐：相对路径
  }
}
```

### 2. 根据数据量选择索引
- **小数据集 (<10K)**: `IndexFlatL2`
- **中等数据集 (10K-100K)**: `IndexIVFFlat`
- **大数据集 (>100K)**: `IndexHNSWFlat`

### 3. 调整 nlist 参数
```json
{
  "faiss_config": {
    "faiss_index_type": "IndexIVFFlat",
    "faiss_nlist": 100  // 数据量大时可增加到 1000+
  }
}
```

## 🔗 相关文档

- [重构指南](REFACTOR_GUIDE.md) - 完整的 v0.6.0 重构说明
- [迁移示例](MIGRATION_EXAMPLES.md) - 详细的迁移使用示例
- [插件数据目录](../guides/PLUGIN_DATA_DIRECTORY.md) - 数据目录管理说明

## 📞 获取帮助

如果在配置迁移过程中遇到问题：

1. 使用 `/memory status` 检查当前状态
2. 使用 `/memory validate_config` 验证配置
3. 查看 AstrBot 日志获取详细错误信息
4. 在 [GitHub Issues](https://github.com/lxfight/astrbot_plugin_mnemosyne/issues) 寻求帮助

---

*最后更新: 2024-06-23*  
*适用版本: v0.6.0+*
