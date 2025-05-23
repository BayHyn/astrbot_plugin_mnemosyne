# 🧠 Mnemosyne - AstrBot 的长期记忆中枢

> *"Memory is the process of retaining information over time."*
> *"Memory is the means by which we draw on our past experiences in order to use this information in the present."*
> — (Paraphrased concepts based on memory research, attributing specific short quotes can be tricky)
>
> **让 AI 真正记住与你的每一次对话，构建持久的个性化体验。**

---

## 💬 支持与讨论

遇到问题或想交流使用心得？加入我们的讨论群：
[![加入QQ群](https://img.shields.io/badge/QQ群-953245617-blue?style=flat-square&logo=tencent-qq)](https://qm.qq.com/cgi-bin/qm/qr?k=WdyqoP-AOEXqGAN08lOFfVSguF2EmBeO&jump_from=webapi&authKey=tPyfv90TVYSGVhbAhsAZCcSBotJuTTLf03wnn7/lQZPUkWfoQ/J8e9nkAipkOzwh)

在这里，你可以直接与开发者和其他用户交流，获取更及时的帮助。

---

## 📖 目录

1.  [项目简介](#-项目简介)
2.  [主要功能](#-主要功能)
3.  [⚠️ 重要提示：测试版本风险须知](#️-重要提示测试版本风险须知)
4.  [🚀 快速开始](#-快速开始)
5.  [🛠️ 安装](#️-安装)
    *   [1. 获取插件](#1-获取插件)
    *   [2. 安装依赖](#2-安装依赖)
    *   [3. Milvus 设置](#3-milvus-设置)
6.  [⚙️ 配置](#️-配置)
7.  [💡 使用方法](#-使用方法)
    *   [记忆的自动运作](#记忆的自动运作)
    *   [指令列表](#指令列表)
8.  [🏗️ 插件架构简介](#️-插件架构简介)
9.  [🎉 更新日志](#-更新日志)
10. [🧩 插件生态推荐](#-插件生态推荐)
11. [🙏 致谢](#-致谢)
12. [🌟 贡献者](#-贡献者)
13. [✨ Star History](#-star-history)

---

## 📜 项目简介

**Mnemosyne (记忆女神)** 是一款为 [AstrBot](https://github.com/lxfater/astrbot) 设计的长期记忆插件。其核心目标是赋予 AstrBot 持久记忆的能力，使其能够记住与用户的历史交互，并在后续对话中运用这些记忆，从而提供更连贯、个性化且智能的聊天体验。

本插件基于 **RAG (Retrieval Augmented Generation)** 技术构建，利用 **Milvus** (或其轻量级版本 **Milvus Lite**) 作为向量数据库来存储和检索记忆片段。通过将对话历史进行总结和向量化，Mnemosyne 能够在用户提问时，高效地检索出最相关的记忆信息，并将其融入到语言模型的提示中，辅助生成更具上下文感知能力的回复。

---

## ✨ 主要功能

*   **长期对话记忆 (Long-term Conversation Memory):** 能够跨越单次会话，持久存储用户与 AI 的交互信息。
*   **RAG 检索增强生成 (RAG - Retrieval Augmented Generation):** 在生成回复前，从记忆库中检索与当前对话最相关的历史记忆，为大型语言模型 (LLM) 提供更丰富的上下文。
*   **自动记忆总结 (Automatic Memory Summarization):**
    *   **事件触发总结:** 当对话达到一定轮次（可配置）后，自动触发对近期对话历史的总结。
    *   **周期性时间触发总结:** 即使用户与 AI 长时间未交互，如果距离上次总结已超过设定时间阈值，且会话中有未总结内容，插件也会在后台检查并触发总结。此功能依赖于现已持久化存储的 `last_summary_time`（每个会话的上次总结时间戳，存储于 `message_counters.db`），确保了在插件重启后依然能够可靠地进行周期性总结。
*   **多种嵌入模型支持 (Support for Multiple Embedding Models):**
    *   直接支持 OpenAI 的 Embedding API。
    *   直接支持 Google Gemini 的 Embedding API。
    *   通过兼容 [astrbot_plugin_embedding_adapter](https://github.com/TheAnyan/astrbot_plugin_embedding_adapter) 插件，可以间接使用该适配器支持的更多嵌入模型，获得更灵活和高质量的文本向量化能力。
*   **Milvus / Milvus Lite 支持 (Milvus / Milvus Lite Support):**
    *   **Milvus Lite:** 内置轻量级向量数据库，无需额外安装和配置 Milvus 服务，适合快速上手和本地开发测试。数据默认存储于 `[AstrBot 数据目录]/mnemosyne_data/mnemosyne_lite.db`。
    *   **Standard Milvus:** 支持连接到标准的、独立部署的 Milvus 服务实例，适合生产环境或对性能、可扩展性有更高要求的场景。
*   **会话管理指令 (Session Management Commands):** 提供了一系列聊天指令，允许用户和管理员查询插件状态、管理 Milvus 集合、查看或删除特定会话的记忆等。
*   **人格特定记忆 (Personality-Specific Memory):** （可选功能）在查询记忆时，可以根据当前 AstrBot 使用的人格 (Persona) 来过滤记忆，使得 AI 在不同人格下展现出与该人格相关的特定记忆。

---

## ⚠️ 重要提示：测试版本风险须知

❗️ **请注意：本插件目前仍处于活跃开发和测试阶段。**

### 1. 功能与数据风险
*   由于插件仍在快速迭代中，新功能的加入、代码重构或与 AstrBot 主程序的兼容性调整，**可能在某些情况下引发系统不稳定或数据处理异常**。
*   当前版本**尚未包含**完善的自动化数据迁移方案。这意味着在进行大版本更新时，**存在丢失历史记忆数据的风险**。

### 2. 使用建议
*   ✅ **强烈建议：** 在更新插件版本前，务必**备份重要数据**，包括但不限于：
    *   插件配置文件 (`_conf_schema.json` 同目录下的用户配置文件，通常是 `Mnemosyne.json` 或类似名称)。
    *   **Mnemosyne 数据目录:** 整个 `[AstrBot 数据目录]/mnemosyne_data/` 目录。此目录包含了：
        *   Milvus Lite 数据库文件 (如果使用 Lite 模式，默认为 `mnemosyne_lite.db`)。
        *   消息计数和上次总结时间数据库文件 (默认为 `message_counters.db`)。
        *(通常，`[AstrBot 数据目录]` 指的是 AstrBot 主程序下的 `data` 目录。例如，如果您的 AstrBot 安装在 `~/astrbot`，那么数据目录通常是 `~/astrbot/data`。插件数据会存放在 `~/astrbot/data/mnemosyne_data/` 下。请根据您的实际 AstrBot 安装路径确认。)*
    *   如果您使用的是标准 Milvus 服务，请参考其官方文档进行数据备份。
*   🧪 **推荐操作：** 如果条件允许，建议先在非生产环境（例如测试用的 AstrBot 实例）中测试新版本，确认无误后再更新到您的主环境。

> 🛡️ **数据安全箴言:**
> *"请像保护重要关系一样重视您的数据安全——毕竟，谁都不希望自己的数字伴侣突然'失忆'。"*

---

## 🚀 快速开始

想要立刻体验 Mnemosyne 的强大记忆力？请查阅我们的快速入门指南：

➡️ **[如何正确且快速地食用本插件 (Wiki)](https://github.com/lxfight/astrbot_plugin_mnemosyne/wiki/%E5%A6%82%E4%BD%95%E6%AD%A3%E7%A1%AE%E4%B8%94%E5%BF%AB%E9%80%9F%E7%9A%84%E9%A3%9F%E7%94%A8%E6%9C%AC%E6%8F%92%E4%BB%B6)** ⬅️

**核心步骤概览:**
1.  **安装插件:** 将插件放置到 AstrBot 的 `plugins` 目录下。
2.  **安装依赖:** 进入插件目录，运行 `pip install -r requirements.txt`。
3.  **配置插件:** 根据您的需求和环境，在 AstrBot UI 中或通过编辑配置文件 (`Mnemosyne.json`) 来设置插件参数（详见 [配置](#️-配置) 部分）。
4.  **重启 AstrBot:** 使插件生效。
5.  **开始对话:** Mnemosyne 会在后台自动工作，开始记忆和学习。

---

## 🛠️ 安装

### 1. 获取插件
   - **手动下载/克隆:**
     将本插件仓库下载或 `git clone` 到您的 AstrBot `plugins` 目录下。确保插件目录名为 `Mnemosyne` (或者您在 AstrBot 中期望的插件标识符，但通常保持与仓库名一致)。
     ```bash
     cd [您的 AstrBot 安装路径]/plugins
     git clone https://github.com/lxfight/astrbot_plugin_mnemosyne.git Mnemosyne
     ```

### 2. 安装依赖
   进入插件目录 (`Mnemosyne`)，并安装所需的 Python 依赖包：
   ```bash
   cd Mnemosyne
   pip install -r requirements.txt
   ```
   主要依赖包括：
   *   `pymilvus`: 用于与 Milvus 向量数据库交互。
   *   `google-genai`: 如果您计划使用 Google Gemini 作为嵌入模型。
   *   (OpenAI SDK 通常会作为 AstrBot 或其他核心依赖的一部分，若提示缺失 `openai` 模块，请自行安装。)

### 3. Milvus 设置

您可以选择以下任一方式作为向量数据的存储后端：

   *   **Milvus Lite (推荐入门):**
      *   这是最简单的方式，无需安装和运行独立的 Milvus 服务。插件会在本地文件系统上创建和管理一个轻量级的 Milvus 数据库。
      *   **数据存储路径:**
          *   Milvus Lite 数据库文件: 默认存储在 `[AstrBot 数据目录]/mnemosyne_data/mnemosyne_lite.db`。
          *   `[AstrBot 数据目录]` 通常指 AstrBot 主程序根目录下的 `data` 文件夹。例如，若 AstrBot 位于 `D:\Apps\AstrBot`，则此文件路径通常为 `D:\Apps\AstrBot\data\mnemosyne_data\mnemosyne_lite.db`。
          *   您可以在插件配置中通过 `milvus_lite_path` 指定自定义路径（详见 [配置](#️-配置) 部分）。
      *   **适用场景:** 个人使用、快速体验、本地开发和测试。
      *   **操作系统兼容性:** Milvus Lite 对操作系统有一定要求 (通常是较新的 Linux 发行版如 Ubuntu >= 20.04, 以及 macOS >= 11.0)。Windows 用户可能需要通过 WSL2 来运行。

   *   **Standard Milvus (标准版，可选):**
      *   如果您需要更强大、可独立部署和管理的向量数据库服务（例如用于生产环境、多人协作或大规模数据），可以选择安装和配置标准的 Milvus 服务。
      *   **安装指南:**
          *   🐧 **Linux (Docker):** [Milvus 独立版 Docker 安装指南](https://milvus.io/docs/zh/install_standalone-docker.md)
          *   💻 **Windows (Docker):** [Milvus 独立版 Windows Docker 安装指南](https://milvus.io/docs/zh/install_standalone-windows.md)
      *   安装完成后，您需要在插件配置中提供 Milvus 服务的地址 (`address`) 和可能的认证信息。

---

## ⚙️ 配置

插件的配置通常通过 AstrBot 的用户界面进行，或者直接编辑位于 AstrBot 配置目录下该插件的 JSON 配置文件 (例如 `Mnemosyne.json`)。以下是主要的配置项说明，这些选项基于插件的 `_conf_schema.json` 文件定义：

| 键 (Key)                             | 描述 (Description)                                                                 | 类型 (Type)    | 默认值 (Default)                                                                                                                                                                                                                                                                                       | 提示 (Hint)                                                                                                                                                                                                                                |
| :----------------------------------- | :--------------------------------------------------------------------------------- | :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLM_providers`                      | 用于执行记忆总结任务的 LLM 服务提供商的 ID。                                             | `string`       | (空字符串)                                                                                                                                                                                                                                                                                         | 必须是 AstrBot 中已注册并有效的 Provider ID。如果留空，总结功能可能无法使用。                                                                                                                                                                   |
| `milvus_lite_path`                   | Milvus Lite 数据库文件的本地存储路径。                                                   | `string`       | (空字符串)                                                                                                                                                                                                                                                                                         | **重要**: 如果配置此项，插件将优先使用 Milvus Lite 模式。如果路径不是以 `.db` 结尾，则会假定为目录，并在其下创建 `mnemosyne_lite.db`。例如：`data/mnemosyne_db/` 或 `data/mnemosyne.db`。留空则依赖 `address` 配置或插件内部计算的默认 Lite 路径 (`[AstrBot 数据目录]/mnemosyne_data/mnemosyne_lite.db`)。 |
| `address`                            | 标准 Milvus 服务的网络地址。                                                           | `string`       | (空字符串)                                                                                                                                                                                                                                                                                         | 仅在 `milvus_lite_path` 未配置时使用。支持格式如: `http://localhost:19530`, `localhost:19530` (默认http和19530端口), `https://secure.milvus.com:19530`。                                                                                |
| `authentication`                     | (对象) 标准 Milvus 服务的认证信息。                                                    | `object`       | (空对象 `{}`)                                                                                                                                                                                                                                                                                       | 仅当连接到需要认证的标准 Milvus 服务时需要填写。                                                                                                                                                                                             |
| `authentication.token`               | Milvus 认证Token (API Key)。                                                         | `string`       | `null`                                                                                                                                                                                                                                                                                             | 如果提供了 Token，将优先于用户名/密码认证。                                                                                                                                                                                                   |
| `authentication.user`                | Milvus 认证用户名。                                                                  | `string`       | `null`                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                        |
| `authentication.password`            | Milvus 认证密码。                                                                    | `string`       | `null`                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                        |
| `summary_check_task`                 | (对象) 后台自动总结计时器相关设置。                                                      | `object`       | (空对象 `{}`)                                                                                                                                                                                                                                                                                       | 用于配置基于时间的自动总结功能。                                                                                                                                                                                                           |
| `summary_check_task.SUMMARY_CHECK_INTERVAL_SECONDS` | 自动总结的检查周期（单位：秒）。                                                       | `int`          | `300` (常量 `DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS`)                                                                                                                                                                                                                                                                          | 插件会每隔这么长时间检查一次是否有会话需要进行基于时间的总结。原默认值为 `60`，现已调整。                                                                                                                                                           |
| `summary_check_task.SUMMARY_TIME_THRESHOLD_SECONDS` | 自动总结的时间阈值（单位：秒）。                                                       | `int`          | `1800` (常量 `DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS`)                                                                                                                                                                                                                                                                         | 如果一个会话自上次总结以来已超过此时间，并且有新消息，则会触发总结。设置为 `-1` 或 `0` (或任何非正数) 可以禁用基于时间的自动总结功能。原默认值为 `-1` (禁用)，现已调整为默认开启。                                                                                               |
| `collection_name`                    | 在 Milvus 中用于存储记忆的集合 (Collection) 的名称。                                       | `string`       | `mnemosyne_default_memory` (常量 `DEFAULT_COLLECTION_NAME`)                                                                                                                                                                                                                                  | 请使用英文、数字和下划线命名。如果通过嵌入适配器加载模型，此名称可能会被适配器提供的值覆盖（通常带有`ea_`前缀）。                                                                                                                                                   |
| `use_personality_filtering`          | 在进行记忆检索时，是否根据当前 AstrBot 的人格 (Persona) 进行过滤。                           | `bool`         | `true`                                                                                                                                                                                                                                                                                             | 如果为 `true`，插件会尝试只检索与当前人格相关的记忆。                                                                                                                                                                                           |
| `num_pairs`                          | 触发记忆自动总结的对话轮次数量。                                                           | `int`          | `10`                                                                                                                                                                                                                                                                                               | 一个“轮次”通常指用户的一次提问和 AI 的一次回答。当一个会话中的消息数量达到此阈值时，插件会尝试进行总结。                                                                                                                                                     |
| `embedding_service`                  | 选择用于文本向量化的嵌入服务提供商。                                                       | `string`       | `openai` (常量 `DEFAULT_EMBEDDING_SERVICE`)                                                                                                                                                                                                                                                            | 可选项: `openai` (适用于 OpenAI官方API及兼容API), `gemini` (适用于 Google Gemini API)。                                                                                                                                                          |
| `embedding_key`                      | 所选嵌入服务提供商的 API 密钥。                                                          | `string`       | (空字符串)                                                                                                                                                                                                                                                                                         | 必须填写，除非通过环境变量 (`OPENAI_API_KEY` 或 `GEMINI_API_KEY`) 提供。                                                                                                                                                                         |
| `embedding_url`                      | (可选) OpenAI 兼容的嵌入 API 的自定义端点 URL。                                            | `string`       | (空字符串，此时使用官方 OpenAI URL `https://api.openai.com/v1`)                                                                                                                                                                                                                                                | 仅当 `embedding_service` 为 `openai` 且您使用的是第三方兼容 API 或 Azure OpenAI 服务时需要填写。例如：`https://your-azure-openai.openai.azure.com/`。请填写到包含 `/v1` 的路径。如果使用 Gemini，此项应留空。                                             |
| `embedding_model`                    | 使用的嵌入模型的具体名称。                                                               | `string`       | (根据 `embedding_service` 自动选择默认模型，如 `text-embedding-3-small` 或 `gemini-embedding-exp-03-07`)                                                                                                                                                                                                 | 示例: OpenAI 的 `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`。Gemini 的 `models/embedding-001`。如果通过嵌入适配器加载，此配置可能会被适配器覆盖。 |
| `embedding_dim`                      | 嵌入向量的维度。                                                                       | `int`          | `768` (常量 `DEFAULT_EMBEDDING_DIM`)                                                                                                                                                                                                                                                                    | **重要**: 此维度必须与您选择的 `embedding_model` 输出的向量维度一致。如果更改模型导致维度变化，强烈建议同时更改 `collection_name` 以避免 Milvus 出现兼容性错误。如果通过嵌入适配器加载，此配置会被适配器提供的值覆盖。例如：OpenAI `text-embedding-3-small` 是 1536 维, `text-embedding-3-large` 是 3072 维; Gemini `models/embedding-001` 是 768 维。 |
| `top_k`                              | 在 RAG 检索时，返回最相关的长期记忆的数量。                                                  | `int`          | `3` (常量 `DEFAULT_TOP_K` 是 5, schema.json 中此默认值可能需要与常量同步)                                                                                                                                                                                                           | 返回的记忆条数越多，LLM 获取的上下文越丰富，但相应消耗的 Token 也越多。请根据您的需求和预算进行调整。                                                                                                                                                           |
| `contexts_memory_len`                | 控制在LLM的上下文中实际保留多少条从数据库中取出的长期记忆。                                      | `int`          | `0`                                                                                                                                                                                                                                                                                                | 如果设置为 `0`，则保留所有 `top_k` 获取到的记忆。如果设置为正整数 `N`，则仅保留最新的 `N` 条记忆。如果为负数，行为可能未明确定义或等同于 `0`（建议使用 `0` 或正整数）。                                                                                              |
| `memory_injection_method`            | 长期记忆内容注入到 LLM 提示词 (Prompt) 中的方式。                                          | `string`       | `user_prompt`                                                                                                                                                                                                                                                                                      | 可选项: `user_prompt` (添加到用户最新消息之前), `system_prompt` (追加到系统提示之后), `insert_system_prompt` (作为一条独立的系统角色消息插入到上下文历史中)。                                                                                             |
| `long_memory_prompt`                 | 用于指导 LLM 进行对话历史总结的提示词模板。                                                  | `string`       | (详见 `_conf_schema.json` 中的默认长提示词，常量 `DEFAULT_LONG_MEMORY_PROMPT`)                                                                                                                                                                                                                           | 您可以根据自己的需求定制此提示词，以获得不同风格或侧重点的记忆总结。                                                                                                                                                                                 |
| `milvus_flush_after_insert`        | 是否在每次向 Milvus 插入数据后立即执行 `flush` 操作。                                      | `bool`         | `false`                                                                                                                                                                                                                                                                                            | `flush` 操作可以确保数据立即可被搜索，但频繁 `flush` 可能对大规模数据写入性能有轻微影响。对于 Milvus Lite，影响通常较小。如果对数据实时性要求不高，可以保持 `false`。                                                                                    |
| `db_name`                          | (标准 Milvus) 要连接的数据库名称。                                                      | `string`       | `default` (常量 `DEFAULT_DB_NAME`)                                                                                                                                                                                                                                                                      |  仅适用于 Milvus 2.2.x 及以上版本，且在连接标准 Milvus 服务时使用。Milvus Lite 通常不直接支持此概念。                                                                                                                                                 |
| `enable_dynamic_field`             | (Milvus) 是否允许集合 Schema 支持动态字段。                                               | `bool`         | `false`                                                                                                                                                                                                                                                                                            | 除非您有特殊需求并了解其含义，否则建议保持 `false`。                                                                                                                                                                                             |
| `index_params`                     | (Milvus, 高级) 向量字段的索引参数。                                                      | `object`       | `{"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}}` (常量 `DEFAULT_INDEX_PARAMS`)                                                                                                                                                                                                         | 通常保持默认即可。`AUTOINDEX` 会根据数据规模自动选择合适的索引。如需手动指定，例如 `{"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}`。更改此项需要深入了解 Milvus 索引机制。                                                     |
| `search_params`                    | (Milvus, 高级) 向量搜索时的参数。                                                      | `object`       | `{"metric_type": "L2", "params": {"nprobe": 10}}` (常量 `DEFAULT_SEARCH_PARAMS`, metric_type 会自动跟随 index_params)                                                                                                                                                                                     | 搜索参数通常需要与索引类型匹配。例如，`IVF_FLAT` 索引对应 `nprobe` 参数。`AUTOINDEX` 通常会忽略此处的 `params`。                                                                                                                                       |

**关于数据持久化:**
*   **记忆内容 (向量和文本):** 存储在您配置的 Milvus 实例 (Lite 或标准版) 中。
*   **消息轮次计数 & 上次总结时间戳:** 存储在插件自动创建的 SQLite 数据库中，路径默认为 `[AstrBot 数据目录]/mnemosyne_data/message_counters.db`。这个文件对于周期性总结功能的正常运作至关重要。

---

## 💡 使用方法

### 记忆的自动运作
Mnemosyne 插件在后台默默工作，您无需进行太多手动干预：
*   **RAG 记忆检索与注入:** 在每次处理用户消息并准备调用 LLM 之前，插件会自动查询相关的历史记忆，并将这些记忆信息注入到发送给 LLM 的请求中。这使得 LLM 能够“看到”之前的对话上下文。
*   **自动总结:**
    *   **基于对话轮次:** 当一个会话中的消息交互达到您在配置中设定的 `num_pairs` 阈值时，插件会自动对这段对话历史进行总结，并将总结存入长期记忆库。
    *   **基于时间:** 即便没有新的对话发生，如果一个会话距离上次总结的时间超过了配置的 `SUMMARY_TIME_THRESHOLD_SECONDS`，并且该会话中有未被计入总结的新消息，插件也会在后台检查并触发总结。

### 指令列表
您可以通过聊天指令与 Mnemosyne 插件进行交互，管理和查看记忆。所有指令都以 `/memory` 开头。

| 指令 (Command)                                  | 描述 (Description)                                     | 用法 (Usage)                                                              | 参数说明 (Parameters)                                                                                                                                        | 权限 (Permissions) | 示例 (Example)                                                                  |
| :---------------------------------------------- | :----------------------------------------------------- | :------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------- | :------------------------------------------------------------------------------ |
| `/memory list`                                  | 列出当前 Milvus 实例中所有的集合 (Collections)。           | `/memory list`                                                            | 无                                                                                                                                                           | 管理员 (Admin)     | `/memory list`                                                                  |
| `/memory drop_collection`                       | **[危险]** 删除指定的 Milvus 集合及其所有数据。          | `/memory drop_collection <集合名称> --confirm`                             | `集合名称` (必填): 要删除的集合的名称。<br>`--confirm` (必填): 执行此危险操作的确认标志 (常量 `CONFIRM_FLAG`)。                                                                               | 管理员 (Admin)     | `/memory drop_collection my_memory_collection --confirm`                        |
| `/memory list_records`                          | 查询指定集合的记忆记录 (按创建时间倒序显示最新的)。        | `/memory list_records [集合名称] [数量上限]`                               | `[集合名称]` (可选): 要查询的集合。如果未提供，则查询当前插件配置的集合。<br>`[数量上限]` (可选): 要显示的记录数量，默认为 5 (常量 `DEFAULT_LIST_RECORDS_LIMIT`)，最大为 50 (常量 `MAX_LIST_RECORDS_LIMIT`)。                                | 管理员 (Admin)     | `/memory list_records my_memories 10`                                       |
| `/memory delete_session_memory`                 | **[危险]** 删除指定会话 ID 相关的所有记忆信息。        | `/memory delete_session_memory <会话ID> --confirm`                       | `会话ID` (必填): 要删除记忆的会话的唯一标识符。<br>`--confirm` (必填): 执行此危险操作的确认标志 (常量 `CONFIRM_FLAG`)。                                                                       | 管理员 (Admin)     | `/memory delete_session_memory user123_conv456 --confirm`                       |
| `/memory reset`                                 | 清除当前用户/群聊会话的所有记忆信息。                    | `/memory reset --confirm`                                                 | `--confirm` (必填): 执行此操作的确认标志 (常量 `CONFIRM_FLAG`)。                                                                                                                            | 成员 (Member)      | `/memory reset --confirm`                                                       |
| `/memory get_session_id`                        | 获取当前对话的会话 ID。                                  | `/memory get_session_id`                                                  | 无                                                                                                                                                           | 成员 (Member)      | `/memory get_session_id`                                                        |

---

## 🏗️ 插件架构简介

Mnemosyne 插件通过以下核心组件协同工作，实现长期记忆功能：

1.  **事件处理钩子 (Event Hooks - `main.py`):**
    *   **`on_llm_request` (RAG流程):** 在 AstrBot 准备向大语言模型 (LLM) 发送请求之前触发。此钩子调用 `MemoryOperations` 模块：
        1.  将当前用户输入添加到短期会话历史 (`ContextManager`)。
        2.  使用嵌入服务 (`EmbeddingAPI`) 将用户输入（或整个对话历史的查询友好形式）转换为向量。
        3.  使用此向量在 Milvus 数据库 (`MilvusManager`) 中搜索最相关的历史记忆片段。
        4.  将检索到的记忆片段格式化后，注入到即将发送给 LLM 的请求的上下文中。
    *   **`on_llm_response` (学习与总结触发):** 在 AstrBot 收到 LLM 的回复后触发。
        1.  将 LLM 的回复添加到短期会话历史 (`ContextManager`)。
        2.  更新消息计数 (`MessageCounter`)。
        3.  调用总结服务 (`SummarizationService`) 检查是否满足基于对话轮次的总结条件。如果满足，则异步启动总结任务。

2.  **核心服务与管理器 (Core Services & Managers):**
    *   **`MemoryOperations` (`core/memory_operations.py`):** 封装 RAG 查询、记忆格式化和注入逻辑。
    *   **`SummarizationService` (`core/summarization_service.py`):**
        *   负责调用 LLM 对话历史进行总结。
        *   将总结结果向量化并存储到 Milvus (`MilvusManager`)。
        *   包含一个后台任务 (`_periodic_summarization_check`)，该任务由 `main.py` 在插件初始化时启动，用于定期检查并触发基于时间的会话总结。
    *   **`MilvusManager` (`memory_manager/vector_db/milvus_manager.py`):** 封装所有与 Milvus 数据库的交互，包括连接管理、Schema 定义、数据增删改查、索引管理等。支持 Milvus Lite 和标准 Milvus。
    *   **`ContextManager` (`memory_manager/context_manager.py`):** 在内存中管理每个会话的短期对话历史、上次总结的时间戳以及相关的 AstrBot 事件对象。与 `MessageCounter` 交互以持久化和加载上次总结时间。
    *   **`MessageCounter` (`memory_manager/message_counter.py`):** 使用 SQLite 数据库持久化存储每个会话的消息轮次计数和上次总结的时间戳。这确保了即使插件重启，基于时间和轮次的总结逻辑也能正确恢复。数据文件默认位于 `[AstrBot 数据目录]/mnemosyne_data/message_counters.db`。
    *   **`EmbeddingAPI` (`memory_manager/embedding.py`):** 提供了对 OpenAI 和 Gemini 嵌入服务的封装，用于将文本转换为向量表示。

3.  **配置与初始化 (`core/initialization.py`, `core/constants.py`):**
    *   负责加载和校验插件配置，定义 Milvus Schema，初始化所有核心组件和服务。`core/constants.py` 文件集中管理了插件中使用的各种常量、默认值和配置键名。

4.  **聊天指令处理 (`core/commands.py`):**
    *   实现了用户和管理员可通过聊天界面与插件交互的指令后端逻辑。

5.  **工具函数 (`core/tools.py`):**
    *   提供了一些通用的辅助函数，如地址解析、记忆标签移除、上下文格式化等。

**数据流简述:**
*   **记忆存储 (学习):** 用户对话 -> `ContextManager` (短期历史) -> (达到条件) -> `SummarizationService` -> LLM (总结) -> `EmbeddingAPI` (向量化) -> `MilvusManager` (存入 Milvus)。
*   **记忆检索 (回忆):** 用户新消息 -> `EmbeddingAPI` (向量化查询) -> `MilvusManager` (Milvus 搜索) -> `MemoryOperations` (格式化并注入上下文) -> LLM (生成回复)。

这种模块化的架构旨在提高代码的可维护性、可读性和可扩展性。

---

## 🎉 更新日志

我们持续改进，以下是本插件的近期更新亮点和重要里程碑：

### 🚀 v0.5.1 (内部重构与文档更新)
*   ✨ **大规模内部重构 (Major Internal Refactoring):** 对插件的整体代码结构进行了深度优化和重构，包括模块拆分 (例如将总结逻辑移至 `SummarizationService`)、依赖关系梳理、以及代码风格统一，显著提升了代码的可读性、可维护性和可扩展性。
*   🐛 **错误处理增强 (Enhanced Error Handling):** 全面审查并加强了整个插件的错误捕获和处理机制，特别是在与外部服务（如 Milvus、LLM、嵌入API）交互以及数据库操作中，确保了更细致的异常处理和更明确的日志记录 (`exc_info=True`)。
*   💾 **会话总结时间持久化 (Persistent Session Summary Time):** `last_summary_time` (每个会话的上次总结时间戳) 现在通过 `MessageCounter` 组件持久化存储在 SQLite 数据库中 (`message_counters.db`)。这使得基于时间的周期性自动总结功能在插件重启后依然能够可靠地恢复和执行。
*   🧹 **依赖与配置清理 (Dependency & Config Cleanup):**
    *   移除了未实际使用的 `pypinyin` 依赖。
    *   对 `core/constants.py` 进行了大幅扩展和整理，集中管理了更多的配置键名、默认值、日志名、数据库表/列名等，减少了代码中的硬编码值。
*   📄 **README 更新 (README Update):** 全面更新和美化了 `README.md` 文档，提供了更详细的安装指南、配置说明（基于 `_conf_schema.json`）、功能介绍、架构简介以及更新的指令用法，使其更符合当前插件状态并对用户更友好。
*   ✍️ **代码规范与注释 (Code Style & Comments):** 整体提升了代码注释的覆盖率和质量，所有主要函数和类都添加了中文文档字符串（Docstrings），并对复杂逻辑增加了内联注释。对类型提示 (Type Hinting) 进行了全面检查和补充。

### 🚀 v0.5.0
*   🔗 **生态兼容:** 增加了对 [astrbot_plugin_embedding_adapter](https://github.com/TheAnyan/astrbot_plugin_embedding_adapter) 插件的兼容支持，现在可以与该插件联动，获取更优质的 embedding 效果。特别感谢 [@TheAnyan](https://github.com/TheAnyan) 的贡献！
*   ⚡️ **优化与修复:** 进行了多项内部优化，并修复了若干已知问题，提升了整体稳定性和用户体验。
*   ⚖️ **协议更新:** 插件的开源协议已进行变更，请查阅项目根目录下的 `LICENSE` 文件以获取详情。

### 🚀 v0.4.1
*   🐛 **Bug 修复:** 修复了在某些特定环境（如 pymilvus 2.5.4）下，处理 Milvus 搜索结果可能引发的 `TypeError: 'SequenceIterator' object is not iterable` 问题。特别感谢 [@HikariFroya](https://github.com/HikariFroya) 发现并贡献了解决方案！
*   ✨ **指令优化:** 简化了 `/memory list_records` 指令的使用，使其更专注于查询最新的记忆记录。
    *   命令格式变更为：`/memory list_records [collection_name] [limit]`，**移除了 `offset` 参数**。
    *   现在，您只需指定需要查看的记录数量 (`limit`)，系统将自动获取符合条件的所有记录（在安全上限内），并从中选取最新的几条按时间倒序显示，无需再手动计算偏移量，大大提升了便捷性。
*   ✨ **模型支持:** 嵌入模型现在增加了对 Google Gemini 嵌入模型的支持。感谢 [@Yxiguan](https://github.com/Yxiguan) 提供的关键代码！

<details>
<summary><strong>📜 更早历史版本回顾 (v0.4.0 及更早)</strong></summary>

### 🚀 v0.4.0
*   ✨ **核心新功能: 基于时间的自动总结**:
    *   插件内部集成计时器，当用户和BOT之间的消息长时间未被总结时（即使没有新的互动），系统将自动触发对先前历史消息的总结，有效减少手动总结的频率和遗漏。
*   ⚙️ **新增配置项**: 引入了用于自定义计时器间隔时间 (`SUMMARY_CHECK_INTERVAL_SECONDS`) 和总结阈值时间 (`SUMMARY_TIME_THRESHOLD_SECONDS`) 的配置项，用户可根据需求灵活调整自动总结行为。
*   🛠️ **架构优化**: 重构了上下文管理器，优化了会话历史的存储和获取逻辑，显著提升了效率和稳定性。
*   🏗️ **后台任务**: 在主程序中完善了后台自动总结检查任务的启动与停止逻辑，确保该功能稳定可靠运行。

### 🚀 v0.3.14
*   🐛 **关键修复:** 解决了 v0.3.13 版本中导致数据插入失败的重大问题。**强烈建议所有用户更新至此版本以确保插件正常运行！**

### 🚀 v0.3.13
*   ✨ **新功能:** 新增 `Milvus Lite` 支持！现在可以在本地运行轻量级向量数据库，无需复杂部署完整的 Milvus 服务，极大简化了入门门槛和本地开发体验。（特别感谢提出此建议的社区群友！）
*   ⚠️ **重要提示:** `Milvus Lite` 目前仅支持 `Ubuntu >= 20.04` 和 `MacOS >= 11.0` 操作系统环境。

### 📜 v0.3.12 及更早版本 (主要优化与修复)
*   ✅ **核心修复:** 包含了多个关键 Bug 修复、紧急问题处理和指令逻辑修正，提升了插件的稳定性和健壮性。
*   🔧 **性能与逻辑优化:** 对会话历史检查、异步IO处理等核心模块进行了优化，有效提升了运行效率和响应速度。
*   ⚙️ **配置与功能完善:** 更新了配置架构以支持更多自定义选项，并恢复或优化了部分早期版本的功能设定，以满足更多使用场景的需求。

*此范围内包含了多次迭代的更新内容，上述为主要类别总结。如需查看更详细的历史更新日志，请查阅项目的 Release Notes 或 Git Commit 历史记录。*
</details>

---

## 🧩 插件生态推荐：优化 DeepSeek API 体验

**1. 本插件 (Mnemosyne v0.3+ 系列) 🚀**

*   Mnemosyne 插件自 `v0.3` 系列起，集成了由开发者 **[Rail1bc](https://github.com/Rail1bc)** 贡献的关键优化代码。
*   **核心优势**: 此优化专门针对 DeepSeek 官方 API 的缓存机制。通过智能调整发送给 API 的历史对话内容，能够**显著提高缓存命中率**。这意味着您可以更频繁地复用之前的计算结果，有效**降低 Token 消耗量** 💰，为您节省 API 调用成本。

**2. 堆肥桶 (Composting Bucket) 插件 ♻️**

*   除了对 Mnemosyne 的贡献，开发者 **Rail1bc** 还独立开发了一款名为 **“堆肥桶” (Composting Bucket)** 的 AstrBot 插件。
*   **主要功能**: 该插件专注于提升 DeepSeek API 的缓存利用效率。即使您不使用 Mnemosyne 的记忆功能，也可以将“堆肥桶”作为一个独立的增强工具，进一步优化缓存表现，减少不必要的 Token 开销。（“堆肥桶”对用户体验的影响较小，主要在后台优化）
*   **项目地址**: 感兴趣的用户可以访问了解详情：
    🔗 **[astrbot_plugin_composting_bucket on GitHub](https://github.com/Rail1bc/astrbot_plugin_composting_bucket)**

> ✨ 如果您是 DeepSeek API 用户，强烈推荐关注并尝试由 **Rail1bc** 带来的这些优秀工具，让您的 AI 体验更经济、更高效！

---

## 🙏 致谢

*   感谢 **AstrBot 核心开发团队** 提供的强大平台和技术支持。
*   感谢 **[Rail1bc](https://github.com/Rail1bc)** 对 DeepSeek API 优化提供的关键代码贡献。
*   感谢所有在 QQ 群和 GitHub Issues 中提出宝贵意见和反馈的用户。

**如果本项目给您带来了帮助或乐趣，请不吝点亮 Star ⭐ ！您的支持是我持续开发和改进的最大动力！**

发现 Bug？有好点子？请随时通过 [GitHub Issues](https://github.com/lxfight/astrbot_plugin_mnemosyne/issues) 告诉我们。每一条反馈我们都会认真对待。

---

## 🌟 贡献者

感谢所有为 Mnemosyne 项目做出贡献的朋友们！

[![GitHub Contributors](https://img.shields.io/github/contributors/lxfight/astrbot_plugin_mnemosyne?style=flat-square)](https://github.com/lxfight/astrbot_plugin_mnemosyne/graphs/contributors)

<a href="https://github.com/lxfight/astrbot_plugin_mnemosyne/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lxfight/astrbot_plugin_mnemosyne" alt="Contributor List" />
</a>

---

## ✨ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lxfight/astrbot_plugin_mnemosyne)](https://github.com/lxfight/astrbot_plugin_mnemosyne)

_每一个 Star 都是我们前进的灯塔！感谢您的关注！_