# AgentOffice

> 智能 Agent 办公助手，基于 LangGraph 编排引擎，支持多工具调用、RAG 知识库、三层记忆体系和实时对话。

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-1c3c3c.svg)](https://langchain-ai.github.io/langgraph/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [API 文档](#api-文档)
- [Docker 部署](#docker-部署)
- [测试](#测试)
- [许可证](#许可证)

---

## 项目简介

AgentOffice 是一个企业级 AI Agent 系统。它结合了 **LangGraph 驱动的 Agent 编排引擎**、**FastAPI 后端**、**React 前端**以及模块化工具生态，提供智能任务自动化、文档分析和知识检索能力。

Agent 遵循完整的推理闭环：**记忆调取 → 意图理解 → 任务规划 → 工具调用 → 回答生成 → 记忆归档**。

---

## 系统架构

```
┌─ 客户端 ──────────────────────────────────────────┐
│  React SPA / API Client                           │
└──────────────────────┬────────────────────────────┘
                       │ HTTP / SSE
┌─ API 层 ─────────────┴────────────────────────────┐
│  FastAPI (main.py)                                 │
│  路由: /api/chat, /api/knowledge, /api/auth ...     │
└──────────────────────┬────────────────────────────┘
                       │
┌─ 服务层 ─────────────┴────────────────────────────┐
│  ChatService      对话编排                         │
│  ├─ ChatMemory  (短期滑动窗口)                     │
│  ├─ RedisKV     (会话/摘要缓存)                    │
│  └─ AgentGraph  (LangGraph 工作流)                 │
│                                                    │
│  LLMService      模型抽象                          │
│  ├─ LocalModelClient (规则兜底)                    │
│  └─ OpenAICompatible (Qwen/DeepSeek/OpenAI)        │
│                                                    │
│  ToolService     工具注册                          │
│  ├─ 7个内置工具 + MCP 远程工具                     │
│  └─ 跨工具依赖注入 (weather → email)               │
│                                                    │
│  KnowledgeService  RAG 文档处理                    │
│  └─ Milvus 向量检索                                │
└──────────────────────┬────────────────────────────┘
                       │
┌─ Agent 层 ───────────┴────────────────────────────┐
│  LangGraph StateGraph (6 节点)                     │
│                                                    │
│  mem_pre     检索 Milvus 长期记忆                  │
│    ↓                                              │
│  understand  三层意图识别 (关键词→向量→LLM)       │
│    ↓                                              │
│  planning    生成工具执行计划                      │
│    ↓        ↘                                      │
│  tool        顺序执行工具链 + 参数自动注入         │
│    ↓                                              │
│  action      融合记忆与工具结果生成回答            │
│    ↓                                              │
│  mem_post    归档本轮执行记录到 Milvus             │
└──────────────────────┬────────────────────────────┘
                       │
┌─ 存储层 ─────────────┴────────────────────────────┐
│  内存    ChatMemory    (对话滑动窗口 20条)          │
│  Redis   会话/LLM/摘要缓存                         │
│  Milvus  agent_memory + knowledge (向量)           │
│  MySQL   ChatSession/ChatRecord/ToolRecord          │
└───────────────────────────────────────────────────┘
```

---

## 功能特性

- **🤖 智能 Agent** — LangGraph 驱动的 6 节点工作流：记忆 → 理解 → 规划 → 工具 → 行动 → 归档
- **🔧 多工具生态** — 内置天气、邮件、代码、文件、时间、知识库、浏览器 7 种工具，支持跨工具依赖注入
- **🧠 三层记忆体系** — 短期对话滑动窗口 + Milvus 向量长期记忆 + Redis 缓存（会话/LLM/摘要）
- **🎯 三层意图识别** — 关键词快速匹配(L1) → 哈希向量相似度(L2) → LLM 深度推理(L3)，平衡速度与准确率
- **📚 RAG 知识库** — 基于 Milvus 的文档向量检索，支持相似度阈值过滤和文档分类
- **💬 对话式 UI** — React SPA，支持 SSE 流式对话、会话管理
- **🔗 MCP 协议支持** — 通过 Model Context Protocol 自动发现并集成远程工具
- **🔐 认证与权限** — JWT 认证 + 用户权限控制
- **🐳 Docker 部署** — 一键启动 MySQL + Redis + Agent 服务
- **📊 结构化日志** — 完整的 Agent 执行链路追踪

---

## 技术栈

### 后端

| 组件 | 技术 |
|-----------|-----------|
| 框架 | [FastAPI](https://fastapi.tiangolo.com/) |
| Agent 引擎 | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| ORM | [SQLAlchemy 2.0](https://www.sqlalchemy.org/) |
| 数据库 | [MySQL 8.0](https://www.mysql.com/) |
| 向量存储 | [Milvus Lite](https://milvus.io/) / [Milvus](https://milvus.io/) |
| 缓存 | [Redis 7](https://redis.io/) |
| 认证 | JWT + bcrypt |

### 前端

| 组件 | 技术 |
|-----------|-----------|
| 框架 | [React 18](https://react.dev) |
| 构建工具 | [Vite](https://vitejs.dev) |
| UI 库 | [Ant Design](https://ant.design) |
| 状态管理 | [Zustand](https://github.com/pmndrs/zustand) |
| 路由 | [React Router 6](https://reactrouter.com) |

### LLM 支持

- 通义千问 (Qwen)
- OpenAI (GPT-4o)
- DeepSeek
- 本地规则引擎兜底（无需 API 密钥）

---

## 快速开始

### 环境要求

- Python 3.12+
- Node.js 18+（前端开发需要）
- Docker（推荐）或 MySQL 8.0 + Redis 7

### 方式一：Docker 一键启动（推荐）

```bash
# 克隆项目
git clone https://github.com/yourusername/AgentOffice.git
cd AgentOffice

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 LLM API Key 等配置

# 启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f
```

访问 http://localhost:8000/docs 查看 API 文档。

### 方式二：本地开发

```bash
# 创建并激活虚拟环境
conda create -n agentoffice python=3.12
conda activate agentoffice

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入数据库和 LLM 配置

# 启动 MySQL 和 Redis（可用 Docker）
docker compose up -d mysql redis

# 启动后端
cd backend
python main.py

# 另开终端 — 启动前端
cd frontend
npm install
npm run dev
```

### 访问服务

| 服务 | 地址 |
|---------|-----|
| Web 界面 | http://localhost:3000 |
| API 文档 | http://localhost:8000/docs |
| API 直连 | http://localhost:8000/api |

---

## 配置说明

### 环境变量

核心配置项（`.env` 文件）：

```env
# 应用配置
APP_ENV=production
APP_PORT=8000

# 数据库
DATABASE_URL=mysql+pymysql://agentoffice:agentoffice123@127.0.0.1:3306/agentoffice?charset=utf8mb4

# Redis（可选，不可用时系统自动降级）
REDIS_URL=redis://localhost:6379/0

# LLM 供应商（选其一）
MODEL_PROVIDER=qwen
QWEN_API_KEY=sk-...
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.6-plus

# SMTP 邮件配置
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=

# RAG 配置
KNOWLEDGE_SIMILARITY_THRESHOLD=0.12
AGENT_MEMORY_SIMILARITY_THRESHOLD=0.08

# MCP 配置（可选）
MCP_HTTP_ENDPOINT=
MCP_API_KEY=
```

### LLM 供应商

| 供应商 | `MODEL_PROVIDER` | 所需密钥 |
|----------|-----------------|---------------|
| 通义千问 | `qwen` | `QWEN_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |

> 未配置 API 密钥时，系统会自动回退到本地规则引擎，可处理天气、时间、计算等基础任务，无需任何外部依赖。

---

## 项目结构

```
AgentOffice/
├── backend/                          # FastAPI 后端服务
│   ├── main.py                       # 应用入口
│   ├── app.py                        # FastAPI 应用工厂
│   │
│   ├── agent/                        # Agent 编排引擎
│   │   ├── state.py                  # LangGraph AgentState 定义
│   │   ├── nodes.py                  # 6 个 Agent 节点实现
│   │   └── graph.py                  # LangGraph StateGraph 流水线
│   │
│   ├── api/                          # REST API 路由
│   │   ├── route.py                  # 聊天与工具接口
│   │   ├── auth_route.py             # 认证相关接口
│   │   └── admin_route.py            # 管理后台接口
│   │
│   ├── services/                     # 业务服务层
│   │   ├── llm_service.py            # LLM 客户端（Qwen/OpenAI/DeepSeek）
│   │   ├── chat_service.py           # 聊天业务编排
│   │   ├── knowledge_service.py      # 知识库 RAG 服务
│   │   ├── tool_service.py           # 工具注册与调度
│   │   └── mcp_service.py            # MCP 协议服务
│   │
│   ├── tools/                        # 工具实现
│   │   ├── base.py                   # 工具基类与 ToolRegistry
│   │   ├── weather_tool.py           # 天气查询（Open-Meteo API）
│   │   ├── email_tool.py             # 邮件发送（SMTP）
│   │   ├── code_tool.py              # Python 代码执行
│   │   ├── file_tool.py              # 文件读取
│   │   ├── browser_tool.py           # 网页抓取
│   │   ├── knowledge_tool.py         # RAG 知识检索
│   │   └── time_tool.py              # 时间查询
│   │
│   ├── memory/                       # 三层记忆模块
│   │   ├── chat_memory.py            # 短期记忆（滑动窗口 + 自动压缩）
│   │   ├── vector_memory.py          # Milvus 向量记忆 + RedisKV
│   │   ├── memory_context.py         # 统一记忆上下文
│   │   └── store.py                  # 单例记忆存储适配器
│   │
│   ├── database/                     # 数据库层
│   │   ├── db.py                     # 会话与引擎配置
│   │   ├── tables.py                 # SQLAlchemy 模型
│   │   └── init_mysql.sql            # 数据库初始化脚本
│   │
│   ├── config/                       # 全局配置
│   │   └── settings.py               # Pydantic 配置类
│   │
│   ├── utils/                        # 通用工具
│   │   ├── auth.py                   # JWT、密码哈希
│   │   ├── exception.py              # 错误码与异常处理
│   │   ├── common.py                 # 通用函数
│   │   ├── document_classifier.py    # 文档分类器
│   │   └── structured_log.py         # 结构化日志
│   │
│   ├── schemas/                      # Pydantic 数据模型
│   │   ├── chat.py                   # 聊天数据模型
│   │   ├── knowledge.py              # 知识库数据模型
│   │   └── tool.py                   # 工具数据模型
│   │
│   ├── integrations/                 # 外部集成
│   │   └── mcp_client.py             # MCP HTTP 客户端
│   │
│   ├── static/                       # 静态资源
│   │
│   └── tests/                        # 测试用例
│       ├── conftest.py               # 测试配置
│       ├── unit/                     # 单元测试（170+）
│       └── integration/              # 集成测试
│
├── frontend/                         # React SPA
│   ├── src/
│   │   ├── pages/                    # 页面组件
│   │   ├── components/               # 共享组件
│   │   ├── stores/                   # Zustand 状态管理
│   │   ├── api/                      # API 调用封装
│   │   └── styles/                   # 全局样式
│   └── vite.config.ts
│
├── data/                             # 本地数据（Docker volume）
│   ├── uploads/                      # 上传文件
│   └── vector_store/                 # Milvus Lite 向量库
├── logs/                             # 日志文件
├── Dockerfile                        # Docker 构建文件
├── requirements.txt                  # Python 依赖
├── docker-compose.yml                # Docker 编排（MySQL+Redis+App）
├── .env.example                      # 环境变量模板
└── .gitignore                        # Git 忽略规则
```

---

## API 文档

### 认证

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/auth/register` | 用户注册 |
| GET | `/api/auth/me` | 获取当前用户信息 |

### 聊天

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| POST | `/api/chat/completions` | 发送消息并获取回复 |
| GET | `/api/chat/stream` | SSE 流式聊天 |
| GET | `/api/chat/history` | 获取聊天历史 |
| GET | `/api/chat/sessions` | 获取会话列表 |
| PUT | `/api/chat/sessions/{id}` | 重命名会话 |
| DELETE | `/api/chat/sessions/{id}` | 删除会话 |

### 工具

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| GET | `/api/tool/list` | 获取可用工具列表 |

### 知识库

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| POST | `/api/knowledge/search` | 搜索知识库 |
| POST | `/api/knowledge/upload` | 上传文档（PDF/TXT/DOCX） |

> 服务运行时可访问 `/docs` 查看交互式 API 文档（Swagger UI）。

---

## Docker 部署

```bash
# 启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

Docker Compose 启动的服务：

| 容器 | 镜像 | 端口 |
|-----------|-------|---------|
| agent-office | 自构建 (python:3.12-slim) | 8000 |
| agent-office-mysql | mysql:8.0 | 3307 → 3306 |
| agent-office-redis | redis:7-alpine | 6379 |

Redis 不可用时系统自动降级，不影响核心功能。

---

## 测试

```bash
# 运行所有单元测试
cd backend
python -m pytest tests/unit/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 运行全部测试
python -m pytest tests/ -v
```

---

## 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。
