# AgentOffice

> 智能 Agent 办公助手，基于 LangGraph 编排引擎，支持多工具调用、RAG 知识库和实时对话。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
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
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

AgentOffice 是一个 AI Agent 系统。它结合了 **LangGraph 驱动的 Agent 编排引擎**、**FastAPI 后端**、**React 前端**以及模块化的工具生态，提供智能任务自动化、文档分析和知识检索能力。

Agent 遵循完整的推理闭环：**任务理解 → 记忆调取 → 边界校验 → 任务规划 → 工具调用 → 行动生成 → 反思复盘 → 记忆归档**。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    React SPA (Port 3000)                  │
│              Chat UI · Admin Dashboard                   │
└──────────────────────────┬──────────────────────────────┘
                           │ API / Proxy
┌──────────────────────────▼──────────────────────────────┐
│              FastAPI Backend (Port 8000)                  │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Auth/Users │ │  Chat    │ │  Admin   │ │  Tools   │ │
│  └────────────┘ └──────────┘ └──────────┘ └──────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Agent 编排引擎                                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  LangGraph StateGraph                                │ │
│  │  ┌─────┐ ┌──────┐ ┌──────┐ ┌────┐ ┌────────┐       │ │
│  │  │理解 │ │记忆  │ │规划  │ │工具│ │反思   │──▶ ... │ │
│  │  └─────┘ └──────┘ └──────┘ └────┘ └────────┘       │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌───────────┐ ┌──────────┐ ┌────────────┐              │
│  │   工具层   │ │  记忆层   │ │   LLM 层    │              │
│  │ Search ·  │ │ Vector · │ │ OpenAI ·   │              │
│  │ Code ·    │ │ Chat     │ │ DeepSeek · │              │
│  │ File · KB │ │          │ │ Qwen       │              │
│  └───────────┘ └──────────┘ └────────────┘              │
└──────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    数据层                                  │
│       MySQL · FAISS 向量存储 · 本地文件系统                │
└─────────────────────────────────────────────────────────┘
```

---

## 功能特性

- **🤖 智能 Agent** — LangGraph 驱动的任务规划、工具执行与自我反思闭环
- **🔧 多工具生态** — 内置搜索、代码执行、文件操作、网页抓取和知识库检索工具
- **📚 RAG 知识库** — 基于向量的文档检索，支持相似度阈值过滤和文档分类
- **💬 对话式 UI** — React SPA，支持实时流式对话、会话管理和深色模式
- **🔐 认证与权限** — JWT 认证 + 管理员/用户角色权限控制
- **🧠 持久化记忆** — 滑动窗口聊天历史 + 向量记忆用于长期经验归档
- **🔗 MCP 协议支持** — 通过 Model Context Protocol 集成远程工具
- **📊 管理后台** — 用户管理、系统监控、知识库管理和链路追踪
- **🐳 Docker 部署** — 一行命令启动全部服务

---

## 技术栈

### 后端

| 组件 | 技术 |
|-----------|-----------|
| 框架 | [FastAPI](https://fastapi.tiangolo.com/) |
| Agent 引擎 | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| ORM | [SQLAlchemy 2.0](https://www.sqlalchemy.org/) |
| 数据库 | [MySQL 8.0](https://www.mysql.com/) |
| 向量存储 | [FAISS](https://github.com/facebookresearch/faiss) |
| 认证 | JWT (python-jose) + bcrypt |

### 前端

| 组件 | 技术 |
|-----------|-----------|
| 框架 | [React 18](https://react.dev) |
| 构建工具 | [Vite](https://vitejs.dev) |
| UI 库 | [Ant Design](https://ant.design) |
| 状态管理 | [Zustand](https://github.com/pmndrs/zustand) |
| 路由 | [React Router 6](https://reactrouter.com) |

### LLM 支持

- OpenAI (GPT-4o, GPT-4o-mini)
- DeepSeek
- 通义千问 (Qwen)
- 本地规则引擎兜底（无需 API 密钥）

---

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+（前端开发需要）
- MySQL 8.0（或使用 Docker 运行数据库）

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/AgentOffice.git
cd AgentOffice
```

### 2. 后端安装

```bash
# 创建并激活虚拟环境
conda create -n agentoffice python=3.12
conda activate agentoffice

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入数据库和 LLM 配置（参见下方配置说明）
```

### 3. 初始化数据库

```bash
# 方式一：手动创建 MySQL 数据库
mysql -u root -p < database/init_mysql.sql

# 方式二：使用 Docker 启动 MySQL
docker compose up -d mysql
```

### 4. 启动应用

```powershell
# 启动后端
& "你的Python路径\python.exe" main.py

# 另开终端 — 启动前端（开发模式）
cd frontend
npm install
npm run dev
```

### 5. 访问服务

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
APP_RELOAD=false

# 数据库
DATABASE_URL=mysql+pymysql://agentoffice:agentoffice123@127.0.0.1:3306/agentoffice?charset=utf8mb4

# LLM 供应商（选其一）
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

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
| OpenAI | `openai` | `OPENAI_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| 通义千问 | `qwen` | `QWEN_API_KEY` |

> 未配置 API 密钥时，系统会自动回退到本地规则引擎，可处理分类、简单问答、时间查询、计算等基础任务，无需任何外部依赖。

---

## 项目结构

```
AgentOffice/
├── main.py                          # 应用入口，服务启动
├── app.py                           # FastAPI 应用工厂
│
├── agent/                           # Agent 编排引擎
│   ├── state.py                     #  LangGraph 状态定义
│   ├── nodes.py                     #  Agent 节点实现
│   └── graph.py                     #  LangGraph StateGraph 流水线
│
├── api/                             # REST API 路由
│   ├── route.py                     #  聊天与工具接口
│   ├── auth_route.py                #  认证相关接口
│   └── admin_route.py               #  管理后台接口
│
├── tools/                           # 工具层实现
│   ├── base.py                      #  工具基类与注册表
│   ├── search_tool.py               #  网络搜索
│   ├── code_tool.py                 #  Python 代码执行
│   ├── file_tool.py                 #  文件读写
│   ├── browser_tool.py              #  网页抓取
│   └── knowledge_tool.py            #  RAG 知识检索
│
├── memory/                          # 记忆模块
│   ├── chat_memory.py               #  聊天历史（滑动窗口）
│   └── vector_memory.py             #  向量长期记忆
│
├── services/                        # 业务服务层
│   ├── llm_service.py               #  LLM 客户端（多供应商）
│   ├── chat_service.py              #  聊天业务编排
│   └── knowledge_service.py         #  知识库服务
│
├── frontend/                        # React SPA
│   ├── src/
│   │   ├── pages/                   #  页面组件
│   │   ├── components/              #  共享组件
│   │   ├── stores/                  #  Zustand 状态管理
│   │   ├── api/                     #  API 调用封装
│   │   └── styles/                  #  全局样式
│   └── vite.config.ts
│
├── database/                        # 数据库层
│   ├── db.py                        #  会话与引擎配置
│   ├── tables.py                    #  SQLAlchemy 模型
│   ├── init_mysql.sql               #  数据库初始化脚本
│   └── migrate.sql                  #  迁移脚本
│
├── config/                          # 全局配置
│   └── settings.py                  #  Pydantic 配置类
│
├── utils/                           # 通用工具
│   ├── auth.py                      #  JWT、密码哈希
│   ├── exception.py                 #  错误码与异常处理
│   ├── logger.py                    #  日志配置
│   └── common.py                    #  通用函数
│
├── schemas/                         #  Pydantic 数据模型
├── scripts/                         #  启停与诊断脚本
├── static/                          #  静态资源
├── data/                            #  本地数据与上传文件
├── logs/                            #  日志文件
│
├── tests/                           #  测试用例
│   ├── unit/                        #  单元测试
│   └── integration/                 #  集成测试
│
├── requirements.txt
│
├── docker-compose.yml                 # Docker 编排
└── .env.example
```

---

## API 文档

### 认证

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/auth/register` | 用户注册 |
| GET | `/api/auth/me` | 获取当前用户信息 |
| PUT | `/api/auth/profile` | 更新个人资料 |

### 聊天

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| POST | `/api/chat/completions` | 发送消息并获取回复 |
| GET | `/api/chat/history` | 按会话获取聊天历史 |
| GET | `/api/chat/sessions` | 获取所有会话列表 |

### 工具

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| GET | `/api/tool/list` | 获取可用工具列表 |

### 管理

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| GET | `/api/admin/dashboard` | 仪表盘统计数据 |
| GET | `/api/admin/users` | 用户管理列表 |
| PUT | `/api/admin/users/{id}` | 更新用户信息 |
| DELETE | `/api/admin/users/{id}` | 删除用户 |

### 知识库

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| POST | `/api/knowledge/search` | 搜索知识库 |

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

Docker Compose 会启动：
- **MySQL 8.0** — 数据库服务
- **AgentOffice API** — FastAPI 应用（端口 8000）

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 发起 Pull Request

---

## 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。
