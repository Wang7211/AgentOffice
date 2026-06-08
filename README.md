# AgentOffice

AgentOffice 是一个面向企业办公场景的智能 Agent 系统。项目提供对话式任务处理、知识库检索、工具调用、链路追踪、后台管理、用户认证和用户级数据隔离，适合作为企业 Agent 原型继续扩展。

## 功能概览

- 对话 Agent：基于 LangGraph 编排“记忆加载 -> 意图识别 -> 任务规划 -> 工具执行 -> 观察 -> 回答生成 -> 记忆归档”流程。
- 真正执行器：planner 不只是顺序调用工具，支持多步骤计划、依赖检查、失败跳过、最大步数保护和工具结果观察。
- 记忆系统：区分短期对话记忆、长期语义记忆和执行事件记忆，避免把普通聊天或失败工具结果当成可复用事实。
- 工具协议：工具声明 `required_permissions` 和 `context_schema`，执行器统一传入用户、会话和权限上下文。
- 用户隔离：聊天记录、知识库检索、Agent 记忆和管理接口均按当前用户隔离。
- RAG 知识库：支持 PDF、TXT、DOCX 上传、分片、向量检索和文档类别过滤。
- 内置工具：天气、邮件、计算、文件解析、时间、知识库检索、网页读取。
- MCP 扩展：可通过 MCP HTTP 端点发现并注册远程工具。
- 后台管理：提供总览、知识库、链路追踪、用户管理和系统配置页面。

## 技术栈

### 后端

| 模块 | 技术 |
| --- | --- |
| Web 框架 | FastAPI |
| Agent 编排 | LangGraph |
| ORM | SQLAlchemy 2.x |
| 数据库 | MySQL 8 |
| 缓存 | Redis 7 |
| 向量存储 | Milvus |
| 认证 | JWT + bcrypt |
| 工具适配 | 内置工具 + MCP HTTP |

### 前端

| 模块 | 技术 |
| --- | --- |
| 框架 | React 18 |
| 构建 | Vite |
| UI | Ant Design 5 |
| 状态管理 | Zustand |
| 路由 | React Router 6 |

## Agent 流程

```text
用户输入
  -> ChatService 恢复会话历史和 Redis 摘要
  -> mem_pre 加载当前用户可复用的长期记忆
  -> understand 识别意图并规范化任务
  -> planning 生成工具计划
  -> tool 执行一个可运行工具步骤
  -> observe 更新计划状态、聚合成功/失败结果
  -> tool / action 根据依赖和剩余步骤继续执行或生成回答
  -> mem_post 归档语义记忆或执行事件记忆
```

## 项目结构

```text
AgentOffice/
├── backend/
│   ├── agent/                 # LangGraph 状态、节点和图
│   ├── api/                   # 聊天、认证、管理、知识库接口
│   ├── config/                # 环境配置
│   ├── database/              # SQLAlchemy 连接和表模型
│   ├── integrations/          # MCP 客户端
│   ├── memory/                # 短期记忆、向量记忆、Redis KV
│   ├── schemas/               # Pydantic 数据模型
│   ├── services/              # Chat/LLM/Knowledge/Tool/MCP 服务
│   ├── tools/                 # 内置工具和工具协议
│   ├── utils/                 # 认证、异常、日志、通用工具
│   ├── tests/                 # 单元测试和集成测试
│   ├── app.py                 # FastAPI 应用工厂
│   └── main.py                # 后端启动入口
├── frontend/
│   ├── src/
│   │   ├── api/               # API 调用封装
│   │   ├── components/        # 通用组件
│   │   ├── pages/             # 聊天和后台页面
│   │   ├── stores/            # Zustand 状态
│   │   └── styles/            # 全局样式
│   └── vite.config.ts
├── data/                      # 上传文件和本地向量数据
├── logs/                      # 运行日志
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 快速开始

### 环境要求

- Python 3.12+
- Node.js 18+
- Docker Desktop，或本地 MySQL 8 + Redis 7

### 1. 准备环境变量

```powershell
Copy-Item .env.example .env
```

按需修改 `.env`。本地 Docker Compose 默认暴露：

- MySQL: `127.0.0.1:3307`
- Redis: `127.0.0.1:6379`

如果使用 compose 里的 MySQL，建议本地开发连接：

```env
DATABASE_URL=mysql+pymysql://agentoffice:agentoffice123@127.0.0.1:3307/agentoffice?charset=utf8mb4
TEST_DATABASE_URL=mysql+pymysql://agentoffice:agentoffice123@127.0.0.1:3307/agentoffice_test?charset=utf8mb4
REDIS_URL=redis://localhost:6379/0
```

### 2. 启动 MySQL 和 Redis

```powershell
docker compose up -d mysql redis
```

### 3. 启动后端

项目作者当前使用的 Conda 环境：

```powershell
conda activate agentoffice
```

首次安装依赖：

```powershell
pip install -r requirements.txt
pip install "pymilvus[milvus_lite]>=2.4.0"
```

启动服务：

```powershell
cd backend
python main.py
```

后端默认地址：

- API: `http://127.0.0.1:8000/api`
- Swagger: `http://127.0.0.1:8000/docs`

### 4. 启动前端

```powershell
cd frontend
npm install
npm run dev
```

前端默认地址：`http://127.0.0.1:3000`

## Docker 运行

完整启动：

```powershell
docker compose up -d
```

停止服务：

```powershell
docker compose down
```

Compose 服务：

| 服务 | 容器 | 端口 |
| --- | --- | --- |
| 后端 | `agent-office` | `8000` |
| MySQL | `agent-office-mysql` | `3307 -> 3306` |
| Redis | `agent-office-redis` | `6379` |

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/auth/login` | 登录 |
| POST | `/api/auth/register` | 注册 |
| GET | `/api/auth/me` | 当前用户 |
| POST | `/api/chat/completions` | 发送消息并返回 Agent 回答 |
| GET | `/api/chat/history` | 会话历史 |
| GET | `/api/chat/sessions` | 会话列表 |
| POST | `/api/knowledge/upload` | 上传知识库文档 |
| POST | `/api/knowledge/search` | 搜索知识库 |
| GET | `/api/tool/list` | 工具列表和权限声明 |
| GET | `/api/admin/dashboard` | 后台总览 |
| GET | `/api/admin/traces` | 工具链路追踪 |
| GET | `/api/admin/config` | 系统配置 |

完整接口以 `/docs` 为准。

## 配置说明

常用环境变量：

| 变量 | 说明 |
| --- | --- |
| `DATABASE_URL` | 业务数据库连接 |
| `TEST_DATABASE_URL` | 测试数据库连接 |
| `REDIS_URL` | Redis 连接 |
| `MODEL_PROVIDER` | 模型供应商，支持 `qwen`、`openai`、`deepseek` |
| `QWEN_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | 模型密钥 |
| `SMTP_HOST` / `SMTP_FROM_EMAIL` | 邮件发送配置 |
| `UPLOAD_DIR` | 知识库上传目录 |
| `VECTOR_STORE_DIR` | 本地向量存储目录 |
| `KNOWLEDGE_SIMILARITY_THRESHOLD` | 知识库检索相似度阈值 |
| `AGENT_MEMORY_SIMILARITY_THRESHOLD` | Agent 记忆检索相似度阈值 |
| `MCP_HTTP_ENDPOINT` | MCP HTTP 工具端点 |
| `JWT_SECRET_KEY` | JWT 签名密钥，生产环境必须替换 |

未配置外部 LLM Key 时，系统会回退到本地规则模型，仍可处理部分天气、时间、计算、知识库等基础任务。

## 当前设计边界

- 工具权限目前是执行器内构造的能力集，已经具备协议边界，但还没有接入更细粒度的角色/策略表。
- 长期记忆已经区分 `semantic` 和 `episodic`，历史旧数据如果没有 `memory_kind` 元数据，不会被新的检索策略复用。
- Redis 不可用时会降级，但 MySQL 是当前运行和测试的必需依赖。
- MCP 工具需要配置 `MCP_HTTP_ENDPOINT` 后才会注册。

## License

本项目按 MIT License 发布。若仓库中未包含 `LICENSE` 文件，请在正式发布前补充。
