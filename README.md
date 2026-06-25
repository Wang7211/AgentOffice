# AgentOffice

AgentOffice 是一个面向办公和知识处理场景的智能 Agent 系统。项目提供聊天式任务处理、RAG 知识库、工具调用、会话记忆、用户认证、后台管理和 MCP 工具扩展能力，适合作为企业办公智能体原型继续扩展。

## 功能概览

- **聊天 Agent**：基于 LangGraph 编排理解、规划、执行、观察、最终回复和记忆写入流程。
- **结构化任务规划**：LLM 负责产出计划，执行器负责校验工具边界、依赖关系和执行状态。
- **工具调用**：内置天气、邮件、计算、文件解析、时间、知识库检索、网页读取等工具。
- **RAG 知识库**：支持上传 PDF、TXT、DOCX，完成解析、切片、向量化、检索和用户隔离。
- **会话与记忆**：保留短期对话窗口、最近工具结果和可持久化的长期用户事实。
- **用户系统**：JWT 认证，聊天记录、知识库和后台管理按用户隔离。
- **后台管理**：提供仪表盘、用户管理、知识库文件、工具链路追踪和系统配置接口。
- **MCP 扩展**：可通过 HTTP MCP 端点发现并注册远程工具。

## 技术栈

### 后端

| 模块 | 技术 |
| --- | --- |
| Web 框架 | FastAPI |
| Agent 编排 | LangGraph |
| ORM | SQLAlchemy 2.x |
| 数据库 | MySQL 8 |
| 缓存 | Redis 7 |
| 向量检索 | Milvus |
| 模型接入 | Qwen、OpenAI、DeepSeek 兼容接口 |
| 认证 | JWT、bcrypt |

### 前端

| 模块 | 技术 |
| --- | --- |
| 框架 | React 18 |
| 构建 | Vite |
| UI | Ant Design 5 |
| 状态管理 | Zustand |
| 路由 | React Router 6 |

## 项目结构

```text
AgentOffice/
├─ backend/
│  ├─ agent/          # LangGraph 状态、节点和执行图
│  ├─ api/            # REST / SSE 接口
│  ├─ config/         # 环境配置
│  ├─ database/       # SQLAlchemy 连接、表模型和初始化 SQL
│  ├─ integrations/   # MCP 客户端
│  ├─ memory/         # 会话记忆、Redis KV、向量记忆
│  ├─ schemas/        # Pydantic 和结构契约
│  ├─ services/       # Chat、LLM、Knowledge、Tool、Admin 服务
│  ├─ tools/          # 内置工具和工具协议
│  ├─ utils/          # 认证、日志、异常、通用函数
│  ├─ tests/          # 单元测试和集成测试
│  ├─ app.py          # FastAPI 应用工厂
│  └─ main.py         # 后端启动入口
├─ frontend/
│  ├─ src/
│  │  ├─ api/         # 前端 API 封装
│  │  ├─ components/  # 通用组件
│  │  ├─ pages/       # 聊天和后台页面
│  │  ├─ stores/      # Zustand 状态
│  │  └─ styles/      # 全局样式
│  └─ vite.config.ts
├─ data/              # 上传文件和本地向量数据
├─ logs/              # 运行日志
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
└─ .env.example
```

## 快速启动

### 环境要求

- Python 3.12+
- Node.js 18+
- Docker Desktop，或本地安装 MySQL 8 + Redis 7

### 1. 创建环境变量文件

```powershell
Copy-Item .env.example .env
```

如果使用 `docker compose` 启动 MySQL，宿主机访问端口是 `3307`，建议将 `.env` 中的数据库地址改为：

```env
DATABASE_URL=mysql+pymysql://agentoffice:agentoffice123@127.0.0.1:3307/agentoffice?charset=utf8mb4
TEST_DATABASE_URL=mysql+pymysql://agentoffice:agentoffice123@127.0.0.1:3307/agentoffice_test?charset=utf8mb4
REDIS_URL=redis://localhost:6379/0
```

至少还需要配置一个模型供应商的 API Key，例如：

```env
MODEL_PROVIDER=qwen
QWEN_API_KEY=你的 DashScope API Key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
```

### 2. 启动 MySQL 和 Redis

```powershell
docker compose up -d mysql redis
```

如果使用 Windows 本地 Redis 服务：

```powershell
Start-Service Redis
redis-cli ping
```

### 3. 启动后端

```powershell
pip install -r requirements.txt

cd backend
python main.py
```

默认地址：

- API: `http://127.0.0.1:8000/api`
- Swagger: `http://127.0.0.1:8000/docs`

### 4. 启动前端

```powershell
cd frontend
npm install
npm run dev
```

默认地址：

- 前端: `http://127.0.0.1:3000`
- Vite 会把 `/api` 和 `/static` 代理到 `http://127.0.0.1:8000`

## Docker 运行

完整启动后端、MySQL 和 Redis：

```powershell
docker compose up -d
```

停止：

```powershell
docker compose down
```

服务端口：

| 服务 | 容器名 | 端口 |
| --- | --- | --- |
| 后端 | `agent-office` | `8000:8000` |
| MySQL | `agent-office-mysql` | `3307:3306` |
| Redis | `agent-office-redis` | `6379:6379` |

注意：`agent-office` 容器内部会自动使用 `mysql:3306` 和 `redis:6379`，本机直接运行后端时才需要使用 `127.0.0.1:3307`。

## 主要接口

所有接口默认挂载在 `/api` 下。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/auth/login` | 登录 |
| POST | `/api/auth/register` | 注册 |
| GET | `/api/auth/me` | 当前用户 |
| PUT | `/api/auth/profile` | 更新个人资料 |
| POST | `/api/chat/completions` | 发送消息，支持普通 JSON 或 SSE 流式返回 |
| GET | `/api/chat/history` | 查询会话历史 |
| GET | `/api/chat/sessions` | 查询会话列表 |
| PUT | `/api/chat/sessions/{session_id}/rename` | 重命名会话 |
| DELETE | `/api/chat/sessions/{session_id}` | 删除会话 |
| GET | `/api/tool/list` | 查看已注册工具 |
| POST | `/api/knowledge/upload` | 上传知识库文件 |
| POST | `/api/knowledge/search` | 检索知识库 |
| GET | `/api/admin/dashboard` | 后台仪表盘 |
| GET | `/api/admin/users` | 用户列表 |
| GET | `/api/admin/knowledge/files` | 知识库文件列表 |
| GET | `/api/admin/traces` | 工具链路追踪 |
| GET | `/api/admin/config` | 系统配置 |

完整参数以 Swagger 文档 `/docs` 为准。

## Agent 执行流程

```text
用户消息
  -> mem_pre 加载短期摘要和长期记忆
  -> understand 生成语义理解和任务契约
  -> planning 生成结构化计划
  -> execute 执行一个可运行步骤
  -> observe 评估工具结果和任务状态
  -> execute / planning / finalize 按状态继续执行、重规划或生成最终回复
  -> mem_post 写入可持久化记忆
```

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `APP_HOST` / `APP_PORT` | 后端监听地址和端口 |
| `DATABASE_URL` | 业务数据库连接 |
| `TEST_DATABASE_URL` | 测试数据库连接 |
| `REDIS_URL` | Redis 连接 |
| `MODEL_PROVIDER` | 模型供应商：`qwen`、`openai`、`deepseek` 或 `local` |
| `MODEL_NAME` | 会话展示用模型名 |
| `QWEN_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | 模型 API Key |
| `QWEN_BASE_URL` / `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL` | OpenAI 兼容接口地址 |
| `UPLOAD_DIR` | 知识库上传目录 |
| `VECTOR_STORE_DIR` | 本地向量数据目录 |
| `MAX_UPLOAD_MB` | 最大上传文件大小 |
| `CHAT_WINDOW_SIZE` | 短期会话窗口大小 |
| `KNOWLEDGE_SIMILARITY_THRESHOLD` | 知识库检索相似度阈值 |
| `AGENT_MEMORY_SIMILARITY_THRESHOLD` | Agent 长期记忆检索阈值 |
| `MCP_HTTP_ENDPOINT` / `MCP_API_KEY` | MCP HTTP 工具端点和密钥 |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USERNAME` / `SMTP_PASSWORD` / `SMTP_FROM_EMAIL` | 邮件发送配置 |
| `JWT_SECRET_KEY` | JWT 签名密钥 |

## License

当前仓库未看到独立 `LICENSE` 文件。正式发布前请补充许可证文件；如无特殊要求，可使用 MIT License。