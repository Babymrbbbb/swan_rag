# 天鹅公主 RAG 知识问答服务

> **定位**：面向「AI 应用 / 解决方案架构师」岗位的**架构参考实现（showcase）**。
> 它不是又一个 CRUD 教程，而是用来证明：同一套后端能力（FastAPI 全家桶），
> 可以**向上长出一层 AI 应用架构**——抽象解耦、评测闭环、可观测、可演进。

品牌背景取自「天鹅公主 SWAN NOBLE」（清远中高端烘焙连锁，43 家门店）的合成知识库，
用于演示 RAG（检索增强生成）管线的完整链路。

---

## 一、技术栈

| 层 | 选型 |
|---|---|
| Web 框架 | FastAPI（路由 / Depends 注入 / 全局鉴权 / 统一错误信封 / lifespan） |
| 数据模型 | SQLModel（SQLAlchemy + Pydantic v2） |
| 配置 | pydantic-settings（`RAG_` 前缀，env / `.env` 覆盖） |
| 存储 | SQLite + SQLModel（生产可换 pgvector） |
| LLM / 向量化 | **供应商解耦**：`LLMProvider` / `Embedder` 抽象，内置 `Mock`（离线）+ `OpenAICompatible`（真实） |
| 检索 | 文本切分 + 向量余弦召回 top-k |
| 测试 / 评测 | pytest + golden-set 评测脚本 |
| 部署 | Docker 分层构建 + docker-compose |

---

## 二、架构图

### C4 · 容器层（Container）

```
┌──────────────────────────────────────────────────────────────┐
│                         Client / 面试官                         │
└───────────────┬──────────────────────────────────────────────┘
                │  POST /ingest · POST /chat · GET /metrics · POST /eval
                │  Header: X-API-Key
                ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI 服务 (app/main.py)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │ 路由 + 鉴权 │  │ 统一错误信封 │  │ lifespan    │  │ 指标收集  │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
│        │                  │                    │                 │
│        ▼                  ▼                    ▼                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RAG 管线 (app/rag.py)                                    │   │
│  │   ingest: 切分 → 向量化 → 存 Chunk                          │   │
│  │   chat:   向量化查询 → 召回 → 拼 prompt → LLM 生成          │   │
│  └───────────┬───────────────────────┬─────────────────────┘   │
│              ▼                       ▼                          │
│   ┌──────────────────┐      ┌────────────────────────────┐     │
│   │ LLMProvider 抽象  │      │ Embedder 抽象              │     │
│   │  Mock / OpenAI兼容│      │  Mock(哈希) / OpenAI兼容   │     │
│   └──────────────────┘      └────────────────────────────┘     │
│              │                       │                          │
│              ▼                       ▼                          │
│   ┌──────────────────────────────────────────────────────┐    │
│   │  SQLite: Document / Chunk(embedding JSON) / QA        │    │
│   └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 时序图（一次 /chat 请求）

```
Client → API网关(鉴权X-API-Key) → /chat
  /chat → Embedder.embed(question)         // 向量化查询
  /chat → Retriever.search(top_k)          // 余弦召回 Chunk
  /chat → 组装 prompt(系统提示 + 召回原文)
  /chat → LLMProvider.complete(prompt)     // Mock 或 真实 LLM
  /chat → 记录指标 / 落库 QA
  /chat → {answer, citations, latency_ms} → Client
```

---

## 三、目录结构

```
swan_rag/
├── app/
│   ├── main.py          # FastAPI 入口：路由 + 全局鉴权 + 统一错误信封 + lifespan
│   ├── config.py        # pydantic-settings（RAG_ 前缀）
│   ├── database.py      # 引擎 / init_db / get_session（Depends 注入）
│   ├── models.py        # 表模型：Document / Chunk / QA
│   ├── schemas.py       # 请求/响应模型（入参出参分离）
│   ├── metrics.py       # 进程内指标收集（可观测最小实现）
│   ├── llm.py           # LLMProvider 抽象 + Mock + OpenAICompatible
│   ├── embedder.py      # Embedder 抽象 + Mock(哈希) + Real
│   ├── retriever.py     # 切分 + 余弦召回
│   ├── rag.py           # 管线编排：ingest / chat
│   ├── eval.py          # golden-set 评测（召回命中率 + 忠实度）
│   └── static/index.html # 演示页（黑金主题对话 UI，GET / 返回）
├── data/
│   ├── corpus/          # 天鹅公主合成知识库（13 篇 markdown）
│   └── eval_golden.json # 评测黄金集
├── tests/test_main.py   # pytest（全程 Mock，离线可跑）
├── seed_corpus.py       # 灌库脚本
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml / .dockerignore / .env.example
└── README.md
```

---

## 四、快速开始（本地 · 默认 Mock 离线模式）

```bash
# 1. 建虚拟环境并装依赖
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

# 2. 灌库（把 data/corpus 写入向量库）
python seed_corpus.py

# 3. 起服务
uvicorn app.main:app --reload --port 8000

# 4. 调用（注意全局鉴权头）
curl -H "X-API-Key: dev-secret-key" -H "Content-Type: application/json" \
  -d '{"question":"天鹅公主2026夏季推出了哪四款新品？"}' http://127.0.0.1:8000/chat
```

默认 `RAG_LLM_PROVIDER=mock`：不依赖外网即可演示完整 RAG 链路（回答由 `MockProvider`
回显召回原文，明确标注「未调用真实 LLM」）。

---

## 五、接口一览

| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| GET | `/health` | 健康检查（返回当前 llm/embedder 模式） | ✅ |
| POST | `/ingest` | 灌入一篇文档（自动切分 + 向量化） | ✅ |
| POST | `/chat` | RAG 问答，返回 answer + citations + latency | ✅ |
| GET | `/metrics` | 请求数 / 对话数 / 平均延迟 / token | ✅ |
| POST | `/eval` | 基于 golden set 跑评测，返回报告 | ✅ |

> 所有接口强制 `X-API-Key`；缺失/错误返回 `401` 统一错误信封 `{"error":{"code","message"}}`。

---

## 六、评测闭环（AI 质量不是玄学）

```bash
python -m app.eval
```

输出示例（Mock 模式）：
```json
{
  "total": 8,
  "retrieval_hit_rate": 1.0,
  "faithfulness_rate": 1.0,
  "note": "离线 MockProvider：召回命中率验证检索与切分正确；faithfulness 仅在接入真实 LLM 后具备真实意义。"
}
```

- **retrieval_hit_rate**：期望文档是否进入召回 top-3 —— 验证切分 / 向量化 / 检索正确性。
- **faithfulness_rate**：期望事实是否出现在「回答」或「召回原文」中 —— 检索充分性度量。
  > 接入真实 LLM 后，应使用 **LLM-Judge / NLI** 度量生成答案本身的忠实度（ADR-002）。

`pytest` 同样全程 Mock 跑通（7 个用例：鉴权 401 / 422 / ingest+chat / metrics / eval / health）。

---

## 七、接入真实 LLM（供应商解耦的价值）

本服务**不硬编码任何 SDK**。改 `.env` 即可切换，零改代码：

```ini
RAG_LLM_PROVIDER=real
RAG_LLM_BASE_URL=https://api.openai.com/v1   # 或 DeepSeek / 商汤网关等任意 OpenAI 兼容地址
RAG_LLM_API_KEY=sk-xxxx
RAG_LLM_MODEL=gpt-4o-mini

RAG_EMBEDDER_PROVIDER=real
RAG_EMBEDDER_API_KEY=sk-xxxx
RAG_EMBEDDER_MODEL=text-embedding-3-small
```

支持 OpenAI / DeepSeek / 商汤网关等一切 **OpenAI 兼容** 端点。向量库可平滑替换为 pgvector。

---

## 八、Docker

```bash
cp .env.example .env        # 按需填真实 Key
docker compose up --build
# 默认 8000 端口；容器启动自动 seed_corpus
```

---

## 九、架构决策记录（ADR）

**ADR-001 · 供应商解耦（LLM / Embedder 抽象）**
> 把 LLM 与向量化抽象成接口，内置 `Mock` + `OpenAICompatible` 两套实现。
> 动机：① 离线 / CICD 可跑通，不被外部 API 与网络绑架；② 真实供应商可插拔，
> 避免被单一厂商锁定；③ 演示「架构师该做的抽象」，而非「调通一个接口」。

**ADR-002 · 评测闭环优先**
> 必须有 golden-set 评测，把「AI 质量」变成可量化指标。离线用 Mock 验证检索正确性；
> 真实环境用 LLM-Judge 度量生成忠实度。动机：AI 应用最大的坑是「看着能答，实则幻觉」，
> 没有评测闭环的系统不可上线。

**ADR-003 · 复用已学后端能力，而非另起炉灶**
> 直接复用 FastAPI + SQLModel + Depends + 统一错误信封 + 配置分离 + 全局鉴权 + Docker
> （见同系列 `store_tool`）。动机：作品集要展示「能力递进」——同一人会从 CRUD 进阶到
> AI 架构，而不是两段不相关的代码。

---

## 十、与 `store_tool` 的关系

`store_tool` 是 Day01–Day08 的「后端最小项目」（纯 CRUD + 鉴权 + 迁移 + Docker）。
本仓库在**同一套后端能力之上加了一层 AI 应用架构**，构成「从 CRUD 到 AI 架构」的能力证据链，
适合作为面试官能看到成长路径的作品集。

---

## 十一、真实 LLM 链路验证记录（2026-10-22）

> 证明：本服务的 `OpenAICompatibleProvider` 不止能跑 Mock，也能接真实 LLM 端到端产出可用答案。

**验证环境**：沙箱直连商汤真实上游 `https://token.sensenova.cn/v1`（走代理可达；注：`git` 隧道被沙箱策略拦 502，但 LLM REST 端点正常）。

**选用模型**：`sensenova-6.8-flash-lite`（商汤网关 `keys.json` 中的可用模型之一）。

**踩到的坑（已修复）**：该模型是**推理模型**，默认把 token 预算全花在 `reasoning` 思考链上，`message` 里没有 `content` 字段，原 `data["choices"][0]["message"]["content"]` 会取空。
修复方式（已合入 `app/llm.py` + `app/config.py`）：请求体带 `"reasoning":{"enabled":false}`，并做 `content → reasoning_content` 兜底提取；开关 `RAG_LLM_DISABLE_REASONING`（默认 `true`）。

**`.env` 配置（真实链路）**：
```ini
RAG_LLM_PROVIDER=real
RAG_LLM_BASE_URL=https://token.sensenova.cn/v1
RAG_LLM_API_KEY=sk-xxxx          # 取自本地商汤网关 keys.json（acc1/acc2/acc3 三把可轮询）
RAG_LLM_MODEL=sensenova-6.8-flash-lite
RAG_LLM_DISABLE_REASONING=true   # 关闭思考链，直出 content
# 向量化保持 mock（与已存库 dim=256 一致）
```

**验证结果（mock 向量灌库 13 篇 → 真实 /chat，真实网络往返 3.2–4.7s）**：

| 问题 | 真实 LLM 回答 | 召回来源 |
|---|---|---|
| 天鹅公主是哪个城市的品牌？ | **清远**（无 RAG 时模型幻觉为「上海」，RAG 检索上下文纠正为清远并引《品牌介绍》原文） | 《品牌介绍：天鹅公主 SWAN NOBLE》 |
| 2026 夏季推出了哪几款新品？ | 黑金可颂 / 联名桃酥 / 荔枝玫瑰冻芝士 / 青提冷萃欧包 | 《2026 夏季四款新品》 |
| 目前有多少家门店？ | 43 家（清远 28 / 广州 9 / 佛山 4 / 韶关 2） | 《门店网络》 |

> 关键证据：第 1 问证明 **RAG 检索上下文成功压制了模型参数幻觉**——正是 ADR-002「没有评测闭环的系统不可上线」的落地演示。

**说明**：真实 Key 仅存于本地 `.env`（已被 `.gitignore` 排除，**未进入本仓库**）；远程仓库不含任何 Key。代码改进见 commit `52fe027`。

---

## 十二、Demo 指南（让面试官能真问真答）

本服务自带一个**黑金主题对话界面**，打开页面即可提问，回答附资料来源——把作品集从「看代码」升级成「能体验」。

### 本地直接起（最快）
```bash
# 灌库（mock 向量，离线可跑）
python seed_corpus.py
# 起服务
uvicorn app.main:app --reload --port 8000
```
- 演示页：http://127.0.0.1:8000  ← 直接提问
- API 文档：http://127.0.0.1:8000/docs （Swagger，可在线试各接口）
- 接真实 LLM：把 `.env` 配成「十一」节的真实配置后再起服务即可，UI 无需改一行。

### Docker 一键起
```bash
cp .env.example .env          # 按需填真实 Key（见「十一」节）
docker compose up --build
# 浏览器打开 http://localhost:8000
```
> `docker-compose.yml` 的 `environment` 用 `${VAR:-默认}` 读取同目录 `.env`（`RAG_LLM_PROVIDER` 默认 `mock`）。填了真实 Key 即自动走真实 LLM。

### 公网部署（可选）
把镜像推到任意容器平台（Railway / Render / 阿里云等），设置同名环境变量即可。
- 仅 `GET /`、`/docs`、`/openapi.json`、`/redoc` **免鉴权**（方便访客直接体验）；
- 业务接口 `POST /chat` 等**仍强制 `X-API-Key`**——演示页内置 `dev-secret-key` 自动携带，访客无需关心。
- 若要对外公开部署，建议：① 改 `RAG_API_KEY` 为强密钥并更新 UI 里的 `API_KEY`；或 ② 新增一个无鉴权的 `/demo/chat` 端点专门服务演示。

---

## 许可

教学 / 作品集用途。知识库内容为合成数据，与任何真实企业经营数据无关。
