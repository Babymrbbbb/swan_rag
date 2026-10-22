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
│   └── eval.py          # golden-set 评测（召回命中率 + 忠实度）
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

## 许可

教学 / 作品集用途。知识库内容为合成数据，与任何真实企业经营数据无关。
