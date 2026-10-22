# 天鹅公主 AI 知识中台（RAG 服务）

> **定位**：面向「天鹅公主 SWAN NOBLE」**43 家门店烘焙连锁**的 AI 知识中台**落地方案与架构实现**。
> 由品牌运营负责人主导设计，目标是把分散在 SOP、新品资料、竞品情报、加盟政策中的知识，
> 收敛成一个「问即得、带出处、可审计」的检索增强生成（RAG）服务，直接服务一线门店与运营决策。

这不是一个脱离业务的"玩具 demo"——知识库的维度（品牌 / 门店网络 / 夏季新品 / 包材 / 会员 / 供应链 / 竞品金粒门 / 加盟 / 数字化）全部映射品牌真实经营切面；架构已用真实 LLM 端到端验证。**对外呈现的是"为真实业务设计的 AI 落地方案"，而非"练手项目"。**

---

## 一、业务背景与痛点

| 业务切面 | 现状痛点 | AI 中台要解决的 |
|---|---|---|
| 一线门店 | 新员工背 SOP / 产品知识靠口口相传，查标准慢、易走样 | 自然语言问答，答案带资料出处 |
| 新品上市 | 2026 夏季 4 款新品（黑金可颂/联名桃酥/荔枝玫瑰冻芝士/青提冷萃欧包）知识同步滞后 | 上市即入知识库，门店随问随答 |
| 竞品情报 | 金粒门等竞品动态分散在若干分析文档，无人能快速检索 | 竞品维度结构化、可追问 |
| 加盟 / 供应链 | 政策条款多、跨文档，加盟商咨询重复劳动高 | 政策问答 + 引用原文，降低人工成本 |
| 运营决策 | 市场/竞品分析散落，难沉淀复用 | 知识资产化，支撑复盘与决策 |

**量化目标（方案阶段，待落地验收）**：一线查标准 SOP 从「翻文档数分钟」降到「对话秒级」；竞品/政策类咨询的人工承接量目标下降 **30%+**；知识更新从「月度同步」提升到「随业务发生即入库」。

---

## 二、技术栈

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

## 三、架构图

### C4 · 容器层（Container）

```
┌──────────────────────────────────────────────────────────────┐
│                    门店一线 / 运营 / 决策者                     │
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

## 四、目录结构

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
│   └── static/index.html # 内部体验入口（黑金主题对话 UI）
├── data/
│   ├── corpus/          # 知识库（维度映射品牌真实经营切面，13 篇）
│   └── eval_golden.json # 评测黄金集
├── docs/
│   └── business-case.md # 业务方案、简历 bullet、叙事框架
├── tests/test_main.py   # pytest（全程 Mock，离线可跑）
├── seed_corpus.py       # 灌库脚本
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml / .dockerignore / .env.example
└── README.md
```

---

## 五、业务价值（痛点 → 架构 → 收益）

| 痛点 | 架构杠杆 | 预期收益（方案目标） |
|---|---|---|
| 一线查 SOP 慢、易走样 | 向量召回 + LLM 生成 + 原文引用 | 查询从分钟级→秒级，答案可溯源 |
| 竞品情报难检索 | 竞品维度结构化入库、可追问 | 竞品咨询人工承接量目标 ↓30% |
| 知识更新滞后 | ingest 接口随业务即入 | 知识同步从月度→实时 |
| 加盟政策咨询重复 | 政策问答 + 出处引用 | 降低人工客服成本 |
| 分析资产难沉淀 | QA 落库 + 评测闭环 | 知识资产化，支撑复盘决策 |

---

## 六、接口一览

| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| GET | `/health` | 健康检查（返回当前 llm/embedder 模式） | ✅ |
| POST | `/ingest` | 灌入一篇文档（自动切分 + 向量化） | ✅ |
| POST | `/chat` | RAG 问答，返回 answer + citations + latency | ✅ |
| GET | `/metrics` | 请求数 / 对话数 / 平均延迟 / token | ✅ |
| POST | `/eval` | 基于 golden set 跑评测，返回报告 | ✅ |

> 仅 `GET /`、`/docs`、`/openapi.json`、`/redoc` 免鉴权（便于内部体验）；业务接口仍强制 `X-API-Key`。

---

## 七、评测闭环（AI 质量不是玄学）

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

## 八、真实 LLM 链路验证记录（2026-10-22）

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

## 九、落地状态（诚实标注）

| 模块 | 状态 |
|---|---|
| 品牌知识维度建模（13 篇语料映射真实经营切面） | ✅ 已建模（语料为脱敏/合成映射，生产应替换为内部真实文档） |
| RAG 架构实现（FastAPI 全家桶 + 抽象解耦） | ✅ 已实现 |
| 真实 LLM 端到端验证 | ✅ 已验证（sensenova-6.8-flash-lite） |
| 评测闭环（golden-set） | ✅ 已建立 |
| 容器化部署（Docker / compose） | ✅ 已就绪，可一键起 |
| 接入品牌内部真实文档库 | 🟡 规划中（需接入内部 SOP / 新品 / 竞品资料） |
| 生产级向量库（pgvector）/ 多租户 | 🟡 规划中 |

> 本仓库是**方案与架构实现（已用真实 LLM 验证）**；将其推进到品牌内部生产部署，需要接入真实业务文档并过企业安全/数据合规评审。这是甲方落地的标准路径，不是缺陷。

---

## 十、Docker（内部体验 / 部署就绪）

```bash
cp .env.example .env        # 按需填真实 Key
docker compose up --build
# 默认 8000 端口；容器启动自动 seed_corpus
```

---

## 十一、架构决策记录（ADR）

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

## 十二、与 `store_tool` 的关系

`store_tool` 是 Day01–Day08 的「后端最小项目」（纯 CRUD + 鉴权 + 迁移 + Docker）。
本仓库在**同一套后端能力之上加了一层 AI 应用架构**，构成「从 CRUD 到 AI 架构」的能力证据链，
适合作为面试官能看到成长路径的作品集。

---

## 许可

教学 / 作品集用途。知识库内容为合成数据，与任何真实企业经营数据无关。
