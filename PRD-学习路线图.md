# PRD：LangGraph 系统学习路线图

> **版本：** v1.0
> **目标用户：** AI 零基础小白
> **中期目标：** 5个月达到找工作水平
> **长期目标：** 成为专业 AI 专家

---

## 一、产品概述

### 1.1 背景

LangGraph 是当前 AI Agent 开发领域最主流的编排框架。掌握 LangGraph = 掌握构建复杂 AI 应用的核心能力。

### 1.2 学习目标

| 阶段 | 目标 | 可交付能力 |
|------|------|-----------|
| 入门（1-4周） | 理解核心概念，能独立构建简单 Agent | 能写一个带记忆和工具调用的 ReAct Agent |
| 进阶（5-8周） | 掌握多 Agent 协作与人工干预 | 能搭建带人工审批的多 Agent 团队 |
| 实战（9-12周） | 企业级架构能力 | 能设计子图嵌套、并行执行的内容工厂 |
| 产品化（13-19周） | 全栈产品能力 | 能构建带 Web 界面、监控、RAG 的完整产品 |
| 大神（持续） | 架构师/技术专家 | 能设计大规模 Agent 系统、参与社区贡献 |

### 1.3 前置要求

- ✅ **Python 基础语法**（变量、函数、类、列表/字典）
- ✅ **能运行 Python 脚本**（你已经做到了！test/ 下文件都能跑）
- ❌ **不需要** AI/机器学习基础
- ❌ **不需要** LangChain 经验
- ❌ **不需要** 深度学习知识

---

## 二、阶段详细规划

### 第一阶段：基础入门（第0-3周）

**目标：** 理解 LangGraph 最核心的 4 个概念：StateGraph、Node、Edge、State

#### 第0周：Python + AI 基础扫盲

| 学习项 | 内容 | 参考资源 |
|--------|------|---------|
| Python 类型注解 | `TypedDict`、`Annotated`、`List[str]` | Python 官方文档 |
| 什么是 LLM | 大语言模型的基本原理 | 吴恩达 Short Course |
| API 调用 | ChatOpenAI 的使用 | `test/my_first_agent.py` |
| 工具调用 | `@tool` 装饰器、bind_tools | LangChain 文档 |

**任务：** 修改 `my_first_agent.py`，加一个 `calculator` 工具，问 LLM 数学题

#### 第1周：第一个 Agent — ReAct 模式

| 概念 | 一句话理解 | 对应的代码 |
|------|-----------|-----------|
| StateGraph | 画工作流图的画板 | `StateGraph(State)` |
| Node | 图里的一个工作步骤 | `add_node("name", fn)` |
| Edge | 步骤之间的连线 | `add_edge()` / `add_conditional_edges()` |
| State | 所有节点共享的"白板" | `class State(TypedDict)` |
| Compile | 把图画变成可执行的程序 | `app = graph.compile()` |

**任务：** 不用看代码，自己从头写出 `my_first_agent.py`

#### 第2周：状态管理与记忆

| 概念 | 一句话理解 | 关键代码 |
|------|-----------|---------|
| Checkpointer | Agent 的"存档系统" | `compile(checkpointer=memory)` |
| MemorySaver | 内存存档（重启丢） | `from langgraph.checkpoint.memory import MemorySaver` |
| Thread ID | 区分不同对话的 ID | `config = {"configurable": {"thread_id": "xxx"}}` |
| add_messages | 自动把消息追加到列表 | `messages: Annotated[list, add_messages]` |

**关键实验：** 问两次话，第二次说"那上海呢？"，看 Agent 能否理解上下文

#### 第3周：条件路由与分支

| 概念 | 一句话理解 | 关键代码 |
|------|-----------|---------|
| Conditional Edge | 根据当前状态决定下一步 | `add_conditional_edges("node", router, mapping)` |
| 路由函数 | 读 State → 返回目标节点名 | `def router(state): return "a" if ... else "b"` |
| END | 图的终点 | `from langgraph.graph import END` |

**里程碑项目：** 构建一个带记忆的天气 + 计算器 Agent

```
用户 → 理解意图 → [查天气? → 调天气工具] [计算? → 调计算器] → 汇总回答
```

---

### 第二阶段：多 Agent 协作（第4-7周）

**目标：** 理解 Supervisor 架构，掌握人工干预机制

#### 第4周：Supervisor 调度模式

| 概念 | 一句话理解 |
|------|-----------|
| Supervisor | 不干活的"包工头"，只负责派活 |
| TeamState | 多 Agent 共享的白板 |
| 角色分离 | 每个 Agent 只干自己擅长的事 |
| 序列化流程 | Researcher → Writer → Editor |

**关键代码：** `test/multi_agent_team.py`

**实验：** 加一个 Designer 角色让团队更完整

#### 第5周：人工干预（踩坑最多的章节）

**学习顺序：**
1. `interrupt_before` — 编译时设断点
2. `get_state()` — 读当前状态
3. `update_state()` — 改状态（注意 as_node 参数）
4. `interrupt()` 函数 — 官方推荐方式
5. `Command(resume=)` — 恢复执行

**⚠️ 必踩的坑（一定要亲手体验）：**
- 增量 vs 快照：`invoke()` 返回值 ≠ 完整状态
- 状态覆盖陷阱：为什么 `update_state` 设 None 没用？
- 重放缓存：为什么打回重写输出一样？
- 语义死循环：为什么 Supervisor 停不下来？

**任务：** 阅读 `test/笔记/4.人工干预multi_agent_with_human.md`，逐行理解每个 Bug 的原因

#### 第6周：LLM 动态调度

让 Supervisor 不靠 if-else，而是靠 LLM 自己决定下一步。

```python
# 代码判断（本周之前）
def supervisor(state):
    if not state.get("notes"): return "researcher"
    elif not state.get("draft"): return "writer"
    else: return "FINISH"

# LLM 判断（本周学习）
def supervisor(state):
    response = model.invoke([
        SystemMessage(content="根据进度决定下一步"),
        HumanMessage(content=str(state))
    ])
    return {"next_agent": parse(response.content)}
```

**文件：** `test/multi_agent_humantool.py`

#### 第7周：持久化存储

| 类型 | 优势 | 劣势 |
|------|------|------|
| MemorySaver | 简单、无需配置 | 重启丢、不适合生产 |
| SqliteSaver | 文件持久化、重启不丢 | 不适合分布式 |
| PostgresSaver | 分布式、高性能 | 需要 PostgreSQL |

**文件：** `test/multi_agent_sqlitesaver.py`

**里程碑项目：** 带人工审批的内容创作团队

---

### 第三阶段：企业级实战（第8-11周）

**目标：** 掌握并行、子图、Stream 等企业级特性

#### 第8周：并行执行 — Send 机制

**核心理解：** Send = "把同一个任务用不同参数并行执行 N 次"

```python
# 串行
调研(技术角度) → 调研(教育角度) → 调研(伦理角度) → 汇总
# 耗时：3个调研的时间相加

# 并行
[调研(技术角度), 调研(教育角度), 调研(伦理角度)] → 汇总
# 耗时：最慢的那个调研的时间
```

**⚠️ 关键陷阱：** Send 只能在条件边中使用，不能在普通节点返回。

**文件：** `test/multi_agent_Send.py`

#### 第9周：子图嵌套

**核心理解：** 子图 = 把一组节点打包成"黑盒"，主图只关心输入输出。

```
主图:
  Supervisor → [写作子图] → [翻译子图] → END
                     ↓
  写作子图内部:
    Supervisor → Researcher → Writer → Editor
```

**文件：** `test/multi_agent_Subgraphs.py`

#### 第10周：Stream 实时输出

```python
# 阻塞等全部完成
result = app.invoke(inputs)

# 实时输出（每完成一个节点就打印）
for chunk in app.stream(inputs):
    print(chunk)  # 立即看到每个节点的输出
```

**文件：** `test/multi_agent_Stream.py`

#### 第11周：内容工厂 V1

把前面学的综合起来，构建第一个完整项目：写作 → 翻译 → 配图 → 发布。

**文件：** `test/multi_agent_Supervisor.py`

**里程碑项目：** 4 子图嵌套的内容工厂流水线

---

### 第四阶段：产品化与面试（第12-19周）

**目标：** 把内容工厂从"能跑"变成"能用"

#### 第12-13周：Human-in-the-loop 深度实践

- 多审核点（写作后、配图后、发布前）
- 智能重写（用户可输入具体修改意见）
- 编辑反馈闭环（把 feedback 注入 prompt）

**文件：** `test/content_factory_hitl.py`

#### 第14周：Time Travel 状态回溯

**核心能力：** 回到任意历史 checkpoint，修改状态后重新执行。

- `get_state_history()` — 获取所有历史快照
- `update_state()` — 在历史节点上修改状态
- 从历史节点重新执行

**文件：** `test/content_factory_timetravel.py`

#### 第15周：AI 动态决策

让 AI 自己做判断：
- 给文章质量打分（<7分自动重写）
- 判断是否需要配图
- 选择最优发布平台

**技术：** Pydantic 结构化输出 + `with_structured_output()`

**文件：** `test/content_factory_ai_decision.py`

#### 第16周：动态图构建

运行时根据用户选择动态构建图：

```python
def build_graph(features):
    graph = StateGraph(MState)
    graph.add_node("writing", writing_subgraph)
    if features.get("translate"):  # 用户选了翻译？
        graph.add_node("trans", trans_subgraph)
    if features.get("image"):      # 用户选了配图？
        graph.add_node("image", image_subgraph)
    return graph.compile(checkpointer=memory)
```

**文件：** `test/content_factory_dynamic.py`

#### 第17周：LangSmith 监控

- 注册 LangSmith 账号
- 配置环境变量（零代码侵入）
- 查看 Trace、Token 消耗、节点耗时
- 生产环境用法（采样率、标签、元数据）

**文件：** `test/content_factory_langsmith.py`

#### 第18周：Web Search & RAG

- DuckDuckGo 免费搜索（调研节点接入互联网）
- Tavily 精准搜索
- RAGFlow 知识库接入
- 混合检索策略（知识库 + 网络）

**文件：** `test/content_factory_websearch.py`

#### 第19周：面试准备

| 面试类型 | 准备内容 |
|---------|---------|
| 概念题 | StateGraph、Checkpointer、interrupt、Send 的原理 |
| 对比题 | invoke vs stream、MemorySaver vs SqliteSaver、Node vs Subgraph |
| 场景题 | 如何设计多 Agent 团队？如何做人工审批？如何做状态回溯？ |
| 手写题 | 现场写一个 Supervisior 模式的 Agent 团队 |
| 项目题 | 介绍内容工厂项目：架构、难点、解决方案 |
| 系统设计 | 设计一个客服 Agent 系统、设计一个内容生成平台 |

---

### 第五阶段：AI 大神进阶（持续）

#### 源码级理解
- 深入 Pregel 引擎：LangGraph 的底层执行模型
- Checkpointer 接口设计：如何自定义存储后端
- Send 调度源码：并行任务的调度与同步

#### 高级架构
- 多智能体复杂通信（Agent 间消息传递）
- 事件驱动架构（外部 webhook 触发图继续）
- 大规模 Agent 编排（100+ Agent 的调度策略）
- 异步图执行（async/await 模式）

#### AI 工程化
- RAG 系统深入（向量数据库、检索策略、重排序）
- Evaluation 体系（如何评估 Agent 质量）
- A/B 测试框架（Prompt 版本管理）
- 成本优化（Token 预算控制、缓存策略）
- Agent 安全（Prompt 注入防护、权限控制）

#### 前沿方向
- 多模态 Agent（文字 + 图片 + 语音）
- MCP 协议（Model Context Protocol）
- Agent-to-Agent 协议（A2A）
- Deep Agents（规划 + 子任务 + 文件系统）

---

## 三、学习资源清单

### 官方资源
| 资源 | 地址 | 用途 |
|------|------|------|
| LangGraph 文档 | https://docs.langchain.com/oss/python/langgraph/overview | 查概念、API |
| LangGraph 快速开始 | https://docs.langchain.com/oss/python/langgraph/quickstart | 第一个 Agent |
| LangChain Academy | https://academy.langchain.com/courses/intro-to-langgraph | 免费视频课程 |
| LangGraph 源码 | `D:\demo\langgraph-main\libs\langgraph` | 深入学习 |
| LangSmith | https://smith.langchain.com | 全链路监控 |

### 学习社区
- **LangChain 论坛**：https://forum.langchain.com
- **LangChain Discord**：官方 Discord（回答问题很快）
- **GitHub Discussions**：https://github.com/langchain-ai/langgraph/discussions

### 推荐学习路径
1. 读笔记 → 2. 跑代码 → 3. 改代码实验 → 4. 写总结 → 5. 教别人

---

## 四、常见问题 FAQ

**Q：我需要先学 LangChain 吗？**
A：不需要。LangGraph 可以独立使用，LangChain 是可选的。

**Q：每天需要学多久？**
A：建议每天 1-2 小时，周末 3-4 小时。5 个月是合理预期。

**Q：找工作需要学到什么程度？**
A：完成第四阶段（内容工厂完整版 + 项目经验）。核心是理解所有核心概念 + 一个完整的项目演示。

**Q：内容工厂项目够面试用吗？**
A：完全够。它涵盖了：多 Agent 协作、人工干预、并行执行、子图嵌套、Stream、动态构图、监控、RAG。面试官会非常感兴趣。

**Q：需要什么硬件配置？**
A：能运行 Python 的电脑即可。大模型调用走 API，不需要 GPU。

---

## 五、考核标准

| 阶段 | 通过标准 |
|------|---------|
| 基础入门 | 能独立写出带记忆的 ReAct Agent |
| 多Agent协作 | 能搭建带人工审批的多 Agent 团队，并解释每个踩坑的原因 |
| 企业级实战 | 能设计并实现子图嵌套 + 并行执行的流水线 |
| 产品化 | 能完整演示内容工厂项目，回答面试问题 |
| AI大神 | 能参与开源社区，能设计大规模 AI 系统 |

---

> ⚠️ **完整功能补充：** 本文档只覆盖 test/ 目录已有内容。LangGraph 完整功能见 `功能全景与补全学习路线.md`（含异步、Store、RetryPolicy、stream_mode 全系列等缺失功能）

> 📖 **学习顺序：** `总结.md` → `第一阶段-基础入门.md` → 各阶段按顺序学习
