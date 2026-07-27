# 第二阶段：多 Agent 协作 — 让 AI 团队像人类团队一样工作

> 🕐 **时间：** 第4-7周
> 🎯 **目标：** 掌握 Supervisor 调度模式、人工干预机制、LLM 动态调度、持久化存储
> 📁 **对应代码：** `test/multi_agent_team.py` → `multi_agent_sqlitesaver.py`

---

## 第4周：Supervisor 调度模式

### 🤔 为什么需要多 Agent？

单个 Agent 的局限：一个模型既要"查资料"、"写文章"又要"做审核"，容易**角色混淆**。

```python
# 单 Agent 的问题：一个模型身兼数职
def single_agent(state):
    # 它既要查资料、又要写文章、还要审核
    # prompt 越来越长，角色越来越混乱
    pass

# 多 Agent 的方案：各司其职
researcher = Agent("只负责调研")
writer = Agent("只负责写作")
editor = Agent("只负责审核")
supervisor = Agent("负责调度")
```

**实际公司里怎么干活，AI 团队就怎么干活。**

### ⭐ Supervisor 模式（星型拓扑）

```
                ┌─────────────┐
                │  用户提问    │
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │  Supervisor  │ ← 不干活，只派活
                │   (主管)     │
                └──────┬──────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Researcher│ │  Writer  │ │  Editor  │
    │ (调研员)  │ │ (撰稿人)  │ │ (编辑)    │
    └──────────┘ └──────────┘ └──────────┘
          │            │            │
          └────────────┼────────────┘
                       ▼
                ┌─────────────┐
                │  Supervisor │ ← 收集结果
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │  输出给用户   │
                └─────────────┘
```

### 🧩 multi_agent_team.py 核心拆解

```python
# ===== 1. 定义团队共享状态（TeamState）=====
class TeamState(TypedDict):
    """团队的"共享白板""""
    messages: Annotated[list, add_messages]   # 对话历史
    topic: str                                 # 要写的主题
    research_notes: str                        # 调研笔记（Researcher 写）
    draft: str                                 # 文章初稿（Writer 写）
    edit_feedback: str                         # 编辑意见（Editor 写）
    next_agent: str                            # 下一步谁干（Supervisor 写）

# ===== 2. 定义各角色节点 =====
def researcher_node(state: TeamState):
    """调研员：只负责搜集资料"""
    print("🕵️ [Researcher] 正在搜集资料...")
    prompt = f"针对主题 '{state['topic']}' 搜集关键信息，输出调研笔记。"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"research_notes": response.content}  # 只写 research_notes 字段

def writer_node(state: TeamState):
    """撰稿人：只负责写文章"""
    print("✍️ [Writer] 正在撰写初稿...")
    prompt = f"根据调研笔记撰写短文：\n{state['research_notes']}"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content}  # 只写 draft 字段

def editor_node(state: TeamState):
    """编辑：只负责审核"""
    print("📝 [Editor] 正在审核文章...")
    prompt = f"审核以下文章，提出修改意见：\n{state['draft']}"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"edit_feedback": response.content}

# ===== 3. 定义 Supervisor（主管）— 核心！=====
def supervisor_node(state: TeamState):
    """主管：检查共享状态，决定下一步"""
    print("👔 [Supervisor] 正在思考下一步...")
    
    # 核心逻辑：检查白板上缺什么
    if not state.get("research_notes"):
        next_agent = "researcher"    # 缺调研笔记 → 派调研员
    elif not state.get("draft"):
        next_agent = "writer"        # 缺初稿 → 派撰稿人
    elif not state.get("edit_feedback"):
        next_agent = "editor"        # 缺审核意见 → 派编辑
    else:
        next_agent = "FINISH"        # 全齐了 → 结束
    
    return {"next_agent": next_agent}  # 告诉框架下一步去谁

# ===== 4. 构建图 =====
builder = StateGraph(TeamState)

# 注册节点
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)

# 设置入口
builder.set_entry_point("supervisor")

# 条件边：Supervisor 决定下一步
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],  # 读取 Supervisor 的决策
    {
        "researcher": "researcher",
        "writer": "writer",
        "editor": "editor",
        "FINISH": END,
    }
)

# 普通边：各角色干完活 → 回到主管
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("editor", "supervisor")
```

### 🔄 执行流程可视化

```
Step 1: Supervisor 检查 → research_notes 为空 → "researcher"
Step 2: Researcher 执行 → 写 research_notes → 回到 Supervisor
Step 3: Supervisor 检查 → research_notes 有了，draft 为空 → "writer"
Step 4: Writer 执行 → 写 draft → 回到 Supervisor
Step 5: Supervisor 检查 → research_notes 有，draft 有，edit_feedback 为空 → "editor"
Step 6: Editor 执行 → 写 edit_feedback → 回到 Supervisor
Step 7: Supervisor 检查 → 全齐了 → "FINISH"
```

### ⚠️ 必踩的坑：Supervisor 无限循环

**现象：** 主管一直在派活，团队永远停不下来。

**原因：** Supervisor 没有真的检查状态，而是让 LLM 自己"想"下一步。

```python
# ❌ 错误写法：让 LLM 决定
def supervisor_node(state):
    response = model.invoke([SystemMessage(content="决定下一步")])
    next_agent = response.content  # LLM 可能胡说八道
    return {"next_agent": next_agent}

# ✅ 正确写法：代码逻辑判断
def supervisor_node(state):
    if not state.get("research_notes"):
        return {"next_agent": "researcher"}
    # ... 清晰的条件判断
```

**修复：** 用 if-else 而不是 LLM 来判断。

### 💡 第4周任务

1. 运行 `test/multi_agent_team.py`，观察执行流程
2. **修改代码**：给团队加一个 Designer（设计师）角色
3. 理解：为什么 Supervisor 不适合用 LLM 决定下一步？

---

## 第5周：人工干预 — 让人类在关键节点把关

### 🤔 为什么需要人工干预？

```python
# 全自动 AI 的风险
Agent 自动发邮件 → 万一写错了？
Agent 自动退款 → 万一算错了？
Agent 自动删数据 → 万一删错了？

# 解决方案：关键节点前"踩刹车"
Agent 写文章 → ⏸️ 人类审核 → [通过]→发布 [不通过]→重写
```

### 🛑 方案一：interrupt_before（编译时设断点）

```python
# 编译时告诉框架：执行 tools 之前停下来
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # ← 执行 tools 节点前暂停
)

# 运行时
inputs = {"messages": [("user", "帮我算 100*8")]}

# 第1步：执行到 tools 前暂停
for output in app.stream(inputs, config):
    pass  # 停在 tools 节点前

# 第2步：人类审查
pending = output['messages'][-1].tool_calls
print(f"AI 想调用: {pending}")
user_input = input("批准执行？(y/n): ")

# 第3步：如果批准，继续执行
if user_input == 'y':
    for output in app.stream(None, config):  # ← None 表示"从断点继续"
        print(output)
```

### 🛑 方案二：interrupt() 函数（官方推荐）

```python
from langgraph.types import interrupt, Command

def human_review_node(state):
    """人工审核节点"""
    print("⏸️ 等待人工审核...")
    
    # ★ interrupt() 会暂停整个图
    # - 暂停前，自动保存状态到 Checkpointer
    # - 把 payload 暴露给外部代码读取
    # - 等外部调用 Command(resume=...) 才继续
    
    decision = interrupt({
        "article": state.get("draft", ""),
        "message": "请审核这篇草稿"
    })
    
    # interrupt() 的返回值 = 外部传入的 resume 值
    # 比如：外部传 Command(resume="continue") → decision = "continue"
    #      外部传 Command(resume="rewrite")  → decision = "rewrite"
    
    return {"human_decision": decision}

# === 外部代码如何配合 ===

# 1. 正常跑，遇到 interrupt 会自动暂停
for chunk in app.stream(inputs, config):
    if "__interrupt__" in chunk:
        # 2. 读取中断信息
        payload = chunk["__interrupt__"][0].value
        print(payload["article"][:200])  # 显示文章预览
        
        # 3. 人类做决策
        user_input = input("continue/rewrite/stop: ")
        
        # 4. 从断点继续，传入决策
        for chunk in app.stream(Command(resume=user_input), config):
            # 图从 interrupt() 处恢复
            # interrupt() 返回 user_input
            pass
```

### 🔄 interrupt 完整生命周期

```
               外部代码                       LangGraph
                  │                            │
                  │  app.stream(inputs, config) │
                  │ ──────────────────────────→ │
                  │                            │
                  │                            ├── 执行到 human_review_node
                  │                            ├── 调用 interrupt(payload)
                  │                            ├── 保存 Checkpointer
                  │                            └── 暂停！
                  │                            │
                  │  收到 __interrupt__         │
                  │  显示文章给人类              │
                  │  等待输入...                │
                  │                            │
                  │  Command(resume="continue") │
                  │ ──────────────────────────→ │
                  │                            ├── 从断点恢复
                  │                            ├── interrupt() 返回 "continue"
                  │                            ├── 继续执行后续节点
                  │                            └── 正常完成
                  │                            │
                  │  收到最终输出               │
```

### ⚠️ 本阶段最大陷阱：增量 vs 快照

```python
# ❌ 错误写法
event = app.invoke(None, config)
draft = event.get('draft')  # ← 可能是 None！

# ✅ 正确写法
state = app.get_state(config).values  # ← 获取完整快照
draft = state.get('draft', '无')

# 为什么？
# app.invoke() 返回值 = 本次执行的"增量"（刚刚改了哪些字段）
# app.get_state() = 完整的当前状态快照（所有字段的最新值）
```

### ⚠️ 第二大陷阱：update_state 覆盖失败

```python
# ❌ 想清空 draft
app.update_state(config, {"draft": None})
# 结果：draft 还是旧值！因为 update_state 默认是"追加"不是"覆盖"

# ✅ 正确清空
app.update_state(config, {"draft": None, "edit_feedback": None}, as_node="editor")
# 加 as_node 参数，假装是 editor 节点重新执行了一次

# 或者更稳妥：追加一条消息来触发状态变化
app.update_state(config, {
    "messages": [HumanMessage(content="人类主管：文章不合格，请重新撰写。")]
})
```

### ⚠️ 第三大陷阱：重放缓存

```python
# 当你打回重写时...
# 你以为：Agent 会重新调用 LLM 生成新内容
# 实际：Agent 发现输入没变 → 直接返回缓存的上次结果

# 原因：Checkpointer 的"重放"机制
# 如果输入 State 和上次完全一样，就复用上次的输出

# 解决方案：
# 1. 把编辑反馈注入 Prompt（让输入不一样）
if state.get("edit_feedback"):
    prompt += f"\n【上一轮反馈】{state['edit_feedback']}"

# 2. 设置 temperature > 0
model = ChatOpenAI(temperature=0.7)  # 不是 0！
```

### ⚠️ 第四大陷阱：语义死循环

```python
# 当你把 update_state + while 循环 + Supervisor 结合起来
# 可能会陷入 Supervisor 永远停不下来的死循环

# 解决方案：加计数器！
class TeamState(TypedDict):
    # ...
    iteration_count: int  # 新增：记录调度次数

def supervisor_node(state):
    count = state.get("iteration_count", 0)
    
    # 硬性退出条件
    if count >= 5:
        return {"next_agent": "FINISH", "iteration_count": count + 1}
    
    # 正常逻辑
    # ...
    return {"next_agent": next_agent, "iteration_count": count + 1}
```

### 💡 第5周任务

1. 运行 `test/multi_agent_team.py` 先理解基础团队
2. 运行 `test/multi_agent_with_human.py` 体验人工干预
3. 运行 `test/multi_agent_official.py` 体验官方 interrupt 机制
4. **亲手制造每个陷阱**：
   - 故意用 `invoke()` 拿数据 → 看它返回 None
   - 故意不设 as_node → 看状态覆盖失败
   - 故意 temperature=0 → 看重写结果一样
   - 故意不加计数器 → 看 Supervisor 死循环

> **只有亲手踩过每个坑，你才算真正理解了人工干预。**

---

## 第6周：LLM 动态调度 Supervisor

### 🤔 为什么需要动态调度？

上周的 Supervisor 是用 if-else 判断的。如果任务更复杂、不确定，让 LLM 自己决定更灵活：

```python
# 代码判断（适合确定流程）
if not notes: → researcher
elif not draft: → writer
else: → finish

# LLM 判断（适合不确定流程）
# "用户要求调研后直接写，不审核"
# "用户要求先审核再写"
# "用户说如果文章质量好就直接发布"
```

### 🧩 实现方式：结构化输出

```python
from pydantic import BaseModel, Field

# 定义 Supervisor 的输出格式
class SupervisorDecision(BaseModel):
    """主管决策的结构化输出"""
    next_agent: str = Field(description="下一步要执行的 Agent 名称")
    reason: str = Field(description="决策理由")
    confidence: float = Field(description="决策置信度 0-1")

# 给模型绑定结构化输出
structured_model = model.with_structured_output(SupervisorDecision)

def supervisor_node(state):
    """LLM 动态调度"""
    prompt = f"""当前团队状态：
    - 主题：{state.get('topic', '无')}
    - 调研笔记：{'有' if state.get('research_notes') else '无'}
    - 初稿：{'有' if state.get('draft') else '无'}
    - 编辑意见：{'有' if state.get('edit_feedback') else '无'}
    
    请决定下一步派谁执行（researcher/writer/editor/FINISH）"""
    
    decision = structured_model.invoke([HumanMessage(content=prompt)])
    print(f"👔 [Supervisor] 决策：{decision.next_agent}（理由：{decision.reason}）")
    return {"next_agent": decision.next_agent}
```

### 什么时候用 if-else，什么时候用 LLM？

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 固定流水线 | if-else | 稳定、无 Token 消耗 |
| 动态决策 | LLM | 灵活，适应复杂情况 |
| 只有 2-3 个分支 | if-else | 简单可靠 |
| 10+ 个可能分支 | LLM | 代码写不完所有情况 |

### 💡 第6周任务

1. 运行 `test/multi_agent_humantool.py`
2. 对比 if-else Supervisor 和 LLM Supervisor 的优缺点
3. 实验：给 LLM Supervisor 的错误输出加容错处理

---

## 第7周：持久化存储

### 🤔 为什么需要持久化？

```python
# MemorySaver：存在内存里
# ✅ 简单、调试方便
# ❌ 程序重启 → 所有对话丢失
# ❌ 不能跨进程共享

# SqliteSaver：存在磁盘文件里
# ✅ 程序重启 → 对话还在
# ✅ 可以备份
# ❌ 单机使用
```

### 🧩 SqliteSaver 使用

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建持久化存储
memory = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")

# 编译时传入
app = workflow.compile(checkpointer=memory)

# 用法和 MemorySaver 完全一样！
config = {"configurable": {"thread_id": "team_001"}}
```

### 🔍 验证持久化

```python
# 第一次运行
app.stream(inputs, config)

# 关闭程序，重新打开
# 新建一个 Python 文件：
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "team_001"}}

# 恢复状态！
state = app.get_state(config)
print(state.values.get("draft", "无")[:100])  # ← 上次的草稿还在！
```

### 其他 Checkpointer

| 类型 | 安装 | 适用场景 |
|------|------|---------|
| MemorySaver | 内置 | 开发调试 |
| SqliteSaver | `langgraph-checkpoint-sqlite` | 单机生产 |
| PostgresSaver | `langgraph-checkpoint-postgres` | 分布式生产 |

### 💡 第7周任务

1. 把团队代码从 MemorySaver 换成 SqliteSaver
2. 运行一次，查看生成的 `.db` 文件
3. 重启程序，看状态是否恢复
4. 用 `get_state_history()` 查看所有历史快照

---

## 🏆 里程碑考核：带人工审批的多 Agent 团队

### 要求

```python
# 你的团队需要：
# 1. Supervisor 调度（Researcher → Writer → Editor）
# 2. 人工审核节点（写作完成后暂停）
# 3. 支持 continue / rewrite / stop
# 4. SqliteSaver 持久化
# 5. 防死循环计数器

# 测试流程：
输入主题 → 团队调研 → 团队写作 → ⏸️ 你审核
→ [continue] → 显示最终结果
→ [rewrite]  → 团队重写（带上你的修改意见）
→ [stop]     → 终止
```

### 验收清单

| 能力 | 通过标准 |
|------|---------|
| Supervisor 调度 | 能正确按顺序派活 |
| 人工审核 | 写作后暂停，等待人类输入 |
| 重写循环 | rewrite 后能重新执行并产出新内容 |
| 持久化 | 重启程序后状态还在 |
| 防死循环 | 不加计数器验证是否会死循环 |
| 理解陷阱 | 能解释增量 vs 快照、状态覆盖、重放缓存 |

---

## 🚀 第二阶段总结

### 你学到的核心概念

```
✅ Supervisor 模式     — 主管+专家团队架构
✅ TeamState           — 多 Agent 共享白板
✅ interrupt_before    — 编译时设断点
✅ interrupt() 函数    — 运行时暂停点
✅ Command(resume=)    — 从断点恢复
✅ SqliteSaver         — 持久化存储
✅ 结构化输出          — LLM 动态决策
```

### 你踩过的坑（这是最宝贵的经验）

| 坑 | 一句话总结 |
|----|-----------|
| Supervisor 无限循环 | 必须加硬性计数器 |
| invoke 拿不到完整状态 | 用 get_state().values |
| update_state 覆盖失败 | 用 as_node 或追加消息 |
| 重写结果一样 | temperature≠0 + 注入反馈 |
| 语义死循环 | Supervisor + 状态修改 + while 的致命组合 |

### 下一步

你已经掌握了多 Agent 协作的核心。第三阶段将进入企业级架构：并行、子图、Stream。

> 📖 打开 `第三阶段-企业级实战.md` 继续学习
