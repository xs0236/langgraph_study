import os
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, HumanMessage

load_dotenv()

# 1. 初始化模型
model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0
)

# 2. 定义团队共享状态 (Shared State)
# 这是多 Agent 协作的"公共白板"，所有 Agent 都能看到并更新它
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str                # 文章主题
    research_notes: str       # 调研笔记
    draft: str                # 初稿
    edit_feedback: str        # 编辑意见
    next_agent: str           # 主管决定下一步由谁执行

# 3. 定义专业 Agent 节点
def researcher_node(state: TeamState):
    """调研员：只负责搜集资料"""
    print("🕵️ [Researcher] 正在搜集资料...")
    prompt = f"你是一位资深调研员。请针对主题 '{state['topic']}' 搜集关键信息，并输出一份简短的调研笔记。不要写文章，只输出结构化的调研资料。"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"research_notes": response.content, "next_agent": "supervisor"}

def writer_node(state: TeamState):
    """撰稿人：根据调研写初稿"""
    print("✍️ [Writer] 正在撰写初稿...")
    prompt = f"你是一位资深撰稿人。请根据以下调研笔记撰写一篇短文初稿：\n{state['research_notes']}\n要求：逻辑清晰，语言通俗。"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "next_agent": "supervisor"}

def editor_node(state: TeamState):
    """编辑：审核并提出意见"""
    print("📝 [Editor] 正在审核文章...")
    prompt = f"你是一位严格的编辑。请审核以下文章初稿，并给出具体的修改意见：\n{state['draft']}\n要求：只提意见，不要直接重写。"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"edit_feedback": response.content, "next_agent": "FINISH"}

# 4. 定义主管节点 (Supervisor)
# def supervisor_node(state: TeamState):
#     """主管：负责拆解任务并分配给专业 Agent"""
#     print("👔 [Supervisor] 正在思考下一步...")
#     prompt = """你是一个团队主管。根据当前进度，决定下一步由谁执行：
#     - 如果还没有调研笔记，分配给 'researcher'
#     - 如果有调研笔记但没有初稿，分配给 'writer'
#     - 如果有初稿但没有审核意见，分配给 'editor'
#     - 如果审核完成，回复 'FINISH'
#     只回复对应的角色名称或 FINISH，不要说废话。"""
    
#     response = model.invoke([HumanMessage(content=prompt)])
#     next_agent = response.content.strip().lower()
    
#     # 容错处理
#     if next_agent not in ["researcher", "writer", "editor", "finish"]:
#         next_agent = "researcher"
        
#     return {"next_agent": next_agent}

# def supervisor_node(state: TeamState):
#     """主管：负责拆解任务并分配给专业 Agent"""
#     print("👔 [Supervisor] 正在思考下一步...")
#     prompt = """你是一个团队主管。根据当前进度，决定下一步由谁执行：- 如果还没有调研笔记，分配给 'researcher' - 如果有调研笔记但没有初稿，分配给 'writer' - 如果有初稿但没有审核意见，分配给 'editor' - 如果审核完成，回复 'FINISH'只回复对应的角色名称或 FINISH，不要说废话。"""

#     # 【修改点】：将 HumanMessage 改为 HumanMessage，或者使用元组格式
#     # GLM 对纯 HumanMessage 有时支持不好，改用用户提问的方式最稳妥
#     from langchain_core.messages import HumanMessage

#     response = model.invoke([HumanMessage(content=prompt)])
#     next_agent = response.content.strip().lower()

#     # 容错处理：防止大模型回复了额外的标点符号或废话
#     if next_agent not in ["researcher", "writer", "editor", "finish"]:
#         if "researcher" in next_agent: next_agent = "researcher"
#         elif "writer" in next_agent: next_agent = "writer"
#         elif "editor" in next_agent: next_agent = "editor"
#         elif "finish" in next_agent: next_agent = "FINISH"
#         else: next_agent = "researcher" # 默认兜底

#     return {"next_agent": next_agent}

def supervisor_node(state: TeamState):
    """主管：负责拆解任务并分配给专业 Agent"""
    print("👔 [Supervisor] 正在思考下一步...")
    
    # 【核心修改】：根据共享状态 state 里的数据来做决策
    if not state.get("research_notes"):
        # 如果没有调研笔记，就让调研员干活
        next_agent = "researcher"
    elif not state.get("draft"):
        # 如果有调研笔记，但没有初稿，就让撰稿人干活
        next_agent = "writer"
    elif not state.get("edit_feedback"):
        # 如果有初稿，但没有编辑意见，就让编辑干活
        next_agent = "editor"
    else:
        # 所有工作都完成了
        next_agent = "FINISH"

    print(f"👔 [Supervisor] 决策：下一步交给 [{next_agent}]")
    return {"next_agent": next_agent}


# 5. 构建图 (Graph)
workflow = StateGraph(TeamState)

# 添加节点
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

# 设置入口点
workflow.set_entry_point("supervisor")

# 添加条件边：主管根据 next_agent 决定路由
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "researcher": "researcher",
        "writer": "writer",
        "editor": "editor",
        "FINISH": END
    }
)

# 所有专业 Agent 干完活后，都必须回到主管那里汇报
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("editor", "supervisor")

# 6. 编译并运行
app = workflow.compile()

# 测试任务
inputs = {"topic": "人工智能对现代教育的影响", "messages": []}

print("--- 团队开始协作 ---")
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"✅ 节点 [{key}] 执行完毕，当前状态更新: {list(value.keys())}")
print("--- 协作结束 ---")