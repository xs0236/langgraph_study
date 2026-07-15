import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. 初始化模型
model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0
)

# 2. 定义团队共享状态 (Shared State)
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str                
    research_notes: str       
    draft: str                
    edit_feedback: str        
    next_agent: str           

# 3. 定义专业 Agent 节点 (与之前相同)
def researcher_node(state: TeamState):
    print("🕵️ [Researcher] 正在搜集资料...")
    prompt = f"你是一位资深调研员。请针对主题 '{state['topic']}' 搜集关键信息，输出简短的调研笔记。"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"research_notes": response.content, "next_agent": "supervisor"}

def writer_node(state: TeamState):
    print("✍️ [Writer] 正在撰写初稿...")
    prompt = f"你是一位资深撰稿人。请根据以下调研笔记撰写短文初稿：\n{state['research_notes']}"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "next_agent": "supervisor"}

def editor_node(state: TeamState):
    print("📝 [Editor] 正在审核文章...")
    prompt = f"你是一位严格的编辑。请审核以下初稿并给出修改意见：\n{state['draft']}"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"edit_feedback": response.content, "next_agent": "FINISH"}

# 4. 定义主管节点 (Supervisor)
def supervisor_node(state: TeamState):
    print("👔 [Supervisor] 正在思考下一步...")
    if not state.get("research_notes"):
        next_agent = "researcher"
    elif not state.get("draft"):
        next_agent = "writer"
    elif not state.get("edit_feedback"):
        next_agent = "editor"
    else:
        next_agent = "FINISH"
    return {"next_agent": next_agent}

# 5. 构建图 (Graph)
workflow = StateGraph(TeamState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {"researcher": "researcher", "writer": "writer", "editor": "editor", "FINISH": END}
)
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("editor", "supervisor")

# 6. 引入记忆并编译
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 7. 运行多轮对话测试
config = {"configurable": {"thread_id": "team_001"}}

# === 第一轮对话：启动团队写文章 ===
print("\n--- 第一轮对话：启动团队 ---")
inputs1 = {"topic": "人工智能对现代教育的影响", "messages": [("user", "帮我写一篇关于AI教育的文章")]}
final_state_1 = app.invoke(inputs1, config=config)
print(f"🎉 第一轮结束！文章初稿已完成。")

# === 第二轮对话：基于上一轮的草稿继续工作 ===
print("\n--- 第二轮对话：追加任务（测试记忆） ---")
# 注意：这里不需要再传 topic，也不需要重置状态
# 只需要传入新的消息，Agent 会自动从 memory 中恢复之前的白板！
inputs2 = {"messages": [("user", "请把刚才那篇文章的标题改得更吸引人一点，并重新生成一版")]}
final_state_2 = app.invoke(inputs2, config=config)

print("\n--- 最终结果展示 ---")
print(f"最终文章初稿:\n{final_state_2['draft']}")
print(f"编辑最终意见:\n{final_state_2['edit_feedback']}")