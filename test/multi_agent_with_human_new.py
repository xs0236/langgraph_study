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

model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0
)

# 1. 定义团队共享状态
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str                
    research_notes: str       
    draft: str                
    edit_feedback: str        
    next_agent: str           

# 2. 定义专业 Agent 节点
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
    # 【核心修改】：编辑做完后，不再直接 FINISH，而是回到主管那里等待最终决策
    return {"edit_feedback": response.content, "next_agent": "supervisor"}

# 3. 定义主管节点 (Supervisor)
# def supervisor_node(state: TeamState):
#     print("👔 [Supervisor] 正在思考下一步...")
#     if not state.get("research_notes"):
#         next_agent = "researcher"
#     elif not state.get("draft"):
#         next_agent = "writer"
#     elif not state.get("edit_feedback"):
#         next_agent = "editor"
#     else:
#         # 所有工作都完成了，交由外部循环处理人工审批
#         next_agent = "FINISH"
#     return {"next_agent": next_agent}
def supervisor_node(state: TeamState):
    print("👔 [Supervisor] 正在思考下一步...")
    
    # 【核心修改】：检查最后一条消息，看看人类是否要求重写
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1].content.lower()
        if "不合格" in last_message or "重写" in last_message or "重新" in last_message:
            print("👔 [Supervisor] 收到人类重写指令，清空旧状态，重新开始！")
            # 返回清空状态的指令，让团队从头开始
            return {
                "research_notes": None,
                "draft": None,
                "edit_feedback": None,
                "next_agent": "researcher"
            }

    # 常规状态判断
    if not state.get("research_notes"):
        next_agent = "researcher"
    elif not state.get("draft"):
        next_agent = "writer"
    elif not state.get("edit_feedback"):
        next_agent = "editor"
    else:
        next_agent = "FINISH"
        
    return {"next_agent": next_agent}

# 4. 构建图
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

# 5. 【核心修改】：只保留 Memory，去掉所有 interrupt 断点！
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 6. 运行测试
config = {"configurable": {"thread_id": "team_004"}} 

print("\n--- 启动团队 ---")
inputs = {"topic": "人工智能对现代教育的影响", "messages": [("user", "帮我写一篇关于AI教育的文章")]}

# 让团队自动跑完一轮流水线
app.invoke(inputs, config=config)

# 团队跑完后，我们在 Python 代码层面进行人工审批
while True:
    state = app.get_state(config).values
    print("\n⏸️ [系统] 团队已完成初步工作，等待人类审批！")
    # print(f"📄 文章初稿: {state['draft'][:50]}...")
    # print(f"💬 编辑意见: {state['edit_feedback']}")

    # 【修复点】：使用 .get() 安全获取，并用 str() 包裹防止 None 报错
    draft_text = str(state.get('draft', '暂无初稿'))
    edit_feedback_text = str(state.get('edit_feedback', '暂无意见'))

    print(f"📄 文章初稿: {draft_text[:50]}...")
    print(f"💬 编辑意见: {edit_feedback_text[:50]}...")



    
    choice = input("\n👤 人类主管，请决定 (y=批准发布 / n=打回重写): ").lower()
    
    if choice == 'y':
        print("✅ 批准发布！流程结束。")
        break
    elif choice == 'n':
        # print("❌ 打回重写！清空状态，让团队重来...")
        # # 直接清空状态，重新 invoke，逻辑极其简单清晰
        # app.update_state(config, {
        #     "research_notes": None,
        #     "draft": None,
        #     "edit_feedback": None
        # })
        # app.invoke(None, config=config)
        print("❌ 打回重写！注入新指令，让团队重来...")
        
        # 【核心修改】：不要清空旧状态，而是往 messages 里追加一条新消息！
        # 只要 messages 发生了变化，LangGraph 就会认为有新输入，从而唤醒工作流
        app.update_state(config, {
            "messages": [HumanMessage(content="人类主管：上一版文章不合格，请重新调研并撰写。")]
        })
        
        # 传入 None，让工作流从暂停的地方（supervisor）继续执行
        app.invoke(None, config=config)
    else:
        print("无效输入，请重新选择。")

print("\n--- 流程彻底结束 ---")