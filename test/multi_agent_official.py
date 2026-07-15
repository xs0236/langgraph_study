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
    temperature=0.7
)

class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str                
    research_notes: str       
    draft: str                
    edit_feedback: str        
    previous_feedback: str    # ✅ 新增：保存历史反馈供改进参考
    next_agent: str           

def researcher_node(state: TeamState):
    print("🕵️ [Researcher] 正在搜集资料...")
    feedback = state.get("previous_feedback", "")
    
    prompt = f"""你是一位资深调研员。请针对主题 '{state['topic']}' 搜集关键信息，输出简短的调研笔记。
"""
    if feedback:
        prompt += f"""

【⚠️ 上一轮编辑反馈，请务必参考改进】：
{feedback}

请根据以上反馈重新调研，注意避免之前指出的问题，尝试从不同角度切入。
"""
    
    response = model.invoke([HumanMessage(content=prompt)])
    return {"research_notes": response.content, "next_agent": "supervisor"}

def writer_node(state: TeamState):
    print("✍️ [Writer] 正在撰写初稿...")
    feedback = state.get("previous_feedback", "")
    
    prompt = f"""你是一位资深撰稿人。请根据以下调研笔记撰写短文初稿：

调研笔记：
{state['research_notes']}
"""
    if feedback:
        prompt += f"""

【⚠️ 编辑修改意见，请务必参考并解决以下问题】：
{feedback}

请根据以上意见重新撰写，确保所有问题都得到改进。尝试不同的标题和结构。
"""
    else:
        prompt += "\n请撰写一篇高质量的短文初稿。"
    
    response = model.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "next_agent": "supervisor"}

def editor_node(state: TeamState):
    print("📝 [Editor] 正在审核文章...")
    prompt = f"""你是一位严格的编辑。请审核以下初稿并给出详细的修改意见：

{state['draft']}

请从以下维度给出具体、可执行的修改建议：
1. 标题是否吸引人
2. 结构是否清晰
3. 内容是否充实
4. 语言是否流畅
"""
    response = model.invoke([HumanMessage(content=prompt)])
    return {"edit_feedback": response.content, "next_agent": "supervisor"}

def supervisor_node(state: TeamState):
    print("👔 [Supervisor] 正在思考下一步...")
    if not state.get("research_notes"):
        next_agent = "researcher"
    elif not state.get("draft"):
        next_agent = "writer"
    elif not state.get("edit_feedback"):   # 只有 edit_feedback 为空时才走 editor
        next_agent = "editor"
    else:
        next_agent = "FINISH"
    return {"next_agent": next_agent}

workflow = StateGraph(TeamState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "researcher": "researcher", 
        "writer": "writer", 
        "editor": "editor", 
        "FINISH": END,
        "supervisor": "supervisor"
    }
)
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("editor", "supervisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "team_perfect_003"}}
inputs = {
    "topic": "人工智能对现代教育的影响", 
    "messages": [("user", "帮我写一篇关于AI教育的文章")],
    "previous_feedback": ""  # 初始化
}

print("\n--- 启动团队 ---")
app.invoke(inputs, config=config)

while True:
    state = app.get_state(config).values
    print("\n⏸️ [系统] 团队已完成初步工作，等待人类审批！")
    print(f"📄 文章初稿: {str(state.get('draft', '暂无'))[:150]}...")
    print(f"💬 编辑意见: {str(state.get('edit_feedback', '暂无'))[:150]}...")
    
    choice = input("\n👤 人类主管，请决定 (y=批准发布 / n=打回重写): ").lower()
    
    if choice == 'y':
        print("✅ 批准发布！流程结束。")
        break
    elif choice == 'n':
        print("❌ 打回重写！清空状态，让团队重来...")
        
        # 把当前编辑反馈保存到 previous_feedback，同时清空 edit_feedback
        current_feedback = state.get("edit_feedback", "")
        
        app.update_state(config, {
            "research_notes": None,
            "draft": None,
            "edit_feedback": None,           # ✅ 清空，让 editor 重新审核
            "previous_feedback": current_feedback,  # ✅ 保存历史反馈供改进
            "next_agent": "supervisor"
        })
        app.invoke(None, config=config)
    else:
        print("无效输入，请重新选择。")

print("\n--- 流程彻底结束 ---")