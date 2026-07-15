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

# 2. 定义团队共享状态
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str                
    research_notes: str       
    draft: str                
    edit_feedback: str        
    next_agent: str      
    iteration_count: int      # 记录主管调度的次数     

# 3. 定义专业 Agent 节点
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
    return {"edit_feedback": response.content, "next_agent": "supervisor"}

# # 4. 定义主管节点
# def supervisor_node(state: TeamState):
#     print("👔 [Supervisor] 正在思考下一步...")
#     if not state.get("research_notes"):
#         next_agent = "researcher"
#     elif not state.get("draft"):
#         next_agent = "writer"
#     elif not state.get("edit_feedback"):
#         next_agent = "editor"
#     else:
#         next_agent = "FINISH"
#     return {"next_agent": next_agent}
# 4. 定义主管节点
def supervisor_node(state: TeamState):
    print("👔 [Supervisor] 正在思考下一步...")
    
    # 【核心修改】：获取当前迭代次数，默认是 0
    count = state.get("iteration_count", 0)
    
    # 无论状态如何，只要调度超过 5 次，强制结束，防止死循环！
    if count >= 5:
        print("⚠️ [Supervisor] 达到最大调度次数，强制结束流程！")
        return {"next_agent": "FINISH", "iteration_count": count + 1}

    if not state.get("research_notes"):
        next_agent = "researcher"
    elif not state.get("draft"):
        next_agent = "writer"
    elif not state.get("edit_feedback"):
        next_agent = "editor"
    else:
        next_agent = "FINISH"
        
    return {"next_agent": next_agent, "iteration_count": count + 1}

# 5. 构建图
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

# 6. 【核心修改 1】：引入记忆，并在 "supervisor" 节点前设置断点
memory = MemorySaver()
# 当流程走到 supervisor 之前，如果状态已经包含了 edit_feedback，我们就让它暂停
# 这里我们简单粗暴地在 supervisor 前中断，方便演示
app = workflow.compile(checkpointer=memory, interrupt_before=["supervisor"])

# # 7. 运行多轮对话测试
# config = {"configurable": {"thread_id": "team_002"}}

# # === 第一轮：启动团队 ===
# print("\n--- 第一轮对话：启动团队 ---")
# inputs1 = {"topic": "人工智能对现代教育的影响", "messages": [("user", "帮我写一篇关于AI教育的文章")]}

# # 运行直到遇到断点暂停
# final_state = app.invoke(inputs1, config=config)

# complete_state = app.get_state(config).values

# print("\n⏸️ [系统] 团队已完成初步工作，程序已暂停等待人类审批！")

# draft_content = complete_state.get("draft", "暂无初稿")
# edit_feedback_content = complete_state.get("edit_feedback", "暂无意见")

# print(f"📄 当前文章初稿:\n{str(draft_content)[:100]}...") # 只打印前100个字符
# print(f"💬 编辑意见: {edit_feedback_content}")

# # === 第二轮：人类介入审批 ===
# user_choice = input("\n👤 人类主管，请做出决定：\n1. 输入 'y' 批准发布\n2. 输入 'n' 打回重写\n请选择 (y/n): ")

# if user_choice.lower() == 'y':
#     print("\n✅ [人类] 批准发布！")
#     # 【核心修改 2】：传入 None，程序会从断点处继续执行
#     # 因为 supervisor 检查到 edit_feedback 存在，会直接返回 FINISH
#     final_state = app.invoke(None, config=config)
#     print("🎉 文章已正式发布！")
# else:
#     print("\n❌ [人类] 文章不合格，打回重写！")
#     # 如果打回，我们需要清空之前的 edit_feedback，让主管重新调度
#     # 这里可以通过 update_state 来实现（LangGraph 的高级特性）
#     app.update_state(config, {"edit_feedback": None, "draft": None})
#     print("🔄 正在让团队重新开始...")
#     final_state = app.invoke(None, config=config)
#     print("🎉 文章已重新生成并发布！")

# print("\n--- 流程结束 ---")

# 7. 运行多轮对话测试
config = {"configurable": {"thread_id": "team_003"}} # 换个新的 thread_id 避免缓存干扰

# === 第一轮：启动团队 ===
print("\n--- 第一轮对话：启动团队 ---")
inputs1 = {"topic": "人工智能对现代教育的影响", "messages": [("user", "帮我写一篇关于AI教育的文章")]}

# 运行直到遇到断点暂停
# 【修复 1】：在遇到 interrupt 时，final_state 实际上已经包含了完整的状态
final_state = app.invoke(inputs1, config=config)

print("\n⏸️ [系统] 团队已完成初步工作，程序已暂停等待人类审批！")

# 【修复 1】：直接从 final_state 中安全读取数据
draft_content = final_state.get("draft", "暂无初稿")
edit_feedback_content = final_state.get("edit_feedback", "暂无意见")

print(f"📄 当前文章初稿:\n{str(draft_content)[:100]}...") # 只打印前100个字符
print(f"💬 编辑意见: {edit_feedback_content}")

# === 第二轮：人类介入审批 ===
user_choice = input("\n👤 人类主管，请做出决定：\n1. 输入 'y' 批准发布\n2. 输入 'n' 打回重写\n请选择 (y/n): ")

if user_choice.lower() == 'y':
    print("\n✅ [人类] 批准发布！")
    # 传入 None，程序会从断点处继续执行
    final_state = app.invoke(None, config=config)
    print("🎉 文章已正式发布！")
# else:
#     print("\n❌ [人类] 文章不合格，打回重写！")
    
#     # 【修复 2】：使用 update_state 时，必须指定 as_node="editor" 
#     # 这告诉 LangGraph：把这次更新当作是 editor 节点产生的，从而【覆盖】掉之前的旧状态
#     app.update_state(
#         config, 
#         {"edit_feedback": None, "draft": None, "research_notes": None}, # 顺便把调研笔记也清了，让团队从头开始
#         as_node="editor" 
#     )
    
#     print("🔄 正在让团队重新开始...")
#     final_state = app.invoke(None, config=config)
    
#     # 打印最终结果
#     print("\n--- 最终结果 ---")
#     # 从完整的状态中获取数据
#     complete_state = app.get_state(config).values
#     print(f"新版文章初稿:\n{complete_state.get('draft', '暂无')}")

else:
    print("\n❌ [人类] 文章不合格，打回重写！")
    
    # --- 核心修改 ---
    # 1. 不再使用 as_node，避免触发 supervisor 的断点
    # 2. 通过向 messages 添加一条新消息来“唤醒”工作流
    # 3. 同时清空其他字段，为新一轮创作做准备
    app.update_state(
        config, 
        {
            "edit_feedback": None, 
            "draft": None, 
            "research_notes": None,
             "iteration_count": 0,  # 打回重写时，把计数器清零
            "messages": [HumanMessage(content="人类主管：文章不合格，请重新调研并撰写。")]
        }
    )
    
    print("🔄 正在让团队重新开始...")
    
    # 4. 使用一个循环，让工作流自动运行直到结束
    #    因为没有了断点干扰，它会一口气跑完 researcher -> writer -> editor
    while True:
        # 传入 None 让工作流从断点处继续
        event = app.invoke(None, config=config)
        
        # 如果 event 为空，说明工作流已经走到 END，彻底结束了
        if not event:
            break
            
    # 5. 循环结束后，再从完整状态中获取最终结果
    complete_state = app.get_state(config).values
    print("\n--- 最终结果 ---")
    print(f"新版文章初稿:\n{complete_state.get('draft', '暂无')}")

print("\n--- 流程结束 ---")