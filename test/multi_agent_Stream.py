import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ========== 全局模型配置 ==========
model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7
)

# ========== 【子图】写作团队状态 ==========
class WritingState(TypedDict):
    topic: str
    research_notes: str
    draft: str
    edit_feedback: str
    previous_feedback: str
    angle: str
    partial_notes: Annotated[list, add_messages]
    next_agent: str

# ========== 【子图】专业 Agent 节点 ==========

def sub_researcher_node(state: WritingState):
    angle = state.get("angle", "综合角度")
    print(f"   🕵️ [Sub-Researcher - {angle}] 正在从{angle}搜集资料...")
    prompt = f"""你是一位资深调研员。请针对主题 '{state['topic']}' 从【{angle}】搜集关键信息，输出简短的调研笔记。
请专注于{angle}的相关内容。
"""
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"      → {angle} 调研完成")
    return {"partial_notes": [f"【{angle}】\n{response.content}"]}

def sub_aggregator_node(state: WritingState):
    print("   📦 [Sub-Aggregator] 正在汇总各角度调研结果...")
    partial_notes = state.get("partial_notes", [])
    if not partial_notes:
        return {"research_notes": "无调研资料", "next_agent": "sub_supervisor"}
    
    combined = "\n\n".join([str(note) for note in partial_notes])
    prompt = f"""请将以下从不同角度搜集的调研笔记汇总成一份完整、连贯的调研报告：

{combined}

请整合去重，输出一份结构清晰的调研笔记。
"""
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"      → 汇总完成，共整合 {len(partial_notes)} 个角度的资料")
    return {
        "research_notes": response.content,
        "partial_notes": [],
        "next_agent": "sub_supervisor"
    }

def sub_writer_node(state: WritingState):
    print("   ✍️ [Sub-Writer] 正在撰写初稿...")
    feedback = state.get("previous_feedback", "")
    prompt = f"""你是一位资深撰稿人。请根据以下调研笔记撰写短文初稿：

调研笔记：
{state['research_notes']}
"""
    if feedback:
        prompt += f"""

【⚠️ 编辑修改意见，请务必参考改进】：
{feedback}
"""
    response = model.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "next_agent": "sub_supervisor"}

def sub_editor_node(state: WritingState):
    print("   📝 [Sub-Editor] 正在审核文章...")
    prompt = f"""你是一位严格的编辑。请审核以下初稿并给出详细的修改意见：

{state['draft']}

请从以下维度给出具体、可执行的修改建议：
1. 标题是否吸引人
2. 结构是否清晰
3. 内容是否充实
4. 语言是否流畅
"""
    response = model.invoke([HumanMessage(content=prompt)])
    return {"edit_feedback": response.content, "next_agent": "sub_supervisor"}

def sub_supervisor_node(state: WritingState):
    print("   👔 [Sub-Supervisor] 正在思考下一步...")
    
    research_notes = state.get("research_notes", "")
    draft = state.get("draft", "")
    edit_feedback = state.get("edit_feedback", "")
    partial_notes = state.get("partial_notes", [])
    
    if not research_notes and len(partial_notes) == 0:
        next_agent = "parallel_research"
        reasoning = "启动并行调研"
    elif not research_notes and len(partial_notes) >= 3:
        next_agent = "aggregator"
        reasoning = "汇总调研资料"
    elif not draft:
        next_agent = "writer"
        reasoning = "开始撰写初稿"
    elif not edit_feedback:
        next_agent = "editor"
        reasoning = "开始编辑审核"
    else:
        next_agent = "FINISH"
        reasoning = "写作子图完成"
    
    print(f"      → 决策：{next_agent}")
    return {"next_agent": next_agent}

def sub_route_research(state: WritingState):
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "parallel_research":
        print("   🚀 [Sub-Supervisor] 启动并行调研！")
        return [
            Send("sub_researcher", {"topic": state["topic"], "angle": "技术角度", "partial_notes": []}),
            Send("sub_researcher", {"topic": state["topic"], "angle": "教育角度", "partial_notes": []}),
            Send("sub_researcher", {"topic": state["topic"], "angle": "伦理角度", "partial_notes": []})
        ]
    return next_agent

# ========== 【核心】构建写作子图 ==========

def build_writing_subgraph():
    subgraph = StateGraph(WritingState)
    
    subgraph.add_node("sub_supervisor", sub_supervisor_node)
    subgraph.add_node("sub_researcher", sub_researcher_node)
    subgraph.add_node("sub_aggregator", sub_aggregator_node)
    subgraph.add_node("sub_writer", sub_writer_node)
    subgraph.add_node("sub_editor", sub_editor_node)
    
    subgraph.set_entry_point("sub_supervisor")
    
    subgraph.add_conditional_edges(
        "sub_supervisor",
        sub_route_research,
        {
            "aggregator": "sub_aggregator",
            "writer": "sub_writer",
            "editor": "sub_editor",
            "FINISH": END,
            "sub_supervisor": "sub_supervisor"
        }
    )
    
    subgraph.add_edge("sub_researcher", "sub_supervisor")
    subgraph.add_edge("sub_aggregator", "sub_supervisor")
    subgraph.add_edge("sub_writer", "sub_supervisor")
    subgraph.add_edge("sub_editor", "sub_supervisor")
    
    return subgraph.compile()

# ========== 【主图】主状态 ==========
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    draft: str
    edit_feedback: str
    previous_feedback: str
    next_agent: str

# ========== 【主图】节点 ==========

def main_supervisor_node(state: MainState):
    print("👔 [Main-Supervisor] 主调度...")
    
    if not state.get("draft"):
        print("   → 启动写作子图")
        return {"next_agent": "writing_team"}
    else:
        print("   → 写作完成，等待人工审批")
        return {"next_agent": "FINISH"}

# ========== 【主图】构建 ==========

def build_main_graph():
    workflow = StateGraph(MainState)
    
    workflow.add_node("main_supervisor", main_supervisor_node)
    
    writing_subgraph = build_writing_subgraph()
    workflow.add_node("writing_team", writing_subgraph)
    
    workflow.set_entry_point("main_supervisor")
    
    workflow.add_conditional_edges(
        "main_supervisor",
        lambda state: state["next_agent"],
        {
            "writing_team": "writing_team",
            "FINISH": END,
            "main_supervisor": "main_supervisor"
        }
    )
    
    workflow.add_edge("writing_team", "main_supervisor")
    
    return workflow

# ========== 【核心改动】Stream 实时输出 ==========

def run_stream(app, inputs, config):
    """封装 stream 执行，实时打印每个节点的输出"""
    print("\n" + "="*60)
    print("🚀 启动 Stream 实时输出")
    print("="*60)
    
    for chunk in app.stream(inputs, config=config):
        # chunk 格式: {节点名: {状态更新}}
        for node_name, node_output in chunk.items():
            if node_name == "__end__":
                continue
            # 打印节点执行信息（简洁版）
            print(f"📡 [Stream] 节点 '{node_name}' 执行完成")
    
    print("="*60)
    print("✅ Stream 执行完毕")
    print("="*60)

# ========== 运行 ==========

with SqliteSaver.from_conn_string("team_stream.db") as memory:
    workflow = build_main_graph()
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "team_stream_001"}}
    inputs = {
        "topic": "人工智能对现代教育的影响",
        "messages": [("user", "帮我写一篇关于AI教育的文章")],
        "previous_feedback": ""
    }
    
    # ✅ 改用 stream 而不是 invoke
    run_stream(app, inputs, config)
    
    # ========== 人工审批循环 ==========
    while True:
        state = app.get_state(config).values
        print("\n" + "-"*60)
        print("⏸️ [系统] 等待人类审批！")
        print(f"📄 文章初稿: {str(state.get('draft', '暂无'))[:150]}...")
        print(f"💬 编辑意见: {str(state.get('edit_feedback', '暂无'))[:150]}...")
        
        choice = input("\n👤 人类主管，请决定 (y=批准发布 / n=打回重写): ").strip().lower()
        
        if choice == 'y':
            print("✅ 批准发布！流程结束。")
            break
        elif choice == 'n':
            print("❌ 打回重写！重新启动写作子图...")
            current_feedback = state.get("edit_feedback", "")
            
            app.update_state(config, {
                "draft": None,
                "edit_feedback": None,
                "previous_feedback": current_feedback,
                "next_agent": "main_supervisor"
            })
            
            # ✅ 打回后也改用 stream
            run_stream(app, None, config)
        else:
            print("无效输入，请重新选择。")
    
    print("\n" + "="*60)
    print("🏁 流程彻底结束")
    print("="*60)