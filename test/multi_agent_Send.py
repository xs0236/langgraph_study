import os
import json
from dotenv import load_dotenv
from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ========== 1. 模型配置 ==========
model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7
)

# ========== 2. 定义团队共享状态 ==========
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str                
    research_notes: str       
    draft: str                
    edit_feedback: str        
    previous_feedback: str    
    next_agent: str           
    angle: str                 
    partial_notes: Annotated[list, add_messages]  # 各角度的部分调研结果

# ========== 3. 定义专业 Agent 节点 ==========

def researcher_node(state: TeamState):
    """并行调研节点"""
    angle = state.get("angle", "综合角度")
    print(f"🕵️ [Researcher - {angle}] 正在从{angle}搜集资料...")
    
    prompt = f"""你是一位资深调研员。请针对主题 '{state['topic']}' 从【{angle}】搜集关键信息，输出简短的调研笔记。
请专注于{angle}的相关内容，不要涉及其他角度。
"""
    
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"   → {angle} 调研完成")
    return {"partial_notes": [f"【{angle}】\n{response.content}"]}

def aggregator_node(state: TeamState):
    """汇总节点"""
    print("📦 [Aggregator] 正在汇总各角度调研结果...")
    
    partial_notes = state.get("partial_notes", [])
    if not partial_notes:
        print("   ⚠️ 没有调研资料")
        return {"research_notes": "无调研资料", "next_agent": "supervisor"}
    
    # 合并所有部分笔记
    combined = "\n\n".join([str(note) for note in partial_notes])
    
    # 让 LLM 汇总
    prompt = f"""你是一位资深编辑。请将以下从不同角度搜集的调研笔记汇总成一份完整、连贯的调研报告：

{combined}

请整合去重，输出一份结构清晰的调研笔记。
"""
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"   → 汇总完成，共整合 {len(partial_notes)} 个角度的资料")
    
    # ✅ 汇总后清空 partial_notes，避免重复汇总
    return {
        "research_notes": response.content, 
        "partial_notes": [],  # 清空
        "next_agent": "supervisor"
    }

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

# ========== 4. 智能主管节点 ==========

SUPERVISOR_SYSTEM_PROMPT = """你是一个智能调度主管，负责协调一个写作团队的工作。

团队成员：
- researcher（调研员）：从多个角度并行搜集资料
- writer（撰稿人）：根据调研笔记撰写文章初稿  
- editor（编辑）：审核文章初稿，给出修改意见

当前任务状态：
- 调研笔记：{research_notes_status}
- 已收集角度数：{partial_count}
- 文章初稿：{draft_status}
- 编辑意见：{edit_feedback_status}

调度规则：
1. 如果没有调研笔记且角度数 < 3 → 启动并行调研
2. 如果角度数 >= 3 但没有汇总笔记 → 选择 aggregator
3. 如果有调研笔记但没有初稿 → 选择 writer
4. 如果有初稿但没有编辑意见 → 选择 editor
5. 如果三者都已完成 → 选择 FINISH

【重要】请严格按以下 JSON 格式输出你的决策：

{{
  "next_agent": "parallel_research 或 aggregator 或 writer 或 editor 或 FINISH",
  "reasoning": "你的决策理由"
}}"""

def parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    return json.loads(content)

def supervisor_node(state: TeamState):
    print("👔 [Supervisor] 正在思考下一步...")
    
    research_notes = state.get("research_notes", "")
    draft = state.get("draft", "")
    edit_feedback = state.get("edit_feedback", "")
    partial_notes = state.get("partial_notes", [])
    
    research_notes_status = "✅ 已完成" if research_notes else "❌ 未完成"
    partial_count = len(partial_notes)
    draft_status = "✅ 已完成" if draft else "❌ 未完成"
    edit_feedback_status = "✅ 已完成" if edit_feedback else "❌ 未完成"
    
    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        research_notes_status=research_notes_status,
        partial_count=partial_count,
        draft_status=draft_status,
        edit_feedback_status=edit_feedback_status
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请根据当前任务状态，做出调度决策。只输出 JSON。")
    ]
    
    response = model.invoke(messages)
    content = response.content
    
    try:
        decision = parse_llm_json(content)
        next_agent = decision.get("next_agent", "FINISH")
        reasoning = decision.get("reasoning", "无")
    except (json.JSONDecodeError, Exception) as e:
        print(f"   ⚠️ JSON 解析失败: {e}")
        # 回退逻辑
        # if not research_notes and len(partial_notes) < 3:
        #     next_agent = "parallel_research"
        if not research_notes and len(partial_notes) == 0:
            next_agent = "parallel_research"  # 只有完全没有资料时才触发
        elif not research_notes and len(partial_notes) >= 3:
            next_agent = "aggregator"         # 有资料但未汇总 → 汇总
        elif not research_notes and len(partial_notes) >= 3:
            next_agent = "aggregator"
        elif not draft:
            next_agent = "writer"
        elif not edit_feedback:
            next_agent = "editor"
        else:
            next_agent = "FINISH"
        reasoning = "JSON 解析失败，使用回退逻辑"
    
    print(f"   → LLM 决策：{next_agent}")
    print(f"   → 理由：{reasoning[:50]}...")
    
    return {"next_agent": next_agent}

# ========== 5. 【核心】并行调度函数 ==========

def route_from_supervisor(state: TeamState):
    """Supervisor 的条件边路由"""
    next_agent = state.get("next_agent", "FINISH")
    
    if next_agent == "parallel_research":
        # ✅ 返回 Send 列表，LangGraph 并行执行！
        print("🚀 [Supervisor] 启动并行调研！")
        return [
            Send("researcher", {"topic": state["topic"], "angle": "技术角度", "partial_notes": []}),
            Send("researcher", {"topic": state["topic"], "angle": "教育角度", "partial_notes": []}),
            Send("researcher", {"topic": state["topic"], "angle": "伦理角度", "partial_notes": []})
        ]
    else:
        # 其他情况返回字符串，走普通条件边
        return next_agent

# ========== 6. 构建图 ==========

workflow = StateGraph(TeamState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("aggregator", aggregator_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.set_entry_point("supervisor")

# ✅ 条件边：parallel_research 返回 Send 列表，其他返回字符串
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "aggregator": "aggregator",  # 汇总
        "writer": "writer", 
        "editor": "editor", 
        "FINISH": END,
        "supervisor": "supervisor"
    }
)

# ✅ 关键：researcher 执行完后回到 supervisor
# 这样 supervisor 可以检查 partial_notes 数量
workflow.add_edge("researcher", "supervisor")

# aggregator 汇总后回到 supervisor
workflow.add_edge("aggregator", "supervisor")

# writer 和 editor 执行完后回到 supervisor
workflow.add_edge("writer", "supervisor")
workflow.add_edge("editor", "supervisor")

# ========== 7. 编译 & 运行 ==========

with SqliteSaver.from_conn_string("team_parallel_v2.db") as memory:
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "team_parallel_v2_001"}}
    inputs = {
        "topic": "人工智能对现代教育的影响", 
        "messages": [("user", "帮我写一篇关于AI教育的文章")],
        "previous_feedback": "",
        "partial_notes": []
    }
    
    print("\n" + "="*50)
    print("🚀 启动并行调研团队（Send 并行版 v2）")
    print("="*50)
    app.invoke(inputs, config=config)
    
    # ========== 8. 人工审批循环 ==========
    
    while True:
        state = app.get_state(config).values
        print("\n" + "-"*50)
        print("⏸️ [系统] 团队已完成初步工作，等待人类审批！")
        print(f"📄 文章初稿: {str(state.get('draft', '暂无'))[:150]}...")
        print(f"💬 编辑意见: {str(state.get('edit_feedback', '暂无'))[:150]}...")
        
        choice = input("\n👤 人类主管，请决定 (y=批准发布 / n=打回重写): ").lower()
        
        if choice == 'y':
            print("✅ 批准发布！流程结束。")
            break
        elif choice == 'n':
            print("❌ 打回重写！清空状态，让团队重来...")
            
            current_feedback = state.get("edit_feedback", "")
            
            app.update_state(config, {
                "research_notes": None,
                "draft": None,
                "edit_feedback": None,
                "previous_feedback": current_feedback,
                "next_agent": "supervisor",
                "partial_notes": []
            })
            app.invoke(None, config=config)
        else:
            print("无效输入，请重新选择。")
    
    print("\n" + "="*50)
    print("🏁 流程彻底结束")
    print("="*50)