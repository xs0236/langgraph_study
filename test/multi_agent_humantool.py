
import os
import json
import re
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
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

# ========== 3. 定义专业 Agent 节点 ==========

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

# ========== 4. 【核心】智能主管节点（Prompt 引导 + 手动解析）==========

SUPERVISOR_SYSTEM_PROMPT = """你是一个智能调度主管，负责协调一个写作团队的工作。

团队成员：
- researcher（调研员）：负责搜集资料，生成调研笔记
- writer（撰稿人）：根据调研笔记撰写文章初稿  
- editor（编辑）：审核文章初稿，给出修改意见

当前任务状态：
- 调研笔记：{research_notes_status}
- 文章初稿：{draft_status}
- 编辑意见：{edit_feedback_status}

调度规则：
1. 如果没有调研笔记 → 选择 researcher
2. 如果有调研笔记但没有初稿 → 选择 writer
3. 如果有初稿但没有编辑意见 → 选择 editor
4. 如果三者都已完成 → 选择 FINISH

【重要】请严格按以下 JSON 格式输出你的决策，不要添加任何其他内容：

{{
  "next_agent": "researcher 或 writer 或 editor 或 FINISH",
  "reasoning": "你的决策理由"
}}"""

def parse_llm_json(content: str) -> dict:
    """解析 LLM 返回的 JSON，处理 markdown 代码块包裹的情况"""
    # 去掉 markdown 代码块标记
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
    
    # 准备状态描述
    research_notes = state.get("research_notes", "")
    draft = state.get("draft", "")
    edit_feedback = state.get("edit_feedback", "")
    
    research_notes_status = "✅ 已完成" if research_notes else "❌ 未完成"
    draft_status = "✅ 已完成" if draft else "❌ 未完成"
    edit_feedback_status = "✅ 已完成" if edit_feedback else "❌ 未完成"
    
    # 构建系统提示
    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        research_notes_status=research_notes_status,
        draft_status=draft_status,
        edit_feedback_status=edit_feedback_status
    )
    
    # 调用 LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请根据当前任务状态，做出调度决策。只输出 JSON，不要加 markdown 代码块。")
    ]
    
    response = model.invoke(messages)
    content = response.content
    
    # 手动解析 JSON（处理 markdown 包裹）
    try:
        decision = parse_llm_json(content)
        next_agent = decision.get("next_agent", "FINISH")
        reasoning = decision.get("reasoning", "无")
    except (json.JSONDecodeError, Exception) as e:
        print(f"   ⚠️ JSON 解析失败: {e}")
        print(f"   原始内容: {content[:100]}...")
        # 回退到硬编码逻辑
        if not research_notes:
            next_agent = "researcher"
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

# ========== 5. 构建图 ==========

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

# ========== 6. 编译 & 运行 ==========

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "team_manual_json_001"}}
inputs = {
    "topic": "人工智能对现代教育的影响", 
    "messages": [("user", "帮我写一篇关于AI教育的文章")],
    "previous_feedback": ""
}

print("\n" + "="*50)
print("🚀 启动智能主管调度团队（手动 JSON 解析版）")
print("="*50)
app.invoke(inputs, config=config)

# ========== 7. 人工审批循环 ==========

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
            "next_agent": "supervisor"
        })
        app.invoke(None, config=config)
    else:
        print("无效输入，请重新选择。")

print("\n" + "="*50)
print("🏁 流程彻底结束")
print("="*50)