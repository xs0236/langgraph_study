from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class State(TypedDict):
    draft: str
    approved: bool
    final: str

# ========== 节点 ==========
def write_draft(state):
    print("  ✍️ 生成草稿...")
    return {"draft": "【草稿】人工智能正在改变教育行业..."}

def human_review(state):
    print("  ⏸️  等待人工审核...")
    # 这里会暂停！payload 里的数据可以被外部读取
    decision = interrupt({
        "draft": state["draft"],
        "hint": "输入 'y' 通过，其他拒绝"
    })
    print(f"  ✅ 收到用户决策: {decision}")
    return {"approved": decision == "y"}

def publish(state):
    if state["approved"]:
        print("  🚀 文章已发布！")
        return {"final": "已发布"}
    else:
        print("  ❌ 文章被拒绝")
        return {"final": "已拒绝"}

# ========== 构图 ==========
builder = StateGraph(State)
builder.add_node("write", write_draft)
builder.add_node("review", human_review)
builder.add_node("publish", publish)

builder.set_entry_point("write")
builder.add_edge("write", "review")
builder.add_conditional_edges(
    "review",
    lambda s: "publish" if s["approved"] else END,
    {"publish": "publish", END: END}
)
builder.add_edge("publish", END)

# 必须加 checkpointer，否则无法暂停/恢复
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# ========== 第一次运行：启动图（会在 review 节点暂停）==========
config = {"configurable": {"thread_id": "demo_1"}}
print("=== 第一次 stream ===")
for event in app.stream({"draft": ""}, config):
    print("事件:", event)

# 此时图已暂停在 human_review 节点
# 你可以在这里查看中断携带的数据
print("\n=== 图已暂停，等待用户输入 ===")

# ========== 第二次运行：恢复图 ==========
print("\n=== 第二次 stream（恢复）===")
# Command(resume="y") 把 "y" 传给 interrupt() 的返回值
for event in app.stream(Command(resume="y"), config):
    print("事件:", event)

print("\n=== 最终状态 ===")
print(app.get_state(config).values)