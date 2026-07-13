import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

# 1. 加载环境变量
load_dotenv() 

# 2. 导入 LangGraph 和 LangChain 核心组件
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# --- 定义状态 (State) ---
# 这是 Agent 的"记忆"，add_messages 表示新消息会追加到列表末尾
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- 定义工具 (Tools) ---
# 这是一个模拟的天气查询工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气情况"""
    print(f"[系统] 正在查询 {city} 的天气...")
    return f"{city} 今天天气晴朗，气温 25 度，适合出游！"

tools = [get_weather]

# --- 定义节点 (Nodes) ---
# 初始化大模型，并绑定工具
model = ChatOpenAI(model="glm-4-flash",
      openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
      temperature=0,
  ).bind_tools(tools)

def chatbot(state: State):
    """
    Agent 节点：负责思考
    它接收当前状态，把历史消息发给 LLM，LLM 决定是回答还是调工具
    """
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# --- 构建图 (Graph) ---
# 1. 创建图实例
workflow = StateGraph(State)

# 2. 添加节点
# "agent" 是大脑，"tools" 是手脚
workflow.add_node("agent", chatbot)
workflow.add_node("tools", ToolNode(tools))

# 3. 设置入口点 (Entry Point)
# 对话总是从 agent 开始
workflow.set_entry_point("agent")

# 4. 添加条件边 (Conditional Edges)
# tools_condition 是 LangGraph 内置的路由器：
# 如果 LLM 决定调工具 -> 去 "tools" 节点
# 如果 LLM 决定直接回答 -> 结束 (END)
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# 5. 添加工具执行后的回路
# 工具执行完后，必须回到 "agent" 节点，让 LLM 根据工具结果生成最终回复
workflow.add_edge("tools", "agent")

# --- 编译并运行 ---
app = workflow.compile()

# 模拟用户输入
inputs = {"messages": [("user", "你好，请帮我查一下青岛的天气")]}

print("--- 开始运行 Agent ---")
# stream 方法可以让我们看到每一步的执行过程
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"节点 [{key}] 输出:")
        # 打印最后一条消息的内容
        print(value["messages"][-1].content)
        print("-" * 20)

print("\n--- 运行结束 ---")