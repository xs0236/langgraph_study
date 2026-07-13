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

#引入记忆
from langgraph.checkpoint.memory import MemorySaver
# 初始化一个内存存储器
memory = MemorySaver()

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

# 定义一个新的工具-计算
@tool
def calculator(expression: str) -> float:
    """计算数学表达式的结果，例如 '2 + 2' 或 '15 * 4'"""
    try:
        # 注意：在生产环境中 eval 是不安全的，但在本地学习 demo 中可以使用
        result = eval(expression)
        return result
    except Exception as e:
        return f"计算错误: {e}"

# 更新工具列表
tools = [get_weather, calculator] 

# 重要：更新模型的绑定
model = ChatOpenAI(model="glm-4-flash", temperature=0).bind_tools(tools) 
# 注意：如果你用的是 GLM，确保模型名称是你刚才调试通过的那个



# --- 定义节点 (Nodes) ---
# 初始化大模型，并绑定工具
model = ChatOpenAI(model="glm-4-flash",
      openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
      temperature=0,
  ).bind_tools(tools)

# def chatbot(state: State):
#     
#     response = model.invoke(state["messages"])
#     return {"messages": [response]}
def chatbot(state: State):
    response = model.invoke(state["messages"])
    """
     Agent 节点：负责思考
     它接收当前状态，把历史消息发给 LLM，LLM 决定是回答还是调工具
     """

    # 打印 LLM 的原始决策
    print(f"[DEBUG] LLM 回复内容: {response.content}")
    print(f"[DEBUG] LLM 想要调用的工具: {response.tool_calls}")

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
# 添加记忆
app = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "user_001"}}

# 第一次提问
inputs = {"messages": [("user", "你好，帮我查一下北京的天气。")]}
for output in app.stream(inputs, config=config):
    # ... (保持原有的打印逻辑)
    pass

# 第二次提问 (测试记忆)
inputs2 = {"messages": [("user", "那如果温度打八折是多少？")]} # 注意：这里不需要传历史消息，Agent会自动从 memory 读取
print("\n--- 第二轮对话 ---")
for output in app.stream(inputs2, config=config):
    # ...
    pass


# print("--- 开始运行 Agent ---")
# # stream 方法可以让我们看到每一步的执行过程
# for output in app.stream(inputs, config=config):
#     for key, value in output.items():
#         print(f"节点 [{key}] 输出:")
#         # 打印最后一条消息的内容
#         print(value["messages"][-1].content)
#         print("-" * 20)

# print("\n--- 运行结束 ---")

print("--- 开始运行带人工干预的 Agent ---")
# 1. 启动对话

inputs = {"messages": [("user", "帮我查一下北京天气，然后计算 100 乘以 8 是多少。")]}

# 使用 stream 运行，它会走到 "tools" 节点前自动暂停

for output in app.stream(inputs, config=config, stream_mode="values"):
    print(f"当前状态: {output['messages'][-1].content}")
    print("-" * 20)

# 2. 审查与干预

print("\n--- 程序已暂停，等待人工干预 ---")
print("当前待执行的工具调用：")

# 获取最新的消息，也就是 LLM 想要调用工具的请求

pending_tool_calls = output['messages'][-1].tool_calls
print(pending_tool_calls)

# 这里可以加入你的审查逻辑，比如打印出来让用户确认

user_approval = input("\n是否批准执行以上操作？(y/n): ")

if user_approval.lower() == 'y': # 3. 批准后继续
    print("\n--- 用户已批准，继续执行 ---") # 再次调用 stream，但不传入 inputs，它会从暂停的地方继续
for output in app.stream(None, config=config, stream_mode="values"):
    print(f"最终结果: {output['messages'][-1].content}")
    print("-" * 20)
else:
    print("操作已取消。")

print("\n--- 运行结束 ---")