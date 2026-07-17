import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7
)

# ========== 【主图状态】==========
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    article: str           # 写作子图输出
    edit_feedback: str
    previous_feedback: str
    english_version: str
    japanese_version: str
    image_prompt: str
    wechat_summary: str
    zhihu_summary: str
    weibo_text: str
    next_agent: str

# ============================================================
# 【子图 1】写作子图（字段名和主图对齐）
# ============================================================

class WritingState(TypedDict):
    topic: str
    research_notes: str
    article: str           # ✅ 对齐主图：article
    edit_feedback: str
    previous_feedback: str
    next_agent: str

def w_researcher(state: WritingState):
    print("      🕵️ [写作-调研] 搜集资料...")
    prompt = f"针对主题 '{state['topic']}' 搜集关键信息，输出简短调研笔记。"
    if state.get("previous_feedback"):
        prompt += f"\n\n参考上轮反馈改进：{state['previous_feedback'][:200]}"
    return {"research_notes": model.invoke([HumanMessage(content=prompt)]).content}

def w_writer(state: WritingState):
    print("      ✍️ [写作-撰稿] 撰写文章...")
    prompt = f"根据调研笔记撰写文章：\n{state['research_notes']}"
    return {"article": model.invoke([HumanMessage(content=prompt)]).content}  # ✅ article

def w_editor(state: WritingState):
    print("      📝 [写作-编辑] 审核文章...")
    prompt = f"审核以下文章并给出修改意见：\n{state['article'][:500]}"  # ✅ article
    return {"edit_feedback": model.invoke([HumanMessage(content=prompt)]).content}

def w_supervisor(state: WritingState):
    if not state.get("research_notes"):
        return {"next_agent": "researcher"}
    elif not state.get("article"):   # ✅ article
        return {"next_agent": "writer"}
    elif not state.get("edit_feedback"):
        return {"next_agent": "editor"}
    else:
        return {"next_agent": "FINISH"}

def build_writing_subgraph():
    g = StateGraph(WritingState)
    g.add_node("w_supervisor", w_supervisor)
    g.add_node("w_researcher", w_researcher)
    g.add_node("w_writer", w_writer)
    g.add_node("w_editor", w_editor)
    g.set_entry_point("w_supervisor")
    g.add_conditional_edges("w_supervisor", lambda s: s["next_agent"], {
        "researcher": "w_researcher", "writer": "w_writer", 
        "editor": "w_editor", "FINISH": END
    })
    g.add_edge("w_researcher", "w_supervisor")
    g.add_edge("w_writer", "w_supervisor")
    g.add_edge("w_editor", "w_supervisor")
    return g.compile()

# ============================================================
# 【子图 2】翻译子图（也需要对齐）
# ============================================================

class TranslationState(TypedDict):
    article: str           # ✅ 读取主图的 article
    english_version: str
    japanese_version: str
    next_agent: str

def t_english(state: TranslationState):
    print("      🇺🇸 [翻译-英文] 翻译中...")
    prompt = f"将以下文章翻译成地道英文：\n\n{state['article'][:800]}"
    return {"english_version": model.invoke([HumanMessage(content=prompt)]).content}

def t_japanese(state: TranslationState):
    print("      🇯🇵 [翻译-日文] 翻译中...")
    prompt = f"将以下文章翻译成地道日文：\n\n{state['article'][:800]}"
    return {"japanese_version": model.invoke([HumanMessage(content=prompt)]).content}

def t_supervisor(state: TranslationState):
    if not state.get("english_version"):
        return {"next_agent": "english"}
    elif not state.get("japanese_version"):
        return {"next_agent": "japanese"}
    else:
        return {"next_agent": "FINISH"}

def build_translation_subgraph():
    g = StateGraph(TranslationState)
    g.add_node("t_supervisor", t_supervisor)
    g.add_node("t_english", t_english)
    g.add_node("t_japanese", t_japanese)
    g.set_entry_point("t_supervisor")
    g.add_conditional_edges("t_supervisor", lambda s: s["next_agent"], {
        "english": "t_english", "japanese": "t_japanese", "FINISH": END
    })
    g.add_edge("t_english", "t_supervisor")
    g.add_edge("t_japanese", "t_supervisor")
    return g.compile()

# ============================================================
# 【子图 3】配图子图
# ============================================================

class ImageState(TypedDict):
    topic: str
    article: str           # ✅ 读取主图的 article
    image_prompt: str
    next_agent: str

def i_generator(state: ImageState):
    print("      🎨 [配图] 生成 AI 绘画提示词...")
    prompt = f"""根据以下文章主题，生成一张配图用的 AI 绘画提示词（Midjourney 风格）。
要求：画面感强、色彩丰富、适合文章配图。

文章主题：{state['topic']}
文章摘要：{state['article'][:300]}

请直接输出提示词，不要解释。
"""
    return {"image_prompt": model.invoke([HumanMessage(content=prompt)]).content}

def build_image_subgraph():
    g = StateGraph(ImageState)
    g.add_node("i_generator", i_generator)
    g.set_entry_point("i_generator")
    g.add_edge("i_generator", END)
    return g.compile()

# ============================================================
# 【子图 4】发布子图
# ============================================================

class PublishState(TypedDict):
    topic: str
    article: str           # ✅ 读取主图的 article
    image_prompt: str
    wechat_summary: str
    zhihu_summary: str
    weibo_text: str
    next_agent: str

def p_wechat(state: PublishState):
    print("      📱 [发布-公众号] 生成文案...")
    prompt = f"""为以下文章生成微信公众号发布文案：
标题：{state['topic']}
内容：{state['article'][:400]}

请输出吸引眼球的标题、摘要和标签。
"""
    return {"wechat_summary": model.invoke([HumanMessage(content=prompt)]).content}

def p_zhihu(state: PublishState):
    print("      💡 [发布-知乎] 生成文案...")
    prompt = f"""为以下文章生成知乎发布文案：
标题：{state['topic']}
内容：{state['article'][:400]}

请输出知乎风格标题、摘要和话题标签。
"""
    return {"zhihu_summary": model.invoke([HumanMessage(content=prompt)]).content}

def p_weibo(state: PublishState):
    print("      🔥 [发布-微博] 生成文案...")
    prompt = f"""为以下文章生成微博发布文案：
标题：{state['topic']}
内容：{state['article'][:300]}

要求：140字以内，带话题标签，有传播力。
"""
    return {"weibo_text": model.invoke([HumanMessage(content=prompt)]).content}

def p_supervisor(state: PublishState):
    if not state.get("wechat_summary"):
        return {"next_agent": "wechat"}
    elif not state.get("zhihu_summary"):
        return {"next_agent": "zhihu"}
    elif not state.get("weibo_text"):
        return {"next_agent": "weibo"}
    else:
        return {"next_agent": "FINISH"}

def build_publish_subgraph():
    g = StateGraph(PublishState)
    g.add_node("p_supervisor", p_supervisor)
    g.add_node("p_wechat", p_wechat)
    g.add_node("p_zhihu", p_zhihu)
    g.add_node("p_weibo", p_weibo)
    g.set_entry_point("p_supervisor")
    g.add_conditional_edges("p_supervisor", lambda s: s["next_agent"], {
        "wechat": "p_wechat", "zhihu": "p_zhihu", "weibo": "p_weibo", "FINISH": END
    })
    g.add_edge("p_wechat", "p_supervisor")
    g.add_edge("p_zhihu", "p_supervisor")
    g.add_edge("p_weibo", "p_supervisor")
    return g.compile()

# ============================================================
# 【主图】总调度中心
# ============================================================

def main_supervisor(state: MainState):
    print("👔 [Main-Supervisor] 总调度...")
    
    if not state.get("article"):   # ✅ 检查 article
        print("   → 阶段 1：启动写作子图")
        return {"next_agent": "writing_team"}
    elif not state.get("english_version"):
        print("   → 阶段 2：启动翻译子图")
        return {"next_agent": "translation_team"}
    elif not state.get("image_prompt"):
        print("   → 阶段 3：启动配图子图")
        return {"next_agent": "image_team"}
    elif not state.get("wechat_summary"):
        print("   → 阶段 4：启动发布子图")
        return {"next_agent": "publish_team"}
    else:
        print("   → 全部完成！")
        return {"next_agent": "FINISH"}

def build_main_graph():
    workflow = StateGraph(MainState)
    
    workflow.add_node("main_supervisor", main_supervisor)
    workflow.add_node("writing_team", build_writing_subgraph())
    workflow.add_node("translation_team", build_translation_subgraph())
    workflow.add_node("image_team", build_image_subgraph())
    workflow.add_node("publish_team", build_publish_subgraph())
    
    workflow.set_entry_point("main_supervisor")
    
    workflow.add_conditional_edges("main_supervisor", lambda s: s["next_agent"], {
        "writing_team": "writing_team",
        "translation_team": "translation_team",
        "image_team": "image_team",
        "publish_team": "publish_team",
        "FINISH": END,
        "main_supervisor": "main_supervisor"
    })
    
    workflow.add_edge("writing_team", "main_supervisor")
    workflow.add_edge("translation_team", "main_supervisor")
    workflow.add_edge("image_team", "main_supervisor")
    workflow.add_edge("publish_team", "main_supervisor")
    
    return workflow

# ============================================================
# 运行
# ============================================================

with SqliteSaver.from_conn_string("team_factory.db") as memory:
    workflow = build_main_graph()
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "content_factory_001"}}
    inputs = {
        "topic": "人工智能对现代教育的影响",
        "messages": [("user", "帮我写一篇AI教育文章并发布")],
        "previous_feedback": ""
    }
    
    print("\n" + "="*60)
    print("🏭 启动内容工厂（多子图嵌套版）")
    print("="*60)
    
    for chunk in app.stream(inputs, config=config):
        for node_name in chunk.keys():
            if node_name != "__end__":
                print(f"📡 [Stream] 节点 '{node_name}' 执行完成")
    
    print("="*60)
    print("✅ 全部子图执行完毕！")
    print("="*60)
    
    state = app.get_state(config).values
    
    print("\n" + "🎉"*30)
    print("📦 最终成果汇总")
    print("🎉"*30)
    
    print(f"\n📝 【原文】\n{state.get('article', '无')[:200]}...")
    print(f"\n🇺🇸 【英文版】\n{state.get('english_version', '无')[:150]}...")
    print(f"\n🇯🇵 【日文版】\n{state.get('japanese_version', '无')[:100]}...")
    print(f"\n🎨 【配图提示词】\n{state.get('image_prompt', '无')[:150]}...")
    print(f"\n📱 【公众号文案】\n{state.get('wechat_summary', '无')[:150]}...")
    print(f"\n💡 【知乎文案】\n{state.get('zhihu_summary', '无')[:150]}...")
    print(f"\n🔥 【微博文案】\n{state.get('weibo_text', '无')[:140]}...")
    
    print("\n" + "🎉"*30)