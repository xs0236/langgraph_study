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

# ========== 模型配置（只用智谱，不需要 OpenAI）==========
model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7
)

# ========== 【主图状态】==========
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    article: str
    edit_feedback: str
    previous_feedback: str
    english_version: str
    japanese_version: str
    image_prompt: str
    image_url: str           # 模拟：图片链接
    wechat_url: str          # 模拟：微信发布链接
    zhihu_url: str           # 模拟：知乎发布链接
    weibo_text: str
    next_agent: str

# ============================================================
# 【子图 1】写作子图
# ============================================================
class WritingState(TypedDict):
    topic: str
    research_notes: str
    article: str
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
    return {"article": model.invoke([HumanMessage(content=prompt)]).content}

def w_editor(state: WritingState):
    print("      📝 [写作-编辑] 审核文章...")
    prompt = f"审核以下文章并给出修改意见：\n{state['article'][:500]}"
    return {"edit_feedback": model.invoke([HumanMessage(content=prompt)]).content}

def w_supervisor(state: WritingState):
    if not state.get("research_notes"):
        return {"next_agent": "researcher"}
    elif not state.get("article"):
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
# 【子图 2】翻译子图
# ============================================================
class TranslationState(TypedDict):
    article: str
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
# 【子图 3】配图子图 —— 模拟版
# ============================================================

class ImageState(TypedDict):
    topic: str
    article: str
    image_prompt: str
    image_url: str
    next_agent: str

def i_generator(state: ImageState):
    """模拟配图生成（保留真实 API 接口，方便替换）"""
    print("      🎨 [配图] 生成 AI 绘画提示词...")
    
    # 让 LLM 生成提示词
    prompt = f"""根据以下文章主题，生成一张配图用的英文 AI 绘画提示词（Midjourney/DALL-E 风格）。
要求：画面感强、色彩丰富、适合文章配图、无文字。

文章主题：{state['topic']}
文章摘要：{state['article'][:300]}

只输出提示词，不要解释。
"""
    image_prompt = model.invoke([HumanMessage(content=prompt)]).content
    print(f"      🎨 [配图] 提示词：{image_prompt[:60]}...")
    
    # ==================== 模拟模式 ====================
    # 真实 API 调用（需要 OPENAI_API_KEY）：
    # from openai import OpenAI
    # client = OpenAI(api_key="sk-xxx")
    # response = client.images.generate(
    #     model="dall-e-3",
    #     prompt=image_prompt[:1000],
    #     size="1024x1024",
    #     n=1,
    # )
    # image_url = response.data[0].url
    
    # 模拟：返回一个占位图 URL
    image_url = f"https://picsum.photos/1024/1024?random={hash(state['topic']) % 10000}"
    print(f"      🖼️ [配图] 模拟图片生成成功！")
    print(f"      💾 [配图] 占位图 URL: {image_url}")
    
    # 保存提示词到本地（方便以后用真实 API 生成）
    with open(f"image_prompt_{state['topic'][:10]}.txt", "w", encoding="utf-8") as f:
        f.write(image_prompt)
    print(f"      💾 [配图] 提示词已保存到: image_prompt_{state['topic'][:10]}.txt")
    
    return {
        "image_prompt": image_prompt,
        "image_url": image_url
    }

def build_image_subgraph():
    g = StateGraph(ImageState)
    g.add_node("i_generator", i_generator)
    g.set_entry_point("i_generator")
    g.add_edge("i_generator", END)
    return g.compile()

# ============================================================
# 【子图 4】发布子图 —— 模拟版
# ============================================================

class PublishState(TypedDict):
    topic: str
    article: str
    image_url: str
    wechat_url: str
    zhihu_url: str
    weibo_text: str
    next_agent: str

def p_wechat(state: PublishState):
    """模拟微信公众号发布（保留真实 API 接口）"""
    print("      📱 [发布-公众号] 生成文案...")
    
    # 生成文案
    prompt = f"""为以下文章生成微信公众号标题和摘要：
标题：{state['topic']}
内容：{state['article'][:400]}
"""
    wechat_content = model.invoke([HumanMessage(content=prompt)]).content
    
    # ==================== 模拟模式 ====================
    # 真实 API 调用（需要 WECHAT_APPID + WECHAT_SECRET）：
    # access_token = get_wechat_access_token()
    # draft_url = f"https://api.weixin.qq.com/cgi-bin/draft/add?access_token={access_token}"
    # resp = requests.post(draft_url, json={...})
    # wechat_url = f"https://mp.weixin.qq.com/s?mid={resp['media_id']}"
    
    # 模拟：返回模拟链接
    wechat_url = f"https://mp.weixin.qq.com/simulated/{hash(state['topic']) % 100000}"
    print(f"      ✅ [微信] 模拟发布成功！")
    print(f"      🔗 [微信] 模拟链接: {wechat_url}")
    
    # 保存发布内容到本地
    with open(f"wechat_draft_{state['topic'][:10]}.md", "w", encoding="utf-8") as f:
        f.write(f"# {state['topic']}\n\n")
        f.write(f"配图: {state['image_url']}\n\n")
        f.write(f"{state['article']}\n")
    print(f"      💾 [微信] 草稿已保存到: wechat_draft_{state['topic'][:10]}.md")
    
    return {"wechat_url": wechat_url}

def p_zhihu(state: PublishState):
    """模拟知乎发布（保留真实 API 接口）"""
    print("      💡 [发布-知乎] 生成文案...")
    
    # 生成知乎风格标题
    prompt = f"""为以下文章生成知乎风格标题：
标题：{state['topic']}
内容：{state['article'][:400]}

只输出标题，不要其他内容。
"""
    zhihu_title = model.invoke([HumanMessage(content=prompt)]).content.strip()
    
    # ==================== 模拟模式 ====================
    # 真实 API 调用（需要 ZHIHU_ACCESS_TOKEN）：
    # url = "https://www.zhihu.com/api/v4/articles"
    # headers = {"Authorization": f"Bearer {ZHIHU_ACCESS_TOKEN}"}
    # resp = requests.post(url, headers=headers, json={"title": zhihu_title, ...})
    # zhihu_url = f"https://zhuanlan.zhihu.com/p/{resp['id']}"
    
    # 模拟：返回模拟链接
    zhihu_url = f"https://zhuanlan.zhihu.com/p/simulated_{hash(state['topic']) % 100000}"
    print(f"      ✅ [知乎] 模拟发布成功！")
    print(f"      🔗 [知乎] 模拟链接: {zhihu_url}")
    
    # 保存发布内容到本地
    with open(f"zhihu_draft_{state['topic'][:10]}.md", "w", encoding="utf-8") as f:
        f.write(f"# {zhihu_title}\n\n")
        f.write(f"配图: {state['image_url']}\n\n")
        f.write(f"{state['article']}\n")
    print(f"      💾 [知乎] 草稿已保存到: zhihu_draft_{state['topic'][:10]}.md")
    
    return {"zhihu_url": zhihu_url}

def p_weibo(state: PublishState):
    """生成微博文案"""
    print("      🔥 [发布-微博] 生成文案...")
    prompt = f"""为以下文章生成微博文案（140字以内）：
标题：{state['topic']}
内容：{state['article'][:300]}
"""
    weibo_text = model.invoke([HumanMessage(content=prompt)]).content
    return {"weibo_text": weibo_text}

def p_supervisor(state: PublishState):
    if not state.get("wechat_url"):
        return {"next_agent": "wechat"}
    elif not state.get("zhihu_url"):
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
# 【主图】总调度
# ============================================================

def main_supervisor(state: MainState):
    print("👔 [Main-Supervisor] 总调度...")
    
    if not state.get("article"):
        print("   → 阶段 1：写作")
        return {"next_agent": "writing_team"}
    elif not state.get("english_version"):
        print("   → 阶段 2：翻译")
        return {"next_agent": "translation_team"}
    elif not state.get("image_url"):
        print("   → 阶段 3：配图（模拟）")
        return {"next_agent": "image_team"}
    elif not state.get("wechat_url"):
        print("   → 阶段 4：发布（模拟）")
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

with SqliteSaver.from_conn_string("team_simulated.db") as memory:
    workflow = build_main_graph()
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "simulated_001"}}
    inputs = {
        "topic": "人工智能对现代教育的影响",
        "messages": [("user", "帮我写一篇AI教育文章并发布")],
        "previous_feedback": ""
    }
    
    print("\n" + "="*60)
    print("🏭 内容工厂 —— 模拟版（无需任何 API Key）")
    print("="*60)
    
    for chunk in app.stream(inputs, config=config):
        for node_name in chunk.keys():
            if node_name != "__end__":
                print(f"📡 [Stream] '{node_name}' 完成")
    
    print("="*60)
    print("✅ 执行完毕！")
    print("="*60)
    
    state = app.get_state(config).values
    
    print("\n" + "🎉"*20)
    print("📦 最终成果汇总")
    print("🎉"*20)
    
    print(f"\n📝 【原文】\n{state.get('article', '无')[:200]}...")
    print(f"\n🇺🇸 【英文版】\n{state.get('english_version', '无')[:150]}...")
    print(f"\n🇯🇵 【日文版】\n{state.get('japanese_version', '无')[:100]}...")
    print(f"\n🎨 【配图提示词】\n{state.get('image_prompt', '无')[:100]}...")
    print(f"\n🖼️ 【配图链接】\n{state.get('image_url', '无')}")
    print(f"\n📱 【微信草稿】\n{state.get('wechat_url', '无')}")
    print(f"\n💡 【知乎草稿】\n{state.get('zhihu_url', '无')}")
    print(f"\n🔥 【微博文案】\n{state.get('weibo_text', '无')[:140]}...")
    
    print("\n\n💾 本地文件：")
    print("   - image_prompt_人工智能对现.txt  （配图提示词）")
    print("   - wechat_draft_人工智能对现.md  （微信草稿）")
    print("   - zhihu_draft_人工智能对现.md   （知乎草稿）")
    
    print("\n" + "🎉"*20)