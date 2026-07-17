import os
os.environ["OPENAI_API_KEY"] = "sk-fake-placeholder-for-zhipu"

import gradio as gr
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZHIPU_API_KEY"),
    temperature=0.7
)

# ========== 关键：with 块要包含所有使用 app 的代码 ==========
with SqliteSaver.from_conn_string("team_gradio.db") as memory:
    
    # ----- 子图 1：写作 -----
    class WritingState(TypedDict):
        topic: str; research_notes: str; article: str
        edit_feedback: str; previous_feedback: str; next_agent: str

    def w_r(s): 
        print("      🕵️ [写作-调研] ...")
        p = f"针对主题 '{s['topic']}' 搜集关键信息，输出简短调研笔记。"
        return {"research_notes": model.invoke([HumanMessage(content=p)]).content}
    def w_w(s): 
        print("      ✍️ [写作-撰稿] ...")
        return {"article": model.invoke([HumanMessage(content=f"根据调研笔记撰写文章：\n{s['research_notes']}")]).content}
    def w_e(s): 
        print("      📝 [写作-编辑] ...")
        return {"edit_feedback": model.invoke([HumanMessage(content=f"审核文章：\n{s['article'][:500]}")]).content}
    # def w_s(s): 
    #     return {"next_agent": "FINISH" if s.get("edit_feedback") else ("writer" if s.get("research_notes") else "researcher")}
    
    # ✅ 修复：加入 article 检查，防止无限循环
    def w_s(s): 
        if not s.get("research_notes"):
            return {"next_agent": "researcher"}
        elif not s.get("article"):
            return {"next_agent": "writer"}
        elif not s.get("edit_feedback"):
            return {"next_agent": "editor"}
        else:
            return {"next_agent": "FINISH"}


    wg = StateGraph(WritingState)
    wg.add_node("w_s", w_s); wg.add_node("w_r", w_r); wg.add_node("w_w", w_w); wg.add_node("w_e", w_e)
    wg.set_entry_point("w_s")
    wg.add_conditional_edges("w_s", lambda s: s["next_agent"], {"researcher":"w_r","writer":"w_w","editor":"w_e","FINISH":END})
    for a,b in [("w_r","w_s"),("w_w","w_s"),("w_e","w_s")]: wg.add_edge(a,b)
    writing_sg = wg.compile()

    # ----- 子图 2：翻译 -----
    class TState(TypedDict):
        article: str; english_version: str; japanese_version: str; next_agent: str
    def t_e(s): 
        print("      🇺🇸 [翻译-英文] ...")
        return {"english_version": model.invoke([HumanMessage(content=f"翻译成地道英文：\n\n{s['article'][:800]}")]).content}
    def t_j(s): 
        print("      🇯🇵 [翻译-日文] ...")
        return {"japanese_version": model.invoke([HumanMessage(content=f"翻译成地道日文：\n\n{s['article'][:800]}")]).content}
    def t_s(s):
        if not s.get("english_version"): return {"next_agent": "english"}
        if not s.get("japanese_version"): return {"next_agent": "japanese"}
        return {"next_agent": "FINISH"}

    tg = StateGraph(TState)
    tg.add_node("t_s", t_s); tg.add_node("t_e", t_e); tg.add_node("t_j", t_j)
    tg.set_entry_point("t_s")
    tg.add_conditional_edges("t_s", lambda s: s["next_agent"], {"english":"t_e","japanese":"t_j","FINISH":END})
    for a,b in [("t_e","t_s"),("t_j","t_s")]: tg.add_edge(a,b)
    trans_sg = tg.compile()

    # ----- 子图 3：配图 -----
    class IState(TypedDict):
        topic: str; article: str; image_prompt: str; image_url: str; next_agent: str
    def i_gen(s):
        print("      🎨 [配图] 生成提示词...")
        p = model.invoke([HumanMessage(content=f"根据主题生成英文AI绘画提示词：{s['topic']}\n摘要：{s['article'][:300]}")]).content
        url = f"https://picsum.photos/1024/1024?random={hash(s['topic'])%10000}"
        try:
            with open(f"prompt_{s['topic'][:10]}.txt","w",encoding="utf-8") as f: f.write(p)
        except: pass
        return {"image_prompt": p, "image_url": url}

    ig = StateGraph(IState)
    ig.add_node("i_gen", i_gen); ig.set_entry_point("i_gen"); ig.add_edge("i_gen", END)
    image_sg = ig.compile()

    # ----- 子图 4：发布 -----
    class PState(TypedDict):
        topic: str; article: str; image_url: str
        wechat_url: str; zhihu_url: str; weibo_text: str; next_agent: str
    def p_wc(s):
        print("      📱 [微信] ...")
        return {"wechat_url": f"https://mp.weixin.qq.com/simulated/{hash(s['topic'])%100000}"}
    def p_zh(s):
        print("      💡 [知乎] ...")
        return {"zhihu_url": f"https://zhuanlan.zhihu.com/p/simulated_{hash(s['topic'])%100000}"}
    def p_wb(s):
        print("      🔥 [微博] ...")
        return {"weibo_text": model.invoke([HumanMessage(content=f"生成微博文案（140字内）：\n{s['article'][:300]}")]).content}
    def p_s(s):
        if not s.get("wechat_url"): return {"next_agent": "wechat"}
        if not s.get("zhihu_url"): return {"next_agent": "zhihu"}
        if not s.get("weibo_text"): return {"next_agent": "weibo"}
        return {"next_agent": "FINISH"}

    pg = StateGraph(PState)
    pg.add_node("p_s", p_s); pg.add_node("p_wc", p_wc); pg.add_node("p_zh", p_zh); pg.add_node("p_wb", p_wb)
    pg.set_entry_point("p_s")
    pg.add_conditional_edges("p_s", lambda s: s["next_agent"], {"wechat":"p_wc","zhihu":"p_zh","weibo":"p_wb","FINISH":END})
    for a,b in [("p_wc","p_s"),("p_zh","p_s"),("p_wb","p_s")]: pg.add_edge(a,b)
    pub_sg = pg.compile()

    # ----- 主图 -----
    class MState(TypedDict):
        messages: Annotated[list, add_messages]
        topic: str; article: str; edit_feedback: str; previous_feedback: str
        english_version: str; japanese_version: str
        image_prompt: str; image_url: str
        wechat_url: str; zhihu_url: str; weibo_text: str
        next_agent: str

    def m_s(s):
        print(f"  👔 [主调度] 当前状态: article={'✅' if s.get('article') else '❌'}, "
          f"english={'✅' if s.get('english_version') else '❌'}, "
          f"image={'✅' if s.get('image_url') else '❌'}, "
          f"wechat={'✅' if s.get('wechat_url') else '❌'}")
        
        if not s.get("article"): return {"next_agent": "writing"}
        if not s.get("english_version"): return {"next_agent": "trans"}
        if not s.get("image_url"): return {"next_agent": "image"}
        if not s.get("wechat_url"): return {"next_agent": "publish"}
        return {"next_agent": "FINISH"}

    mg = StateGraph(MState)
    mg.add_node("m_s", m_s)
    mg.add_node("writing", writing_sg)
    mg.add_node("trans", trans_sg)
    mg.add_node("image", image_sg)
    mg.add_node("publish", pub_sg)
    mg.set_entry_point("m_s")
    mg.add_conditional_edges("m_s", lambda s: s["next_agent"], {
        "writing":"writing", "trans":"trans", "image":"image", "publish":"publish", "FINISH":END, "m_s":"m_s"
    })
    for n in ["writing","trans","image","publish"]: mg.add_edge(n, "m_s")
    app = mg.compile(checkpointer=memory)

    # ========== Gradio 界面（必须在 with 块内部！）==========

    def generate_all(topic):
        """生成函数，用 yield 实时更新进度"""
        if not topic.strip():
            # ✅ 修复：补齐到 10 个值！
            yield "❌ 请输入主题", "", "", "", "", "", "", "", "", ""
            return
        
        config = {"configurable": {"thread_id": f"gradio_{hash(topic) & 0xFFFFFFFF}"}}
        inputs = {
            "topic": topic,
            "messages": [("user", f"帮我写一篇关于{topic}的文章")],
            "previous_feedback": ""
        }
        
        logs = [f"🚀 开始生成主题：{topic}", "-" * 40]
        
        # ✅ 修复：补齐到 10 个值！
        yield "\n".join(logs), "生成中...", "", "", "", "", "", "", "", ""
        
        # Stream 执行
        for chunk in app.stream(inputs, config=config):
            for node_name in chunk.keys():
                if node_name == "__end__":
                    continue
                name_map = {
                    "m_s": "👔 主调度", "writing": "📝 写作子图", 
                    "trans": "🌐 翻译子图", "image": "🎨 配图子图", 
                    "publish": "📱 发布子图"
                }
                logs.append(f"✅ {name_map.get(node_name, node_name)} 执行完成")
                # ✅ 修复：补齐到 10 个值！
                yield "\n".join(logs), "生成中...", "", "", "", "", "", "", "", ""
        
        # 获取最终结果
        state = app.get_state(config).values
        
        logs.append("-" * 40)
        logs.append("🎉 全部完成！")
        
        # 构建图片 HTML
        img_html = ""
        if state.get("image_url"):
            img_html = f'<img src="{state["image_url"]}" width="512" style="border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">'
        
        # ✅ 最终 yield：10 个值
        yield (
            "\n".join(logs),
            state.get("article", ""),
            state.get("edit_feedback", ""),
            state.get("english_version", ""),
            state.get("japanese_version", ""),
            state.get("image_prompt", ""),
            img_html,
            state.get("wechat_url", ""),
            state.get("zhihu_url", ""),
            state.get("weibo_text", "")
        )

    # 构建界面
    with gr.Blocks(title="🏭 AI 内容工厂", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏭 AI 内容工厂
        输入一个主题，自动完成：**写作 → 翻译 → 配图 → 多平台发布**
        """)

        with gr.Row():
            with gr.Column(scale=3):
                topic_input = gr.Textbox(
                    label="📝 文章主题",
                    placeholder="例如：人工智能对现代教育的影响",
                    value="人工智能对现代教育的影响"
                )
            with gr.Column(scale=1):
                generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")

        # 进度日志
        progress_box = gr.Textbox(
            label="📊 生成进度",
            lines=8,
            interactive=False,
            value="等待开始..."
        )

        # 成果展示 Tabs
        with gr.Tabs():
            with gr.TabItem("📝 原文 & 编辑意见"):
                with gr.Row():
                    article_box = gr.Textbox(label="文章正文", lines=15, interactive=False)
                    edit_box = gr.Textbox(label="编辑修改意见", lines=8, interactive=False)

            with gr.TabItem("🇺🇸 英文版"):
                en_box = gr.Textbox(label="English Version", lines=12, interactive=False)

            with gr.TabItem("🇯🇵 日文版"):
                jp_box = gr.Textbox(label="日本語版", lines=12, interactive=False)

            with gr.TabItem("🎨 配图"):
                with gr.Row():
                    prompt_box = gr.Textbox(label="AI 绘画提示词", lines=3, interactive=False)
                with gr.Row():
                    image_html = gr.HTML(label="图片预览")

            with gr.TabItem("📱 发布文案"):
                with gr.Row():
                    wechat_box = gr.Textbox(label="微信公众号链接", lines=2, interactive=False)
                    zhihu_box = gr.Textbox(label="知乎链接", lines=2, interactive=False)
                weibo_box = gr.Textbox(label="微博文案", lines=3, interactive=False)

        # 绑定事件
        outputs = [
            progress_box,        # 1
            article_box,         # 2
            edit_box,            # 3
            en_box,              # 4
            jp_box,              # 5
            prompt_box,          # 6
            image_html,          # 7
            wechat_box,          # 8
            zhihu_box,           # 9
            weibo_box            # 10
        ]

        generate_btn.click(fn=generate_all, inputs=topic_input, outputs=outputs)
        topic_input.submit(fn=generate_all, inputs=topic_input, outputs=outputs)

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)