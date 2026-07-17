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
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZHIPU_API_KEY"),
    temperature=0.7
)

current_config = None
waiting_for_human = False
interrupt_payload = None

with SqliteSaver.from_conn_string("team_gradio.db") as memory:

    class WritingState(TypedDict):
        topic: str
        research_notes: str
        article: str
        edit_feedback: str
        previous_feedback: str
        next_agent: str

    def w_r(s):
        print("      🕵️ [写作-调研] ...")
        p = "针对主题 '{}' 搜集关键信息，输出简短调研笔记。".format(s["topic"])
        return {"research_notes": model.invoke([HumanMessage(content=p)]).content}

    def w_w(s):
        print("      ✍️ [写作-撰稿] ...")
        feedback = s.get("previous_feedback", "")
        if feedback:
            prompt = "根据调研笔记撰写文章，并参考以下修改意见进行改进：\n{}\n\n调研笔记：\n{}".format(
                feedback, s["research_notes"]
            )
        else:
            prompt = "根据调研笔记撰写文章：\n{}".format(s["research_notes"])
        return {"article": model.invoke([HumanMessage(content=prompt)]).content}

    def w_e(s):
        print("      📝 [写作-编辑] ...")
        prompt = "审核文章：\n{}".format(s["article"][:500])
        return {"edit_feedback": model.invoke([HumanMessage(content=prompt)]).content}

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
    wg.add_node("w_s", w_s)
    wg.add_node("w_r", w_r)
    wg.add_node("w_w", w_w)
    wg.add_node("w_e", w_e)
    wg.set_entry_point("w_s")
    wg.add_conditional_edges("w_s", lambda s: s["next_agent"], {
        "researcher": "w_r",
        "writer": "w_w",
        "editor": "w_e",
        "FINISH": END
    })
    for a, b in [("w_r", "w_s"), ("w_w", "w_s"), ("w_e", "w_s")]:
        wg.add_edge(a, b)
    writing_sg = wg.compile()

    class TState(TypedDict):
        article: str
        english_version: str
        japanese_version: str
        next_agent: str

    def t_e(s):
        print("      🇺🇸 [翻译-英文] ...")
        prompt = "翻译成地道英文：\n\n{}".format(s["article"][:800])
        return {"english_version": model.invoke([HumanMessage(content=prompt)]).content}

    def t_j(s):
        print("      🇯🇵 [翻译-日文] ...")
        prompt = "翻译成地道日文：\n\n{}".format(s["article"][:800])
        return {"japanese_version": model.invoke([HumanMessage(content=prompt)]).content}

    def t_s(s):
        if not s.get("english_version"):
            return {"next_agent": "english"}
        if not s.get("japanese_version"):
            return {"next_agent": "japanese"}
        return {"next_agent": "FINISH"}

    tg = StateGraph(TState)
    tg.add_node("t_s", t_s)
    tg.add_node("t_e", t_e)
    tg.add_node("t_j", t_j)
    tg.set_entry_point("t_s")
    tg.add_conditional_edges("t_s", lambda s: s["next_agent"], {
        "english": "t_e",
        "japanese": "t_j",
        "FINISH": END
    })
    for a, b in [("t_e", "t_s"), ("t_j", "t_s")]:
        tg.add_edge(a, b)
    trans_sg = tg.compile()

    class IState(TypedDict):
        topic: str
        article: str
        image_prompt: str
        image_url: str
        next_agent: str

    def i_gen(s):
        print("      🎨 [配图] 生成提示词...")
        p = model.invoke([HumanMessage(
            content="根据主题生成英文AI绘画提示词：{}\n摘要：{}".format(s["topic"], s["article"][:300])
        )]).content
        url = "https://picsum.photos/1024/1024?random={}".format(hash(s["topic"]) % 10000)
        try:
            with open("prompt_{}.txt".format(s["topic"][:10]), "w", encoding="utf-8") as f:
                f.write(p)
        except:
            pass
        return {"image_prompt": p, "image_url": url}

    ig = StateGraph(IState)
    ig.add_node("i_gen", i_gen)
    ig.set_entry_point("i_gen")
    ig.add_edge("i_gen", END)
    image_sg = ig.compile()

    class PState(TypedDict):
        topic: str
        article: str
        image_url: str
        wechat_url: str
        zhihu_url: str
        weibo_text: str
        next_agent: str

    def p_wc(s):
        print("      📱 [微信] ...")
        return {"wechat_url": "https://mp.weixin.qq.com/simulated/{}".format(hash(s["topic"]) % 100000)}

    def p_zh(s):
        print("      💡 [知乎] ...")
        return {"zhihu_url": "https://zhuanlan.zhihu.com/p/simulated_{}".format(hash(s["topic"]) % 100000)}

    def p_wb(s):
        print("      🔥 [微博] ...")
        prompt = "生成微博文案（140字内）：\n{}".format(s["article"][:300])
        return {"weibo_text": model.invoke([HumanMessage(content=prompt)]).content}

    def p_s(s):
        if not s.get("wechat_url"):
            return {"next_agent": "wechat"}
        if not s.get("zhihu_url"):
            return {"next_agent": "zhihu"}
        if not s.get("weibo_text"):
            return {"next_agent": "weibo"}
        return {"next_agent": "FINISH"}

    pg = StateGraph(PState)
    pg.add_node("p_s", p_s)
    pg.add_node("p_wc", p_wc)
    pg.add_node("p_zh", p_zh)
    pg.add_node("p_wb", p_wb)
    pg.set_entry_point("p_s")
    pg.add_conditional_edges("p_s", lambda s: s["next_agent"], {
        "wechat": "p_wc",
        "zhihu": "p_zh",
        "weibo": "p_wb",
        "FINISH": END
    })
    for a, b in [("p_wc", "p_s"), ("p_zh", "p_s"), ("p_wb", "p_s")]:
        pg.add_edge(a, b)
    pub_sg = pg.compile()

    class MState(TypedDict):
        messages: Annotated[list, add_messages]
        topic: str
        article: str
        edit_feedback: str
        previous_feedback: str
        english_version: str
        japanese_version: str
        image_prompt: str
        image_url: str
        wechat_url: str
        zhihu_url: str
        weibo_text: str
        next_agent: str
        human_decision: str

    def human_review(state):
        print("  ⏸️ [人工审核] 等待用户输入...")
        decision = interrupt({
            "stage": "article_review",
            "article": state.get("article", ""),
            "edit_feedback": state.get("edit_feedback", ""),
            "message": "文章已生成，请输入操作：continue(继续) / rewrite(重写) / stop(终止)"
        })
        print("  ✅ [人工审核] 收到决策: {}".format(decision))
        return {"human_decision": decision}

    def m_s(s):
        if not s.get("article"):
            return {"next_agent": "writing"}

        if s.get("article") and not s.get("human_decision"):
            return {"next_agent": "human_review"}

        if s.get("human_decision") == "rewrite":
            return {
                "next_agent": "writing",
                "human_decision": "",
                "article": "",
                "edit_feedback": "",
                "previous_feedback": "用户要求重写，请改进文章质量。"
            }

        if s.get("human_decision") == "stop":
            return {"next_agent": "FINISH"}

        if not s.get("english_version"):
            return {"next_agent": "trans"}
        if not s.get("image_url"):
            return {"next_agent": "image"}
        if not s.get("wechat_url"):
            return {"next_agent": "publish"}

        return {"next_agent": "FINISH"}

    mg = StateGraph(MState)
    mg.add_node("m_s", m_s)
    mg.add_node("writing", writing_sg)
    mg.add_node("human_review", human_review)
    mg.add_node("trans", trans_sg)
    mg.add_node("image", image_sg)
    mg.add_node("publish", pub_sg)
    mg.set_entry_point("m_s")
    mg.add_conditional_edges("m_s", lambda s: s["next_agent"], {
        "writing": "writing",
        "human_review": "human_review",
        "trans": "trans",
        "image": "image",
        "publish": "publish",
        "FINISH": END,
        "m_s": "m_s"
    })
    for n in ["writing", "human_review", "trans", "image", "publish"]:
        mg.add_edge(n, "m_s")

    app = mg.compile(checkpointer=memory)

    def generate_all(topic):
        global current_config, waiting_for_human, interrupt_payload

        if not topic.strip():
            yield "❌ 请输入主题", "", "", "", "", "", "", "", "", ""
            return

        config = {"configurable": {"thread_id": "gradio_{}".format(hash(topic) & 0xFFFFFFFF)}}
        current_config = config
        waiting_for_human = False
        interrupt_payload = None

        inputs = {
            "topic": topic,
            "messages": [("user", "帮我写一篇关于{}的文章".format(topic))],
            "previous_feedback": ""
        }

        logs = ["🚀 开始生成主题：{}".format(topic), "-" * 40]
        yield "\n".join(logs), "生成中...", "", "", "", "", "", "", "", ""

        for chunk in app.stream(inputs, config):
            if "__interrupt__" in chunk:
                waiting_for_human = True
                interrupt_payload = chunk["__interrupt__"][0].value
                logs.append("-" * 40)
                logs.append("⏸️ 文章已生成，等待人工审核...")
                logs.append("📄 文章预览：{}...".format(interrupt_payload.get("article", "")[:200]))
                yield (
                    "\n".join(logs),
                    interrupt_payload.get("article", ""),
                    interrupt_payload.get("edit_feedback", ""),
                    "", "", "", "",
                    "", "",
                    "⏸️ 等待用户输入：continue / rewrite / stop"
                )
                return

            for node_name in chunk.keys():
                if node_name == "__end__":
                    continue
                name_map = {
                    "m_s": "👔 主调度",
                    "writing": "📝 写作子图",
                    "trans": "🌐 翻译子图",
                    "image": "🎨 配图子图",
                    "publish": "📱 发布子图",
                    "human_review": "⏸️ 人工审核"
                }
                logs.append("✅ {} 执行完成".format(name_map.get(node_name, node_name)))
                yield "\n".join(logs), "生成中...", "", "", "", "", "", "", "", ""

        state = app.get_state(config).values
        logs.append("-" * 40)
        logs.append("🎉 全部完成！")

        img_html = ""
        if state.get("image_url"):
            img_html = '<img src="{}" width="512" style="border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">'.format(
                state["image_url"]
            )

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

    def resume_all(user_input):
        global current_config, waiting_for_human, interrupt_payload

        if not waiting_for_human or current_config is None:
            yield "❌ 没有待恢复的任务，请先点击【开始生成】", "", "", "", "", "", "", "", "", ""
            return

        logs = ["🔄 恢复执行...", "-" * 40]
        yield "\n".join(logs), "恢复中...", "", "", "", "", "", "", "", ""

        for chunk in app.stream(Command(resume=user_input), current_config):
            if "__interrupt__" in chunk:
                waiting_for_human = True
                interrupt_payload = chunk["__interrupt__"][0].value
                logs.append("-" * 40)
                logs.append("⏸️ 再次等待人工审核...")
                logs.append("📄 文章预览：{}...".format(interrupt_payload.get("article", "")[:200]))
                yield (
                    "\n".join(logs),
                    interrupt_payload.get("article", ""),
                    interrupt_payload.get("edit_feedback", ""),
                    "", "", "", "",
                    "", "",
                    "⏸️ 等待用户输入：continue / rewrite / stop"
                )
                return

            for node_name in chunk.keys():
                if node_name == "__end__":
                    continue
                name_map = {
                    "m_s": "👔 主调度",
                    "writing": "📝 写作子图",
                    "trans": "🌐 翻译子图",
                    "image": "🎨 配图子图",
                    "publish": "📱 发布子图",
                    "human_review": "⏸️ 人工审核"
                }
                logs.append("✅ {} 执行完成".format(name_map.get(node_name, node_name)))
                yield "\n".join(logs), "恢复中...", "", "", "", "", "", "", "", ""

        state = app.get_state(current_config).values
        logs.append("-" * 40)

        if state.get("human_decision") == "stop":
            logs.append("🛑 用户已终止流程")
        else:
            logs.append("🎉 全部完成！")

        img_html = ""
        if state.get("image_url"):
            img_html = '<img src="{}" width="512" style="border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">'.format(
                state["image_url"]
            )

        waiting_for_human = False

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

    with gr.Blocks(title="🏭 AI 内容工厂 - Human-in-the-loop") as demo:
        gr.Markdown("""
        # 🏭 AI 内容工厂（支持人工审核）
        输入主题，自动完成：**写作 → 人工审核 → 翻译 → 配图 → 多平台发布**

        > 💡 文章写完后会暂停，等待你输入 `continue`（继续）、`rewrite`（重写）或 `stop`（终止）
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

        progress_box = gr.Textbox(
            label="📊 生成进度",
            lines=10,
            interactive=False,
            value="等待开始..."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⏸️ 人工审核区")
                user_decision = gr.Textbox(
                    label="审核决策",
                    placeholder="continue / rewrite / stop",
                    value="continue",
                    info="文章写完后在此输入决策，然后点击【确认并继续】"
                )
            with gr.Column():
                resume_btn = gr.Button("▶️ 确认并继续", variant="secondary", size="lg")

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

        outputs = [
            progress_box,
            article_box,
            edit_box,
            en_box,
            jp_box,
            prompt_box,
            image_html,
            wechat_box,
            zhihu_box,
            weibo_box
        ]

        generate_btn.click(fn=generate_all, inputs=topic_input, outputs=outputs)
        topic_input.submit(fn=generate_all, inputs=topic_input, outputs=outputs)
        resume_btn.click(fn=resume_all, inputs=user_decision, outputs=outputs)

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)