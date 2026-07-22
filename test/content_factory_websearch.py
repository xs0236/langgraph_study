import os

# ========== LangSmith 监控配置 ==========
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ai-content-factory")

os.environ["OPENAI_API_KEY"] = "sk-fake-placeholder-for-zhipu"

import gradio as gr
import json
import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ========== Web Search 导入（Tavily 方案）==========
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
    print("✅ Tavily 搜索已启用")
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠️ 未安装 tavily，调研节点将回退到模型知识。安装命令：pip install tavily")

def web_search_tavily(query, max_results=5):
    """使用 Tavily 搜索获取真实网络数据"""
    if not TAVILY_AVAILABLE:
        return None
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("  ⚠️ [搜索] 未设置 TAVILY_API_KEY")
        return None
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results, search_depth="basic")
        results = response.get("results", [])
        snippets = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            snippets.append("【{}】\n{}\n来源: {}".format(title, content, url))
        return "\n\n".join(snippets)
    except Exception as e:
        print("  ⚠️ [搜索] Tavily 搜索失败: {}".format(e))
        return None

model = ChatOpenAI(
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZHIPU_API_KEY"),
    temperature=0.7
)

# ========== 全局状态 ==========
app = None
current_config = None
active_config = None
waiting_for_human = False
interrupt_payload = None

# ========== Pydantic 结构化输出模型 ==========
class QualityCheck(BaseModel):
    score: int = Field(description="文章质量评分 1-10")
    issues: List[str] = Field(description="存在的问题列表")
    should_rewrite: bool = Field(description="是否需要重写")

class ImageDecision(BaseModel):
    need_image: bool = Field(description="这篇文章是否需要配图")
    reason: str = Field(description="判断理由")

class PublishPlan(BaseModel):
    platforms: List[str] = Field(description="选择要发布的平台，可选：wechat, zhihu, weibo")
    wechat_style: str = Field(description="微信公众号文案风格建议")
    zhihu_style: str = Field(description="知乎文案风格建议")
    weibo_style: str = Field(description="微博文案风格建议")

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
        topic = s["topic"]

        # ========== Tavily Web Search 核心改动 ==========
        search_results = web_search_tavily(topic, max_results=5)

        if search_results:
            print("      🔍 [写作-调研] 搜索到真实数据，基于网络资料撰写...")
            p = """针对主题 '{}' 搜集关键信息。

以下是从互联网搜索到的真实参考资料（按相关度排序）：
{}

请基于以上真实数据，输出简短但有信息量的调研笔记。注意：
1. 优先使用搜索数据中的事实、数据、政策、案例
2. 如果搜索数据不足，可补充你的知识
3. 标注关键数据来源""".format(topic, search_results)
        else:
            print("      🧠 [写作-调研] 无搜索结果，使用模型知识...")
            p = "针对主题 '{}' 搜集关键信息，输出简短调研笔记。".format(topic)

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
        translations: Annotated[list, operator.add]
        features: dict
        ai_quality_score: int
        ai_quality_issues: str
        ai_need_image: bool
        ai_image_reason: str
        ai_publish_platforms: str
        ai_publish_styles: str

    def ai_quality_check(state):
        print("  🤖 [AI自检] 评估文章质量...")
        article = state.get("article", "")
        prompt = """请对以下文章进行质量评估，输出结构化结果：

文章：
{}

请从以下维度评分（1-10分）：
1. 内容深度与专业性
2. 结构清晰度
3. 语言表达流畅度
4. 观点新颖性

如果总分低于7分或存在明显问题，should_rewrite 设为 true，并在 issues 中列出具体问题。""".format(article[:1000])

        try:
            structured_model = model.with_structured_output(QualityCheck)
            result = structured_model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            print("  ⚠️ [AI自检] 结构化输出失败，降级处理: {}".format(e))
            result = QualityCheck(score=8, issues=[], should_rewrite=False)

        print("  🤖 [AI自检] 评分: {}/10, 需重写: {}".format(result.score, result.should_rewrite))
        return {
            "ai_quality_score": result.score,
            "ai_quality_issues": "; ".join(result.issues) if result.issues else "",
            "previous_feedback": "; ".join(result.issues) if result.issues else ""
        }

    def human_review(state):
        print("  ⏸️ [人工审核] 等待用户输入...")
        features = state.get("features", {})
        enabled = []
        if features.get("translate"): enabled.append("翻译")
        if features.get("image"): enabled.append("配图")
        if features.get("publish"): enabled.append("发布")

        decision = interrupt({
            "stage": "article_review",
            "article": state.get("article", ""),
            "edit_feedback": state.get("edit_feedback", ""),
            "ai_score": state.get("ai_quality_score", "N/A"),
            "ai_issues": state.get("ai_quality_issues", ""),
            "enabled_features": enabled,
            "message": "AI自检评分: {}。已启用功能: {}。请输入操作：continue(继续) / stop(终止)，或输入具体修改意见".format(
                state.get("ai_quality_score", "N/A"),
                ", ".join(enabled) if enabled else "仅写作"
            )
        })
        print("  ✅ [人工审核] 收到决策: {}".format(decision))
        return {"human_decision": decision}

    def ai_image_decision(state):
        print("  🤖 [AI决策] 判断是否需要配图...")
        article = state.get("article", "")
        prompt = """请判断以下文章是否需要配图，输出结构化结果：

文章：
{}

判断标准：
- 技术概念类、数据可视化类、教程类文章通常需要配图
- 纯观点评论、简短资讯可能不需要配图
- 如果文章涉及具体产品、界面、流程，建议配图""".format(article[:800])

        try:
            structured_model = model.with_structured_output(ImageDecision)
            result = structured_model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            print("  ⚠️ [AI决策] 结构化输出失败，默认需要配图: {}".format(e))
            result = ImageDecision(need_image=True, reason="默认需要配图")

        print("  🤖 [AI决策] 需要配图: {}, 理由: {}".format(result.need_image, result.reason))
        return {
            "ai_need_image": result.need_image,
            "ai_image_reason": result.reason
        }

    def ai_publish_plan(state):
        print("  🤖 [AI决策] 制定发布计划...")
        article = state.get("article", "")
        prompt = """请根据以下文章内容，制定多平台发布计划，输出结构化结果：

文章：
{}

请分析：
1. 这篇文章最适合在哪些平台发布？（wechat/知乎/weibo）
2. 每个平台应该采用什么风格？
3. 哪些平台可能不太适合？为什么？""".format(article[:800])

        try:
            structured_model = model.with_structured_output(PublishPlan)
            result = structured_model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            print("  ⚠️ [AI决策] 结构化输出失败，默认全平台发布: {}".format(e))
            result = PublishPlan(
                platforms=["wechat", "zhihu", "weibo"],
                wechat_style="深度长文",
                zhihu_style="专业分析",
                weibo_style="轻松短句"
            )

        print("  🤖 [AI决策] 发布平台: {}".format(", ".join(result.platforms)))
        return {
            "ai_publish_platforms": ",".join(result.platforms),
            "ai_publish_styles": "微信:{} | 知乎:{} | 微博:{}".format(
                result.wechat_style, result.zhihu_style, result.weibo_style
            )
        }

    def trans_fan_out(state):
        sends = []
        if not state.get("english_version"):
            sends.append(Send("translate_one", {
                "lang": "en",
                "text": state["article"],
                "topic": state["topic"]
            }))
        if not state.get("japanese_version"):
            sends.append(Send("translate_one", {
                "lang": "ja",
                "text": state["article"],
                "topic": state["topic"]
            }))
        print("  🌐 [翻译派发] 并行派发 {} 个翻译任务".format(len(sends)))
        return sends

    def translate_one(state):
        lang = state["lang"]
        text = state["text"]
        if lang == "en":
            print("      🇺🇸 [翻译-英文] ...")
            prompt = "翻译成地道英文：\n\n{}".format(text[:800])
            result = model.invoke([HumanMessage(content=prompt)]).content
            return {"translations": [{"lang": "en", "text": result}]}
        elif lang == "ja":
            print("      🇯🇵 [翻译-日文] ...")
            prompt = "翻译成地道日文：\n\n{}".format(text[:800])
            result = model.invoke([HumanMessage(content=prompt)]).content
            return {"translations": [{"lang": "ja", "text": result}]}
        return {"translations": []}

    def trans_aggregate(state):
        updates = {}
        for item in state.get("translations", []):
            if item["lang"] == "en":
                updates["english_version"] = item["text"]
            elif item["lang"] == "ja":
                updates["japanese_version"] = item["text"]
        print("  🌐 [翻译聚合] 完成 {} 种语言".format(len(state.get("translations", []))))
        return updates

    def pub_fan_out(state):
        platforms_str = state.get("ai_publish_platforms", "wechat,zhihu,weibo")
        platforms = [p.strip() for p in platforms_str.split(",") if p.strip()]
        sends = []
        if "wechat" in platforms and not state.get("wechat_url"):
            sends.append(Send("pub_wechat", {"topic": state["topic"]}))
        if "zhihu" in platforms and not state.get("zhihu_url"):
            sends.append(Send("pub_zhihu", {"topic": state["topic"]}))
        if "weibo" in platforms and not state.get("weibo_text"):
            sends.append(Send("pub_weibo", {"topic": state["topic"], "article": state["article"]}))
        print("  📱 [发布派发] AI选择平台: {}, 并行派发 {} 个任务".format(", ".join(platforms), len(sends)))
        return sends

    def pub_wechat(state):
        print("      📱 [微信] ...")
        return {"wechat_url": "https://mp.weixin.qq.com/simulated/{}".format(hash(state["topic"]) % 100000)}

    def pub_zhihu(state):
        print("      💡 [知乎] ...")
        return {"zhihu_url": "https://zhuanlan.zhihu.com/p/simulated_{}".format(hash(state["topic"]) % 100000)}

    def pub_weibo(state):
        print("      🔥 [微博] ...")
        prompt = "生成微博文案（140字内）：\n{}".format(state["article"][:300])
        return {"weibo_text": model.invoke([HumanMessage(content=prompt)]).content}

    def pub_aggregate(state):
        print("  📱 [发布聚合] 所有平台发布完成")
        return {}

    def m_s(s):
        features = s.get("features", {})

        if not s.get("article"):
            return {"next_agent": "writing"}

        if s.get("article") and s.get("ai_quality_score") == 0:
            return {"next_agent": "ai_quality_check"}

        if s.get("ai_quality_score", 10) < 7 and s.get("human_decision") != "continue":
            print("  🔄 [主调度] AI自检评分 {} 低于7分，自动重写".format(s["ai_quality_score"]))
            return {
                "next_agent": "writing",
                "human_decision": "",
                "article": "",
                "edit_feedback": "",
                "ai_quality_score": 0,
                "ai_quality_issues": "",
                "previous_feedback": "AI自检发现问题，请改进：{}".format(s.get("ai_quality_issues", ""))
            }

        if s.get("article") and not s.get("human_decision"):
            return {"next_agent": "human_review"}

        decision = s.get("human_decision", "")

        if decision == "stop":
            return {"next_agent": "FINISH"}

        if decision != "continue" and decision:
            print("  🔄 [主调度] 用户要求重写，意见：{}".format(decision))
            return {
                "next_agent": "writing",
                "human_decision": "",
                "article": "",
                "edit_feedback": "",
                "ai_quality_score": 0,
                "ai_quality_issues": "",
                "previous_feedback": decision,
                "translations": [],
                "english_version": "",
                "japanese_version": "",
                "image_prompt": "",
                "image_url": "",
                "wechat_url": "",
                "zhihu_url": "",
                "weibo_text": "",
                "ai_need_image": None,
                "ai_image_reason": "",
                "ai_publish_platforms": "",
                "ai_publish_styles": ""
            }

        if features.get("translate") and (not s.get("english_version") or not s.get("japanese_version")):
            return {"next_agent": "trans_fan_out"}

        if features.get("image"):
            if s.get("ai_need_image") is None:
                return {"next_agent": "ai_image_decision"}
            if s.get("ai_need_image") and not s.get("image_url"):
                return {"next_agent": "image"}

        if features.get("publish"):
            if not s.get("ai_publish_platforms"):
                return {"next_agent": "ai_publish_plan"}
            platforms_str = s.get("ai_publish_platforms", "")
            platforms = [p.strip() for p in platforms_str.split(",") if p.strip()]
            need_pub = False
            if "wechat" in platforms and not s.get("wechat_url"): need_pub = True
            if "zhihu" in platforms and not s.get("zhihu_url"): need_pub = True
            if "weibo" in platforms and not s.get("weibo_text"): need_pub = True
            if need_pub:
                return {"next_agent": "pub_fan_out"}

        return {"next_agent": "FINISH"}

    def build_graph(features):
        print("  🔧 [动态构图] 功能配置: translate={}, image={}, publish={}".format(
            features.get("translate"), features.get("image"), features.get("publish")
        ))

        mg = StateGraph(MState)
        mg.add_node("m_s", m_s)
        mg.add_node("writing", writing_sg)
        mg.add_node("ai_quality_check", ai_quality_check)
        mg.add_node("human_review", human_review)

        if features.get("translate"):
            mg.add_node("trans_fan_out", trans_fan_out)
            mg.add_node("translate_one", translate_one)
            mg.add_node("trans_aggregate", trans_aggregate)

        if features.get("image"):
            mg.add_node("ai_image_decision", ai_image_decision)
            mg.add_node("image", image_sg)

        if features.get("publish"):
            mg.add_node("ai_publish_plan", ai_publish_plan)
            mg.add_node("pub_fan_out", pub_fan_out)
            mg.add_node("pub_wechat", pub_wechat)
            mg.add_node("pub_zhihu", pub_zhihu)
            mg.add_node("pub_weibo", pub_weibo)
            mg.add_node("pub_aggregate", pub_aggregate)

        mg.set_entry_point("m_s")

        cond_map = {
            "writing": "writing",
            "ai_quality_check": "ai_quality_check",
            "human_review": "human_review",
            "FINISH": END,
            "m_s": "m_s"
        }
        if features.get("translate"):
            cond_map["trans_fan_out"] = "trans_fan_out"
        if features.get("image"):
            cond_map["ai_image_decision"] = "ai_image_decision"
        if features.get("publish"):
            cond_map["ai_publish_plan"] = "ai_publish_plan"
            cond_map["pub_fan_out"] = "pub_fan_out"

        mg.add_conditional_edges("m_s", lambda s: s["next_agent"], cond_map)

        if features.get("translate"):
            mg.add_conditional_edges("trans_fan_out", trans_fan_out, ["translate_one"])
            mg.add_edge("translate_one", "trans_aggregate")
            mg.add_edge("trans_aggregate", "m_s")

        if features.get("image"):
            mg.add_edge("image", "m_s")

        if features.get("publish"):
            mg.add_conditional_edges("pub_fan_out", pub_fan_out, ["pub_wechat", "pub_zhihu", "pub_weibo"])
            mg.add_edge("pub_wechat", "pub_aggregate")
            mg.add_edge("pub_zhihu", "pub_aggregate")
            mg.add_edge("pub_weibo", "pub_aggregate")
            mg.add_edge("pub_aggregate", "m_s")

        for n in ["writing", "ai_quality_check", "human_review"]:
            mg.add_edge(n, "m_s")
        if features.get("image"):
            mg.add_edge("ai_image_decision", "m_s")
        if features.get("publish"):
            mg.add_edge("ai_publish_plan", "m_s")

        return mg.compile(checkpointer=memory)

    def _process_stream(stream_iter, logs):
        global waiting_for_human, interrupt_payload
        for chunk in stream_iter:
            if "__interrupt__" in chunk:
                waiting_for_human = True
                interrupt_payload = chunk["__interrupt__"][0].value
                logs.append("-" * 40)
                logs.append("⏸️ 等待人工审核...")
                logs.append("📄 文章预览：{}...".format(interrupt_payload.get("article", "")[:200]))
                logs.append("🤖 AI自检评分: {}".format(interrupt_payload.get("ai_score", "N/A")))
                return "interrupt"
            for node_name in chunk.keys():
                if node_name == "__end__":
                    continue
                name_map = {
                    "m_s": "👔 主调度",
                    "writing": "📝 写作子图",
                    "ai_quality_check": "🤖 AI自检",
                    "human_review": "⏸️ 人工审核",
                    "trans_fan_out": "🌐 翻译派发",
                    "translate_one": "🌐 翻译执行",
                    "trans_aggregate": "🌐 翻译聚合",
                    "ai_image_decision": "🤖 AI配图决策",
                    "image": "🎨 配图子图",
                    "ai_publish_plan": "🤖 AI发布计划",
                    "pub_fan_out": "📱 发布派发",
                    "pub_wechat": "📱 微信",
                    "pub_zhihu": "💡 知乎",
                    "pub_weibo": "🔥 微博",
                    "pub_aggregate": "📱 发布聚合"
                }
                logs.append("✅ {} 执行完成".format(name_map.get(node_name, node_name)))
        return "done"

    def _build_final_yield(state, logs):
        logs.append("-" * 40)
        if state.get("human_decision") == "stop":
            logs.append("🛑 用户已终止流程")
        else:
            logs.append("🎉 全部完成！")
            if state.get("ai_quality_score"):
                logs.append("🤖 AI自检评分: {}/10".format(state["ai_quality_score"]))
            if state.get("ai_image_reason"):
                logs.append("🎨 配图决策: {}".format(state["ai_image_reason"]))
            if state.get("ai_publish_platforms"):
                logs.append("📱 发布平台: {}".format(state["ai_publish_platforms"]))
            if state.get("ai_publish_styles"):
                logs.append("🎨 风格建议: {}".format(state["ai_publish_styles"]))

        img_html = ""
        if state.get("image_url"):
            img_html = '<img src="{}" width="512" style="border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">'.format(
                state["image_url"]
            )

        return (
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

    def generate_all(topic, do_translate, do_image, do_publish):
        global app, current_config, active_config, waiting_for_human, interrupt_payload

        if not topic.strip():
            yield "❌ 请输入主题", "", "", "", "", "", "", "", "", ""
            return

        features = {
            "translate": do_translate,
            "image": do_image,
            "publish": do_publish
        }

        app = build_graph(features)

        config = {"configurable": {"thread_id": "gradio_{}".format(hash(topic) & 0xFFFFFFFF)}}
        current_config = config
        active_config = config
        waiting_for_human = False
        interrupt_payload = None

        inputs = {
            "topic": topic,
            "messages": [("user", "帮我写一篇关于{}的文章".format(topic))],
            "previous_feedback": "",
            "translations": [],
            "features": features,
            "ai_quality_score": 0,
            "ai_need_image": None,
            "ai_publish_platforms": ""
        }

        logs = ["🚀 开始生成主题：{}".format(topic), "-" * 40]
        logs.append("🔧 已启用功能: {}".format(
            ", ".join([k for k, v in features.items() if v]) or "仅写作"
        ))
        if TAVILY_AVAILABLE:
            logs.append("🔍 Web Search 已启用（Tavily）")
        else:
            logs.append("⚠️ Web Search 未启用，调研基于模型知识")
        yield "\n".join(logs), "生成中...", "", "", "", "", "", "", "", ""

        result = _process_stream(app.stream(inputs, config), logs)

        if result == "interrupt":
            yield (
                "\n".join(logs),
                interrupt_payload.get("article", ""),
                interrupt_payload.get("edit_feedback", ""),
                "", "", "", "",
                "", "",
                "⏸️ 等待用户输入：continue / stop / 或输入具体修改意见"
            )
            return

        state = app.get_state(config).values
        yield _build_final_yield(state, logs)

    def resume_all(user_input):
        global active_config, waiting_for_human

        if not waiting_for_human or active_config is None or app is None:
            yield "❌ 没有待恢复的任务，请先点击【开始生成】", "", "", "", "", "", "", "", "", ""
            return

        logs = ["🔄 恢复执行...", "-" * 40]
        yield "\n".join(logs), "恢复中...", "", "", "", "", "", "", "", ""

        result = _process_stream(app.stream(Command(resume=user_input), active_config), logs)

        if result == "interrupt":
            yield (
                "\n".join(logs),
                interrupt_payload.get("article", ""),
                interrupt_payload.get("edit_feedback", ""),
                "", "", "", "",
                "", "",
                "⏸️ 等待用户输入：continue / stop / 或输入具体修改意见"
            )
            return

        state = app.get_state(active_config).values
        waiting_for_human = False
        yield _build_final_yield(state, logs)

    def refresh_history():
        global current_config
        if current_config is None or app is None:
            return "请先点击【开始生成】", gr.update(choices=[])

        history = list(app.get_state_history(current_config))
        lines = []
        choices = []

        for i, snap in enumerate(history):
            node = "unknown"
            if snap.metadata and "langgraph_node" in snap.metadata:
                node = snap.metadata["langgraph_node"]

            cp_id = snap.config["configurable"].get("checkpoint_id", "N/A")
            cp_short = cp_id[:16] if len(cp_id) > 16 else cp_id

            vals = []
            if snap.values:
                if snap.values.get("article"): vals.append("文章")
                if snap.values.get("english_version"): vals.append("英文")
                if snap.values.get("image_url"): vals.append("配图")
                if snap.values.get("wechat_url"): vals.append("发布")
                if snap.values.get("ai_quality_score"): vals.append("AI评分{}".format(snap.values["ai_quality_score"]))
            summary = "+".join(vals) if vals else "初始"

            line = "[{}] {} | {} | cp:{}".format(i, node, summary, cp_short)
            lines.append(line)
            choices.append((line, str(i)))

        return "\n".join(lines), gr.update(choices=choices)

    def time_travel_and_run(step_index_str, modify_json):
        global active_config, waiting_for_human

        if not step_index_str:
            yield "❌ 请先选择历史步骤", "", "", "", "", "", "", "", "", ""
            return

        step_index = int(step_index_str)
        history = list(app.get_state_history(current_config))

        if step_index >= len(history):
            yield "❌ 步骤不存在", "", "", "", "", "", "", "", "", ""
            return

        target = history[step_index]
        target_config = target.config
        active_config = target_config

        if modify_json and modify_json.strip():
            try:
                updates = json.loads(modify_json)
                app.update_state(target_config, updates)
                print("  ⏪ [Time Travel] 应用修改: {}".format(updates))
            except Exception as e:
                yield "❌ JSON格式错误: {}".format(str(e)), "", "", "", "", "", "", "", "", ""
                return

        logs = ["⏪ 回到历史步骤 [{}] 重新执行...".format(step_index), "-" * 40]
        yield "\n".join(logs), *[""]*8, "执行中..."

        result = _process_stream(app.stream(None, target_config), logs)

        if result == "interrupt":
            yield (
                "\n".join(logs),
                interrupt_payload.get("article", ""),
                interrupt_payload.get("edit_feedback", ""),
                "", "", "", "",
                "", "",
                "⏸️ 等待用户输入：continue / stop / 或输入具体修改意见"
            )
            return

        state = app.get_state(target_config).values
        waiting_for_human = False
        yield _build_final_yield(state, logs)

    def preset_modify(preset_name):
        if preset_name == "trans":
            return '{"english_version": "", "japanese_version": "", "translations": []}'
        elif preset_name == "image":
            return '{"image_prompt": "", "image_url": "", "ai_need_image": true}'
        elif preset_name == "publish":
            return '{"wechat_url": "", "zhihu_url": "", "weibo_text": "", "ai_publish_platforms": ""}'
        elif preset_name == "article":
            return '{"article": "", "edit_feedback": "", "human_decision": "", "ai_quality_score": 0}'
        return ""

    with gr.Blocks(title="🏭 AI 内容工厂 - Tavily搜索") as demo:
        gr.Markdown("""
        # 🏭 AI 内容工厂（Tavily Web Search 版）
        **写作调研节点已接入 Tavily 实时搜索！**

        > 🔍 新特性：调研节点先搜索互联网获取真实数据、政策、案例，再基于搜索结果撰写文章。
        > 🤖 动态构图 + AI 决策 + 并行执行 + 人工审核 + 时间旅行 + LangSmith 监控，全部保留。
        """)

        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    label="📝 文章主题",
                    placeholder="例如：人工智能对现代教育的影响",
                    value="人工智能对现代教育的影响"
                )
            with gr.Column(scale=1):
                gr.Markdown("**功能模块**")
                cb_translate = gr.Checkbox(label="🌐 翻译", value=True)
                cb_image = gr.Checkbox(label="🎨 配图", value=True)
                cb_publish = gr.Checkbox(label="📱 发布", value=True)
            with gr.Column(scale=1):
                generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")

        progress_box = gr.Textbox(
            label="📊 生成进度",
            lines=12,
            interactive=False,
            value="等待开始..."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⏸️ 人工审核区")
                user_decision = gr.Textbox(
                    label="审核决策 / 修改意见",
                    placeholder="continue / stop / 或输入具体修改意见",
                    value="continue",
                    info="文章写完后在此输入。输入具体意见（如'请增加案例'）AI 会重写"
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

            with gr.TabItem("⏪ 时间旅行"):
                gr.Markdown("""
                ### 回到任意历史步骤，修改状态后重新执行
                **使用步骤：**
                1. 先点击【开始生成】完成至少一次流程
                2. 点击【🔄 刷新历史】查看所有执行步骤
                3. 在【选择回溯点】下拉框选择要回到的步骤
                4. （可选）在【状态修改JSON】中输入要修改的字段，或点击下方快捷预设
                5. 点击【⏪ 回到此步骤并执行】
                """)

                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 刷新历史", variant="secondary")

                history_box = gr.Textbox(
                    label="执行历史",
                    lines=6,
                    interactive=False,
                    value="点击上方按钮刷新历史..."
                )

                with gr.Row():
                    step_selector = gr.Dropdown(
                        label="选择回溯点",
                        choices=[],
                        value=None,
                        info="选择要回到的历史步骤"
                    )

                modify_json = gr.Textbox(
                    label="状态修改 JSON（可选）",
                    placeholder='{"english_version": ""}',
                    lines=2,
                    info="输入JSON格式的新状态值，覆盖选中步骤的状态后重新执行"
                )

                with gr.Row():
                    gr.Markdown("**快捷预设：**")
                with gr.Row():
                    preset_article_btn = gr.Button("📝 回到写作前", size="sm")
                    preset_trans_btn = gr.Button("🌐 回到翻译前", size="sm")
                    preset_image_btn = gr.Button("🎨 回到配图前", size="sm")
                    preset_publish_btn = gr.Button("📱 回到发布前", size="sm")

                with gr.Row():
                    time_travel_btn = gr.Button("⏪ 回到此步骤并执行", variant="primary", size="lg")

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

        generate_btn.click(
            fn=generate_all,
            inputs=[topic_input, cb_translate, cb_image, cb_publish],
            outputs=outputs
        )
        topic_input.submit(
            fn=generate_all,
            inputs=[topic_input, cb_translate, cb_image, cb_publish],
            outputs=outputs
        )
        resume_btn.click(fn=resume_all, inputs=user_decision, outputs=outputs)

        refresh_history_btn.click(fn=refresh_history, outputs=[history_box, step_selector])

        preset_article_btn.click(fn=lambda: preset_modify("article"), outputs=modify_json)
        preset_trans_btn.click(fn=lambda: preset_modify("trans"), outputs=modify_json)
        preset_image_btn.click(fn=lambda: preset_modify("image"), outputs=modify_json)
        preset_publish_btn.click(fn=lambda: preset_modify("publish"), outputs=modify_json)

        time_travel_btn.click(
            fn=time_travel_and_run,
            inputs=[step_selector, modify_json],
            outputs=outputs
        )

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)