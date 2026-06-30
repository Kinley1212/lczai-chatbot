#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
李丞責2026運勢AI — Flask 後端（google.genai 新SDK版）
路由：GET / → 主網站  GET /chat → 聊天介面  POST /analyze → 運勢分析  POST /chat → 聊天API
"""

import os
import re
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag import RAGEngine
from bazi import calculate_bazi

# ── Gemini 設定 ─────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)

# ── 載入 system prompt ──────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_system_prompt() -> str:
    path = os.path.join(BASE, "system_prompt.md")
    if not os.path.exists(path):
        return "你是李丞責博士本人，香港著名玄學風水專家，以繁體中文書面語回答問題。"
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    marker = "## 正式System Prompt（複製此段用於代碼）"
    if marker in raw:
        raw = raw[raw.index(marker) + len(marker):].strip()
    menu_marker = "## 選單設計"
    if menu_marker in raw:
        raw = raw[:raw.index(menu_marker)].strip()
    return raw

SYSTEM_PROMPT = _load_system_prompt()

# ── 選單定義 ─────────────────────────────────────────────
MENUS: dict[str, list[str]] = {
    "main": ["紫微斗數流年", "奇門遁甲", "塔羅牌占卜", "卜卦（六爻）", "今日運勢", "關於李丞責博士"],
    "ziwei": ["今年財運", "今年感情", "今年事業", "注意月份", "化解方法", "返回主選單"],
    "qimen": ["問事業", "問財運", "問感情", "問搬遷", "如何增強運勢", "返回主選單"],
}

# ── Flask App ──────────────────────────────────────────
app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

rag = RAGEngine()


# ── 工具函數 ───────────────────────────────────────────

def detect_menu(msg: str) -> str:
    if any(k in msg for k in ["紫微", "斗數", "流年", "命盤"]):
        return "ziwei"
    if any(k in msg for k in ["奇門", "遁甲"]):
        return "qimen"
    if "塔羅" in msg:
        return "tarot"
    if any(k in msg for k in ["卜卦", "六爻"]):
        return "gua"
    if any(k in msg for k in ["返回", "主選單"]):
        return "main"
    if any(k in msg for k in ["財運", "感情", "事業", "健康", "月份", "化解", "運勢", "生肖"]):
        return "ziwei"
    return "main"


def call_gemini_raw(prompt: str, max_tokens: int = 2048,
                    disable_thinking: bool = False) -> str:
    """單輪呼叫 Gemini（無歷史）"""
    cfg_kwargs: dict = dict(temperature=0.7, max_output_tokens=max_tokens)
    if disable_thinking:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    return response.text


def call_gemini(system_prompt: str, context: str,
                user_msg: str, history: list[dict]) -> str:
    """多輪對話呼叫 Gemini"""
    full_user_msg = f"{system_prompt}\n\n{context}\n\n用戶問題：{user_msg}"
    contents: list[types.Content] = []
    for turn in history[-5:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if role not in ("user", "model") or not content:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=full_user_msg)]))
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
    )
    return response.text


def _parse_fortune_sections(text: str) -> dict[str, str]:
    """
    從 Gemini 回覆中提取各段。
    使用 SECTION_ 前綴標記 + lookahead，避免內文星名【】截斷問題。
    """
    sections = {"overall": "", "wealth": "", "career": "", "love": "", "health": "", "advice": ""}

    # 前瞻：下一個 SECTION（不含自身）或字串結尾
    next_sec = r"(?=SECTION_(?:財運|事業|感情|健康|化解建議)|$)"

    patterns = {
        "overall": rf"SECTION_整體運勢\s*(.*?)\s*(?=SECTION_(?:財運|事業|感情|健康|化解建議)|$)",
        "wealth":  rf"SECTION_財運\s*(.*?)\s*(?=SECTION_(?:事業|感情|健康|化解建議)|$)",
        "career":  rf"SECTION_事業\s*(.*?)\s*(?=SECTION_(?:感情|健康|化解建議)|$)",
        "love":    rf"SECTION_感情\s*(.*?)\s*(?=SECTION_(?:健康|化解建議)|$)",
        "health":  rf"SECTION_健康\s*(.*?)\s*(?=SECTION_化解建議|$)",
        "advice":  rf"SECTION_化解建議\s*(.*?)(?:\s*（本內容.*?）)?\s*$",
    }

    found_any = False
    for key, pat in patterns.items():
        m = re.search(pat, text, re.DOTALL)
        if m:
            sections[key] = m.group(1).strip()
            found_any = True

    if not found_any:
        sections["overall"] = text.strip()

    return sections


# ── 路由 ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index_new.html")


@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "chunks_loaded": rag.chunk_count,
        "gemini_model": GEMINI_MODEL,
        "api_key_set": bool(GEMINI_API_KEY),
    })


@app.route("/menu/<menu_name>")
def menu(menu_name: str):
    buttons = MENUS.get(menu_name, MENUS["main"])
    items = [
        {"label": btn, "action": "nav_main" if btn == "返回主選單" else "send"}
        for btn in buttons
    ]
    return jsonify({"menu": menu_name, "buttons": items})


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    history  = data.get("history", [])

    if not user_msg:
        return jsonify({"error": "message 不能為空"}), 400

    context = rag.get_context(user_msg)

    t0 = time.time()
    try:
        reply = call_gemini(SYSTEM_PROMPT, context, user_msg, history)
    except Exception as e:
        return jsonify({"error": f"Gemini API 錯誤：{e}"}), 500
    elapsed = round(time.time() - t0, 2)

    return jsonify({"reply": reply, "menu": detect_menu(user_msg), "elapsed": elapsed})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    surname    = data.get("surname", "").strip()
    name       = data.get("name", "").strip()
    birth_date = data.get("birth_date", "")      # "YYYY-MM-DD"
    birth_time = data.get("birth_time")           # "HH:MM" or None
    gender     = data.get("gender", "不透露").strip()
    question   = data.get("question", "").strip()
    email      = data.get("email", "")

    if not all([surname, name, birth_date]):
        return jsonify({"error": "姓名、出生日期為必填"}), 400

    # ── 1. 計算八字 ──
    try:
        y, mo, d = [int(x) for x in birth_date.split("-")]
    except Exception:
        return jsonify({"error": "出生日期格式錯誤，應為 YYYY-MM-DD"}), 400

    hour = minute = None
    if birth_time:
        try:
            hh, mm = birth_time.split(":")
            hour, minute = int(hh), int(mm)
        except Exception:
            hour = minute = None

    try:
        bazi = calculate_bazi(y, mo, d, hour, minute or 0)
    except Exception as e:
        return jsonify({"error": f"八字計算錯誤：{e}"}), 400

    full_name = surname + name

    # ── 2. RAG 搜尋：六主題各取一塊原書內容（直接回傳，不經 Gemini 改寫）──
    shengxiao = bazi['shengxiao']
    topic_map = [
        ("整體運勢", "overall"),
        ("財運",     "wealth"),
        ("事業",     "career"),
        ("感情",     "love"),
        ("健康",     "health"),
        ("化解建議", "remedy"),
    ]
    rag_sections: dict = {}
    context_parts: list = []
    seen_ids: set = set()

    for topic_zh, topic_key in topic_map:
        results = rag.search(f"屬{shengxiao} {topic_zh}", top_k=5)
        chunk_text = ""
        for chunk in results:
            cid = chunk.get("id", id(chunk))
            if cid in seen_ids:
                continue
            if chunk.get("zodiac") == shengxiao and chunk.get("topic") == topic_zh:
                seen_ids.add(cid)
                chunk_text = chunk["text"]
                context_parts.append(f"【{topic_zh}｜屬{shengxiao}】\n{chunk_text}")
                break
        if not chunk_text:
            for chunk in results:
                cid = chunk.get("id", id(chunk))
                if cid not in seen_ids and chunk.get("zodiac") == shengxiao:
                    seen_ids.add(cid)
                    chunk_text = chunk["text"]
                    context_parts.append(f"【{topic_zh}｜屬{shengxiao}】\n{chunk_text}")
                    break
        rag_sections[topic_key] = chunk_text

    context = "\n\n".join(context_parts)

    # ── 3. 計算五行強弱摘要 ──
    wuxing = bazi["wuxing"]
    wuxing_parts = []
    for elem, cnt in wuxing.items():
        if cnt == 0:
            wuxing_parts.append(f"缺{elem}")
        elif cnt >= 3:
            wuxing_parts.append(f"{elem}旺")
    wuxing_summary = "、".join(wuxing_parts) if wuxing_parts else "五行較為均衡"
    wuxing_detail = " ".join(f"{e}{wuxing.get(e,0)}" for e in ["金","木","水","火","土"])

    hour_pillar_str = f" {bazi['hour_pillar']}" if bazi.get("hour_pillar") else ""
    bazi_str = f"{bazi['year_pillar']} {bazi['month_pillar']} {bazi['day_pillar']}{hour_pillar_str}"

    # ── 4. 組合 Prompt（七段：六運勢 + 問題解答）──
    prompt = f"""你是李丞責博士本人，現在為用戶提供2026丙午年個人運勢分析。

用戶資料：
- 姓名：{full_name}（{gender}）
- 出生：{bazi['lunar_date']}，生肖屬{shengxiao}
- 八字四柱：{bazi_str}
- 五行狀況：{wuxing_summary}（{wuxing_detail}）
{f"- 用戶問題：{question}" if question else ""}
【書本參考資料】（以下內容來自李丞責2026全書，是分析的最高依據）：
{context}

請根據以上資料，以李丞責博士第一人稱，用繁體中文書面語，為{full_name}提供2026年個人運勢分析。

分析原則：
1. 以書本生肖運勢為主軸和最高依據
2. 在此基礎上，結合用戶的八字四柱和五行狀況，提供個人化的補充分析
3. 如果八字五行與生肖運勢方向一致，可加強說明
4. 如果八字五行與生肖運勢有出入，以生肖運勢為準，用融合的語言表達，例如「雖然你的八字根基如此，但今年的流年氣場⋯⋯」，絕對不可以直接說兩者矛盾或衝突
5. 引用書本中的具體星曜名稱（如唐符、天廚、歲破等），增加可信度

必須嚴格按以下格式輸出{"七" if question else "六"}個部分，不可增刪標題：

【整體運勢】
根據生肖流年運勢，結合八字日主強弱，說明整體氣場走向。150-200字。

重要寫作規則：
- 【整體運勢】可以提及星曜名稱（如祿勳、擎天、病符等）
- 以下六個部分絕對不可提及任何星曜名稱，無論用何種標點符號（【】「」()等）均不可。只描述實際影響，例如說「收入有望增加」而非「有祿勳入命」，說「需注意高危活動」而非「有亡神」，說「有機會獲得重要職責」而非「有擎天」：
  【財運分析】【事業分析】【感情分析】【健康提示】【化解建議】【問題解答】

【財運分析】
以書本財運指引為主，結合五行喜忌，說明進財方向和注意事項。150-200字。

【事業分析】
以書本事業運勢為主，結合四柱特質，說明發展方向和把握時機。150-200字。

【感情分析】
以書本感情運勢為主，結合八字中的感情宮位特質，提供建議。150-200字。

【健康提示】
以書本健康警示為主，結合五行缺失，說明需要注意的身體部位。150-200字。

【化解建議】
以書本的化解方法為主，結合五行補救，提供具體開運建議。150-200字。

{f"""【問題解答】
針對用戶問題「{question}」，結合以上所有分析，給出深入詳盡的回答。300-400字，分2-3個要點展開。
嚴格禁止在此段提及任何星曜名稱，包括但不限於：祿勳、擎天、病符、亡神、的煞、大耗、天解、解神、豹尾、天狗、吊客、月煞、浮沉、血刃、天廚、唐符、歲破等，即使加任何括號或標點均不可。只說「收入有望增加」「有機會晉升」「需注意健康」等實際影響。
末尾加：「（本內容以李丞責著作及八字五行原理為依據，玄學僅供參考。如需深入個人命盤分析，歡迎預約李丞責博士親身批算。）」""" if question else ""}"""

    # ── 5. 呼叫 Gemini ──
    t0 = time.time()
    try:
        raw_reply = call_gemini_raw(prompt, max_tokens=5000, disable_thinking=True)
    except Exception as e:
        return jsonify({"error": f"Gemini API 錯誤：{e}"}), 500
    elapsed = round(time.time() - t0, 2)

    # ── 6. 解析輸出（【標題】格式，清除 markdown）──
    raw_reply = re.sub(r"\*+", "", raw_reply)

    def _extract(text: str, key: str) -> str:
        # 只在行首的【才視為新節點，避免正文內的【星曜】被截斷
        m = re.search(rf"【{key}】\s*\n(.*?)(?=\n【|\Z)", text, re.S)
        return m.group(1).strip() if m else ""

    gemini_sections = {
        "overall": _extract(raw_reply, "整體運勢"),
        "wealth":  _extract(raw_reply, "財運分析"),
        "career":  _extract(raw_reply, "事業分析"),
        "love":    _extract(raw_reply, "感情分析"),
        "health":  _extract(raw_reply, "健康提示"),
        "remedy":  _extract(raw_reply, "化解建議"),
    }
    question_answer = _extract(raw_reply, "問題解答")

    return jsonify({
        "name":            full_name,
        "shengxiao":       bazi["shengxiao"],
        "bazi":            bazi["bazi_string"],
        "year_pillar":     bazi["year_pillar"],
        "month_pillar":    bazi["month_pillar"],
        "day_pillar":      bazi["day_pillar"],
        "hour_pillar":     bazi.get("hour_pillar"),
        "lunar":           bazi["lunar_date"],
        "wuxing":          bazi["wuxing"],
        "wuxing_summary":  wuxing_summary,
        "rag_sections":    rag_sections,
        "gemini_sections": gemini_sections,
        "question":        question,
        "question_answer": question_answer,
        "email_sent":      False,
        "elapsed":         elapsed,
    })


if __name__ == "__main__":
    print(f"System Prompt 載入（{len(SYSTEM_PROMPT)} 字）")
    print(f"知識庫：{rag.chunk_count} 塊")
    print(f"Gemini 模型：{GEMINI_MODEL}")
    print(f"API Key：{'已設定' if GEMINI_API_KEY else '未設定！'}")
    app.run(debug=True, host="0.0.0.0", port=5000)
