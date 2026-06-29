#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
八字計算模組 - 使用 lunar-python 計算四柱及五行
"""

from lunar_python import Solar

STEM_ELEMENT = {
    "甲": "木", "乙": "木", "丙": "火", "丁": "火",
    "戊": "土", "己": "土", "庚": "金", "辛": "金",
    "壬": "水", "癸": "水",
}

BRANCH_ELEMENT = {
    "子": "水", "丑": "土", "寅": "木", "卯": "木",
    "辰": "土", "巳": "火", "午": "火", "未": "土",
    "申": "金", "酉": "金", "戌": "土", "亥": "水",
}

BRANCH_SHENGXIAO = {
    "子": "鼠", "丑": "牛", "寅": "虎", "卯": "兔", "辰": "龍",
    "巳": "蛇", "午": "馬", "未": "羊", "申": "猴", "酉": "雞",
    "戌": "狗", "亥": "豬",
}

# 時辰名稱（以小時區分）
HOUR_ZHI = [
    "子", "丑", "丑", "寅", "寅", "卯", "卯", "辰", "辰", "巳", "巳", "午",
    "午", "未", "未", "申", "申", "酉", "酉", "戌", "戌", "亥", "亥", "子",
]

WUXING_LABEL = {"金": "金", "木": "木", "水": "水", "火": "火", "土": "土"}


def _wuxing_summary(wuxing: dict) -> str:
    total = sum(wuxing.values())
    if total == 0:
        return ""
    dominant = max(wuxing, key=wuxing.get)
    weak = min(wuxing, key=wuxing.get)
    return f"命局五行中{dominant}較旺（{wuxing[dominant]}個），{weak}較弱（{wuxing[weak]}個）"


def get_zodiac_from_year(year: int) -> str:
    """根據年份粗略返回生肖（正式應以農曆年為準）"""
    zodiacs = ["猴", "雞", "狗", "豬", "鼠", "牛", "虎", "兔", "龍", "蛇", "馬", "羊"]
    return zodiacs[year % 12]


def get_hour_pillar(hour: int) -> str:
    """根據小時返回時柱地支名稱"""
    return HOUR_ZHI[hour % 24]


def calculate_bazi(year: int, month: int, day: int,
                   hour: int | None = None, minute: int = 0) -> dict:
    """
    計算八字四柱。
    輸入：西曆年月日，時辰（24小時制）可選。
    返回 dict。
    """
    try:
        if hour is not None:
            solar = Solar.fromYmdHms(year, month, day, hour, minute, 0)
        else:
            solar = Solar.fromYmd(year, month, day)
        lunar = solar.getLunar()
    except Exception as e:
        raise ValueError(f"日期計算錯誤：{e}")

    year_gz  = lunar.getYearInGanZhi()
    month_gz = lunar.getMonthInGanZhi()
    day_gz   = lunar.getDayInGanZhi()
    _sx_trad = {"鼠":"鼠","牛":"牛","虎":"虎","兔":"兔","龙":"龍","蛇":"蛇",
                "马":"馬","羊":"羊","猴":"猴","鸡":"雞","狗":"狗","猪":"豬"}
    shengxiao = _sx_trad.get(lunar.getYearShengXiao(), lunar.getYearShengXiao())

    # 時柱
    hour_gz = None
    if hour is not None:
        hour_gz = lunar.getTimeInGanZhi()

    # 農曆日期字串
    lunar_date = (
        f"農曆{year_gz}年"
        f"{lunar.getMonthInChinese()}月"
        f"{lunar.getDayInChinese()}"
    )

    # 五行統計（天干+地支各計1個）
    wuxing: dict[str, int] = {"金": 0, "木": 0, "水": 0, "火": 0, "土": 0}
    pillars = [year_gz, month_gz, day_gz] + ([hour_gz] if hour_gz else [])
    for gz in pillars:
        if gz and len(gz) >= 1 and gz[0] in STEM_ELEMENT:
            wuxing[STEM_ELEMENT[gz[0]]] += 1
        if gz and len(gz) >= 2 and gz[1] in BRANCH_ELEMENT:
            wuxing[BRANCH_ELEMENT[gz[1]]] += 1

    bazi_str = " ".join(filter(None, [year_gz, month_gz, day_gz, hour_gz]))

    summary = (
        f"八字：{bazi_str}。"
        f"生肖屬{shengxiao}，農曆生於{lunar_date}。"
        f"{_wuxing_summary(wuxing)}。"
    )

    return {
        "shengxiao":    shengxiao,
        "year_pillar":  year_gz,
        "month_pillar": month_gz,
        "day_pillar":   day_gz,
        "hour_pillar":  hour_gz,
        "lunar_date":   lunar_date,
        "bazi_string":  bazi_str,
        "wuxing":       wuxing,
        "summary":      summary,
    }
