"""轻量文档与查询分类工具。"""

from __future__ import annotations


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "compliance": (
        "合规",
        "法规",
        "监管",
        "准入",
        "市场准入",
        "欧盟",
        "eu",
        "2023/1542",
        "电池法规",
        "碳足迹",
        "碳排放",
        "碳减排",
        "esg",
        "供应链",
        "供应商",
        "风险预警",
        "battery regulation",
        "carbon footprint",
    ),
    "schedule": (
        "日程",
        "行程",
        "排期",
        "计划表",
        "时间表",
        "calendar",
        "schedule",
        "itinerary",
    ),
    "travel": ("旅游", "景点", "酒店", "门票", "预约", "余票", "故宫", "博物院"),
    "contract": ("合同", "协议", "条款", "违约", "甲方", "乙方"),
    "policy": ("制度", "规章", "流程", "规范", "管理办法", "手册"),
    "technical": ("算法","模型","系统架构","区块链","共识", "代码","接口","数据库","性能",),
    "finance": ("财务", "报销", "预算", "发票", "成本", "收入", "利润"),
    "hr": ("招聘", "绩效", "员工", "薪酬", "考勤", "人事"),
    "meeting": ("会议", "纪要", "待办", "议程", "参会"),
}

COMPATIBLE_CATEGORIES: dict[str, set[str]] = {
    "compliance": {"compliance", "policy", "finance"},
    "schedule": {"schedule", "travel", "meeting"},
    "travel": {"travel", "schedule"},
    "contract": {"contract", "policy"},
    "policy": {"policy", "contract", "hr"},
    "technical": {"technical"},
    "finance": {"finance", "policy"},
    "hr": {"hr", "policy", "meeting"},
    "meeting": {"meeting", "schedule", "policy"},
}


def classify_document(file_name: str, text: str) -> str:
    """根据文件名和文本关键词返回粗粒度类别。"""
    haystack = f"{file_name}\n{text}".lower()
    best_category = "general"
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword.lower() in haystack)
        if score > best_score:
            best_category = category
            best_score = score
    return best_category


def categories_compatible(query_category: str, document_category: str) -> bool:
    """判断查询类别和文档类别是否可共同用于 RAG。"""
    if query_category == "general":
        return True
    if document_category == "general":
        return False
    return document_category in COMPATIBLE_CATEGORIES.get(
        query_category,
        {query_category},
    )
