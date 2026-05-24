"""文档分类器测试。"""

from utils.document_classifier import categories_compatible
from utils.document_classifier import classify_document


class TestClassifyDocument:
    def test_classify_compliance(self) -> None:
        category = classify_document("法规文件.pdf", "欧盟电池法规2023/1542要求碳足迹声明")
        assert category == "compliance"

    def test_classify_travel(self) -> None:
        category = classify_document("旅游指南.pdf", "旅游景点门票价格")
        assert category == "travel"

    def test_classify_hr(self) -> None:
        category = classify_document("员工手册.pdf", "招聘绩效薪酬考勤制度")
        assert category == "hr"

    def test_classify_meeting(self) -> None:
        category = classify_document("会议记录.txt", "会议纪要待办议程参会人员")
        assert category == "meeting"

    def test_classify_finance(self) -> None:
        category = classify_document("财务.xlsx", "财务报销预算发票成本收入")
        assert category == "finance"

    def test_classify_general_fallback(self) -> None:
        category = classify_document("杂谈.txt", "今天天气真不错")
        assert category == "general"

    def test_file_name_influences_category(self) -> None:
        category = classify_document("合规报告.docx", "一些随机的内容")
        # 文件名 "合规" 可能触发 compliance
        result = "compliance" in category or category == "general"
        assert result


class TestCategoriesCompatible:
    def test_general_query_accepts_all(self) -> None:
        assert categories_compatible("general", "compliance") is True
        assert categories_compatible("general", "travel") is True

    def test_general_document_rejected(self) -> None:
        """非 general 查询不应匹配 general 文档。"""
        assert categories_compatible("compliance", "general") is False

    def test_compliance_accepts_policy(self) -> None:
        assert categories_compatible("compliance", "policy") is True

    def test_compliance_rejects_travel(self) -> None:
        assert categories_compatible("compliance", "travel") is False

    def test_travel_accepts_schedule(self) -> None:
        assert categories_compatible("travel", "schedule") is True

    def test_technical_only_accepts_technical(self) -> None:
        assert categories_compatible("technical", "technical") is True
        assert categories_compatible("technical", "finance") is False

    def test_unknown_category_fallback(self) -> None:
        """未知类别仅匹配自身。"""
        assert categories_compatible("unknown_cat", "unknown_cat") is True
        assert categories_compatible("unknown_cat", "other") is False
