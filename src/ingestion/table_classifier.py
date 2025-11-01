"""表格分类和标题提取模块

功能：
1. 根据内容和上下文识别表格类型（财报、募资、其他）
2. 提取或生成合适的表格标题
3. 为后续检索和分析提供结构化信息
"""
import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path


class TableClassifier:
    """表格分类器"""

    # 财报相关关键词
    FINANCIAL_KEYWORDS = [
        '资产负债表', '利润表', '现金流量表', '所有者权益',
        '合并资产', '母公司资产', '财务状况', '经营成果',
        '资产总额', '负债总额', '营业收入', '净利润',
        '现金及现金等价物', '应收账款', '存货'
    ]

    # 募资相关关键词
    FUNDRAISING_KEYWORDS = [
        '募集资金', '资金用途', '募资', '投向', '募投项目',
        '募集配套资金', '发行股份', '认购', '配套融资',
        '资金使用', '资金投向', '募集说明'
    ]

    def __init__(self):
        """初始化分类器"""
        pass

    def classify_table(
        self,
        table_data: pd.DataFrame,
        context_text: str = ""
    ) -> Dict[str, Any]:
        """分类表格

        Args:
            table_data: 表格数据
            context_text: 表格周围的文本（前后若干行）

        Returns:
            分类结果，包含：
            - type: 表格类型 (financial_report | fundraising | other)
            - confidence: 置信度 (0-1)
            - category: 细分类别
            - suggested_title: 建议的标题
        """
        # 检查表格内容
        table_text = self._extract_table_text(table_data)
        combined_text = f"{context_text} {table_text}"

        # 判断是否为财报
        is_financial, financial_score, financial_category = self._is_financial_report(
            table_data, combined_text
        )

        # 判断是否为募资表格
        is_fundraising, fundraising_score, fundraising_category = self._is_fundraising_table(
            table_data, combined_text
        )

        # 决定类型
        if financial_score > fundraising_score and financial_score > 0.3:
            return {
                "type": "financial_report",
                "confidence": financial_score,
                "category": financial_category,
                "suggested_title": self._extract_financial_title(combined_text, financial_category)
            }
        elif fundraising_score > 0.3:
            return {
                "type": "fundraising",
                "confidence": fundraising_score,
                "category": fundraising_category,
                "suggested_title": self._extract_fundraising_title(combined_text)
            }
        else:
            return {
                "type": "other",
                "confidence": 0.0,
                "category": "unknown",
                "suggested_title": ""
            }

    def _extract_table_text(self, df: pd.DataFrame) -> str:
        """提取表格中的所有文本"""
        if df.empty:
            return ""

        # 提取前几行和列名
        text_parts = []

        # 列名
        text_parts.extend([str(col) for col in df.columns])

        # 前几行数据
        for idx in range(min(5, len(df))):
            text_parts.extend([str(val) for val in df.iloc[idx].values])

        return " ".join(text_parts)

    def _is_financial_report(
        self,
        df: pd.DataFrame,
        text: str
    ) -> Tuple[bool, float, str]:
        """判断是否为财务报表

        Returns:
            (是否财报, 置信度分数, 细分类别)
        """
        score = 0.0
        category = "unknown"

        # 关键词匹配
        keyword_matches = sum(1 for kw in self.FINANCIAL_KEYWORDS if kw in text)
        score += keyword_matches * 0.15

        # 特定财报识别
        if '资产负债表' in text or '资产总计' in text:
            category = "balance_sheet"
            score += 0.4
        elif '利润表' in text or '营业收入' in text or '净利润' in text:
            category = "income_statement"
            score += 0.4
        elif '现金流量表' in text or '现金及现金等价物' in text:
            category = "cash_flow_statement"
            score += 0.4
        elif '所有者权益' in text:
            category = "equity_statement"
            score += 0.3

        # 表格结构特征
        if not df.empty:
            # 财报通常有"项目"、"金额"等列
            column_text = " ".join([str(col) for col in df.columns])
            if any(kw in column_text for kw in ['项目', '金额', '本期', '上期', '年初', '年末']):
                score += 0.2

        return score > 0.3, min(score, 1.0), category

    def _is_fundraising_table(
        self,
        df: pd.DataFrame,
        text: str
    ) -> Tuple[bool, float, str]:
        """判断是否为募资表格

        Returns:
            (是否募资表格, 置信度分数, 细分类别)
        """
        score = 0.0
        category = "unknown"

        # 关键词匹配
        keyword_matches = sum(1 for kw in self.FUNDRAISING_KEYWORDS if kw in text)
        score += keyword_matches * 0.2

        # 细分类别
        if '募集资金' in text and '使用' in text:
            category = "usage"
            score += 0.4
        elif '募集资金' in text and ('来源' in text or '认购' in text):
            category = "source"
            score += 0.4
        elif '配套融资' in text or '发行股份' in text:
            category = "issuance"
            score += 0.3

        return score > 0.3, min(score, 1.0), category

    def _extract_financial_title(self, text: str, category: str) -> str:
        """提取财报标题"""
        # 标题模板
        category_names = {
            "balance_sheet": "资产负债表",
            "income_statement": "利润表",
            "cash_flow_statement": "现金流量表",
            "equity_statement": "所有者权益变动表"
        }

        base_title = category_names.get(category, "财务报表")

        # 尝试提取完整标题（包含公司名、期间等）
        # 例如："合并资产负债表（2023年12月31日）"
        patterns = [
            r'(合并|母公司)?[^，。\n]{0,20}' + re.escape(base_title) + r'[^，。\n]{0,30}',
            r'[^，。\n]{0,20}' + re.escape(base_title) + r'[^，。\n]{0,30}'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                title = match.group(0).strip()
                # 清理多余字符
                title = re.sub(r'\s+', '', title)
                if len(title) > 5:
                    return title

        return base_title

    def _extract_fundraising_title(self, text: str) -> str:
        """提取募资标题"""
        # 尝试提取包含"募集资金"的完整标题
        patterns = [
            r'募集资金[^，。\n]{0,30}',
            r'[^，。\n]{0,20}募集资金[^，。\n]{0,20}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                title = match.group(0).strip()
                title = re.sub(r'\s+', '', title)
                if len(title) > 3:
                    return title

        return "募集资金相关"

    def extract_title_from_context(
        self,
        text_chunks: List[Dict[str, Any]],
        table_page: int,
        table_index: int
    ) -> Optional[str]:
        """从文本块中提取表格标题

        策略：查找表格所在页或上一页的文本，
        找到表格前最近的短文本行作为标题

        Args:
            text_chunks: 文档的文本块列表
            table_page: 表格所在页码
            table_index: 表格在该页的索引（用于区分同页多个表格）

        Returns:
            提取的标题（如果找到）
        """
        # 找到相关页面的文本
        relevant_texts = []
        for chunk in text_chunks:
            if chunk.get('page') in [table_page - 1, table_page]:
                relevant_texts.append(chunk.get('text', ''))

        if not relevant_texts:
            return None

        # 合并文本并按行分割
        full_text = "\n".join(relevant_texts)
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]

        # 查找可能的标题（短行，包含关键词）
        title_candidates = []
        for i, line in enumerate(lines):
            # 标题特征：
            # 1. 长度适中（5-40字符）
            # 2. 包含表格关键词
            # 3. 不全是数字
            if 5 <= len(line) <= 40:
                has_keyword = any(kw in line for kw in
                                 self.FINANCIAL_KEYWORDS + self.FUNDRAISING_KEYWORDS)
                not_all_numbers = not re.match(r'^[\d\s.,%\-]+$', line)

                if has_keyword and not_all_numbers:
                    title_candidates.append((i, line))

        # 返回最后一个候选（最接近表格的）
        if title_candidates:
            return title_candidates[-1][1]

        return None


def classify_and_title_tables(
    tables: List[Dict[str, Any]],
    text_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """批量分类和提取标题

    Args:
        tables: 表格列表
        text_chunks: 文本块列表

    Returns:
        增强后的表格列表（添加了分类和标题信息）
    """
    classifier = TableClassifier()
    enhanced_tables = []

    for i, table in enumerate(tables):
        # 转换为DataFrame
        df = pd.DataFrame(table.get('data', {}))
        page = table.get('page', 0)

        # 从上下文提取标题
        context_title = classifier.extract_title_from_context(
            text_chunks, page, i
        )

        # 分类表格
        classification = classifier.classify_table(
            df,
            context_text=context_title or ""
        )

        # 决定最终标题
        final_title = context_title or classification['suggested_title'] or f"表格_{i+1}"

        # 增强表格信息
        enhanced_table = table.copy()
        enhanced_table.update({
            'title': final_title,
            'classification': classification,
            'context_title': context_title
        })

        enhanced_tables.append(enhanced_table)

    return enhanced_tables
