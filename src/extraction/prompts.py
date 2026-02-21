"""提示词模板

每个事实字段对应一个提取prompt + 检索query。
每个字段同时输出总结和原文引用（_quote后缀）。
多轮fallback检索使用子字段组（SubFieldGroup）拆分query。
"""

from dataclasses import dataclass, field

# --- 交易概要 ---

DEAL_SUMMARY_QUERY = "交易方案 交易金额 发行价格 支付方式 标的估值 交易对价 收购价款"

DEAL_SUMMARY_SYSTEM = """你是一个专业的并购报告分析师。请从给定的文本片段中提取交易概要信息。

要求：
1. 只从提供的文本中提取，不要编造信息
2. 如果某个字段在文本中找不到，留空字符串
3. 每个关键字段同时提供：
   - 提炼后的简要结论
   - 对应的原文引用（_quote后缀），必须是从原文中逐字复制的句子或段落
4. 以JSON格式返回，包含以下字段：
   - acquirer: 收购方/上市公司名称（简称即可）
   - acquirer_quote: 原文中提到收购方的句子
   - target: 标的方/被收购方名称（写完整名称）
   - target_quote: 原文中提到标的方的句子
   - deal_type: 交易类型（如：发行股份购买资产、现金收购等）
   - deal_amount: 交易总金额
   - deal_amount_quote: 原文中提到金额的句子
   - share_price: 发行价格/每股价格
   - payment_method: 支付方式
   - target_valuation: 标的资产估值/评估值
   - target_valuation_quote: 原文中提到估值的句子
   - valuation_method: 估值方法
   - performance_commitment: 业绩承诺/补偿安排的要点
   - performance_commitment_quote: 原文中关于业绩承诺的句子"""

DEAL_SUMMARY_USER = """请从以下并购报告文本片段中提取交易概要信息：

{chunks_text}

请以JSON格式返回，每个关键字段都要附带原文引用（_quote后缀）。"""


# --- 并购目的 ---

ACQUISITION_PURPOSE_QUERY = "并购目的 战略意义 协同效应 交易原因 收购理由 交易必要性"

ACQUISITION_PURPOSE_SYSTEM = """你是一个专业的并购报告分析师。请从给定的文本片段中提取并购目的信息。

要求：
1. 只从提供的文本中提取，不要编造信息
2. 如果某个字段在文本中找不到，留空字符串
3. 每个字段同时提供：
   - 提炼后的简要总结（1-2句话）
   - 对应的原文引用（_quote后缀），必须是从原文中逐字复制的关键段落
4. 以JSON格式返回，包含以下字段：
   - strategic_purpose: 战略目的（公司层面的战略意图，1-2句话总结）
   - strategic_purpose_quote: 原文中关于战略目的的段落
   - synergy: 协同效应（1-2句话总结）
   - synergy_quote: 原文中关于协同效应的段落
   - industry_logic: 行业逻辑（1-2句话总结）
   - industry_logic_quote: 原文中关于行业背景的段落
   - summary: 用一句话总结本次并购的核心目的"""

ACQUISITION_PURPOSE_USER = """请从以下并购报告文本片段中提取并购目的信息：

{chunks_text}

请以JSON格式返回，每个字段都要附带原文引用（_quote后缀）。"""


# 所有字段配置
FIELD_CONFIGS = {
    "deal_summary": {
        "query": DEAL_SUMMARY_QUERY,
        "system_prompt": DEAL_SUMMARY_SYSTEM,
        "user_prompt_template": DEAL_SUMMARY_USER,
    },
    "acquisition_purpose": {
        "query": ACQUISITION_PURPOSE_QUERY,
        "system_prompt": ACQUISITION_PURPOSE_SYSTEM,
        "user_prompt_template": ACQUISITION_PURPOSE_USER,
    },
}


# --- 多轮fallback子字段组 ---

@dataclass
class SubFieldGroup:
    """子字段组：多轮fallback检索的最小检索单元"""

    name: str                                   # 组名
    query: str                                  # 检索query
    target_fields: list[str] = field(default_factory=list)  # 覆盖的主字段名（不含_quote）
    section_keywords: list[str] = field(default_factory=list)  # Round 3 section路由关键词
    hyde_query: str = ""                        # Round 2 HyDE提问


DEAL_SUMMARY_SUBGROUPS = [
    SubFieldGroup(
        name="parties",
        query="收购方 标的方 上市公司 被收购方",
        target_fields=["acquirer", "target", "deal_type"],
        section_keywords=["交易概述", "交易方案", "重大资产重组"],
        hyde_query="本次交易的收购方是哪家上市公司？标的方是哪家公司？交易类型是什么？",
    ),
    SubFieldGroup(
        name="pricing",
        query="交易金额 交易对价 发行价格 支付方式",
        target_fields=["deal_amount", "share_price", "payment_method"],
        section_keywords=["交易方案", "定价", "发行股份"],
        hyde_query="本次交易的总金额是多少？发行价格是多少？采用什么支付方式？",
    ),
    SubFieldGroup(
        name="valuation",
        query="估值方法 收益法 资产基础法 评估方法 评估值",
        target_fields=["target_valuation", "valuation_method"],
        section_keywords=["评估", "估值", "定价"],
        hyde_query="标的资产采用什么估值方法进行评估？评估值是多少？",
    ),
    SubFieldGroup(
        name="commitment",
        query="业绩承诺 盈利预测补偿 承诺期",
        target_fields=["performance_commitment"],
        section_keywords=["承诺", "补偿", "盈利预测"],
        hyde_query="交易对方做出了哪些业绩承诺？补偿安排是什么？",
    ),
]

ACQUISITION_PURPOSE_SUBGROUPS = [
    SubFieldGroup(
        name="strategy",
        query="战略目的 战略意义 公司发展战略 转型",
        target_fields=["strategic_purpose"],
        section_keywords=["目的", "必要性", "战略"],
        hyde_query="本次并购的战略目的是什么？对公司发展有什么战略意义？",
    ),
    SubFieldGroup(
        name="synergy",
        query="协同效应 业务整合 资源互补 规模效应",
        target_fields=["synergy"],
        section_keywords=["协同", "整合", "效应"],
        hyde_query="本次并购能产生哪些协同效应？如何实现业务整合？",
    ),
    SubFieldGroup(
        name="industry",
        query="行业逻辑 行业趋势 市场机遇 行业背景",
        target_fields=["industry_logic", "summary"],
        section_keywords=["行业", "市场", "背景"],
        hyde_query="本次并购的行业背景是什么？行业发展趋势如何？",
    ),
]

# 字段名 → 子字段组列表
FIELD_SUBGROUPS = {
    "deal_summary": DEAL_SUMMARY_SUBGROUPS,
    "acquisition_purpose": ACQUISITION_PURPOSE_SUBGROUPS,
}


# --- HyDE prompt模板 ---

HYDE_SYSTEM = """你是一个并购报告分析师。请根据问题，假设你正在阅读一份A股并购报告，\
写出一段可能出现在报告中的文字来回答这个问题。
要求：
1. 用中文书面语，模拟报告原文风格
2. 包含具体的数字、名称等细节（可以是假设的）
3. 控制在100-200字"""

HYDE_USER = "{hyde_query}"


# --- 定向抽取prompt模板（只抽取缺失字段）---

TARGETED_EXTRACT_SYSTEM = """你是一个专业的并购报告分析师。请从给定的文本片段中提取以下指定字段的信息。

要求：
1. 只从提供的文本中提取，不要编造信息
2. 如果某个字段在文本中找不到，留空字符串
3. 每个关键字段同时提供原文引用（_quote后缀），必须是从原文中逐字复制的句子
4. 以JSON格式返回，只包含以下字段：
{field_descriptions}"""

TARGETED_EXTRACT_USER = """请从以下并购报告文本片段中提取指定字段：

{chunks_text}

请以JSON格式返回，只包含指定字段。"""

# 字段描述映射（用于定向抽取prompt）
FIELD_DESCRIPTIONS = {
    # deal_summary字段
    "acquirer": "acquirer: 收购方/上市公司名称\n   acquirer_quote: 原文引用",
    "target": "target: 标的方/被收购方名称\n   target_quote: 原文引用",
    "deal_type": "deal_type: 交易类型（如：发行股份购买资产）",
    "deal_amount": "deal_amount: 交易总金额\n   deal_amount_quote: 原文引用",
    "share_price": "share_price: 发行价格/每股价格",
    "payment_method": "payment_method: 支付方式",
    "target_valuation": "target_valuation: 标的资产估值/评估值\n   target_valuation_quote: 原文引用",
    "valuation_method": "valuation_method: 估值方法",
    "performance_commitment": "performance_commitment: 业绩承诺/补偿安排\n   performance_commitment_quote: 原文引用",
    # acquisition_purpose字段
    "strategic_purpose": "strategic_purpose: 战略目的（1-2句话）\n   strategic_purpose_quote: 原文引用",
    "synergy": "synergy: 协同效应（1-2句话）\n   synergy_quote: 原文引用",
    "industry_logic": "industry_logic: 行业逻辑（1-2句话）\n   industry_logic_quote: 原文引用",
    "summary": "summary: 用一句话总结本次并购的核心目的",
}
