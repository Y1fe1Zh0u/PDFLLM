"""提示词模板

每个事实字段对应一个提取prompt + 检索query。
每个字段同时输出总结和原文引用（_quote后缀）。
"""

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
