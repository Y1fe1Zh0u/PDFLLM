"""核心数据模型

所有模块共用的Pydantic数据结构定义。
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# --- 切片相关 ---

class Chunk(BaseModel):
    """文档切片"""

    doc_id: str = Field(description="文档唯一标识（文件名去后缀）")
    chunk_id: int = Field(description="切片在文档内的序号")
    text: str = Field(description="切片文本内容")
    page: int = Field(description="起始页码（从1开始）")
    section: str = Field(default="", description="所属章节标题")
    metadata: dict = Field(default_factory=dict, description="额外元数据")


# --- 事实抽取相关 ---

class ExtractionStatus(str, Enum):
    """抽取状态"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class DealSummary(BaseModel):
    """交易概要"""

    acquirer: str = Field(default="", description="收购方")
    acquirer_quote: str = Field(default="", description="收购方原文引用")
    target: str = Field(default="", description="标的方/被收购方")
    target_quote: str = Field(default="", description="标的方原文引用")
    deal_type: str = Field(default="", description="交易类型（如：发行股份购买资产）")
    deal_amount: str = Field(default="", description="交易金额")
    deal_amount_quote: str = Field(default="", description="交易金额原文引用")
    share_price: str = Field(default="", description="发行价格")
    payment_method: str = Field(default="", description="支付方式")
    target_valuation: str = Field(default="", description="标的估值")
    target_valuation_quote: str = Field(default="", description="标的估值原文引用")
    valuation_method: str = Field(default="", description="估值方法")
    performance_commitment: str = Field(default="", description="业绩承诺")
    performance_commitment_quote: str = Field(default="", description="业绩承诺原文引用")


class AcquisitionPurpose(BaseModel):
    """并购目的"""

    strategic_purpose: str = Field(default="", description="战略目的")
    strategic_purpose_quote: str = Field(default="", description="战略目的原文引用")
    synergy: str = Field(default="", description="协同效应")
    synergy_quote: str = Field(default="", description="协同效应原文引用")
    industry_logic: str = Field(default="", description="行业逻辑")
    industry_logic_quote: str = Field(default="", description="行业逻辑原文引用")
    summary: str = Field(default="", description="一句话总结")


class FactRecord(BaseModel):
    """完整事实档案"""

    doc_id: str = Field(description="文档标识")
    company_name: str = Field(default="", description="上市公司名称")
    stock_code: str = Field(default="", description="股票代码")
    deal_summary: DealSummary = Field(default_factory=DealSummary)
    acquisition_purpose: AcquisitionPurpose = Field(default_factory=AcquisitionPurpose)
    status: ExtractionStatus = Field(default=ExtractionStatus.SUCCESS)
    raw_responses: dict = Field(default_factory=dict, description="LLM原始返回（用于调试）")


class ExtractionResult(BaseModel):
    """单次LLM抽取的结果"""

    field_name: str = Field(description="抽取的字段名（如deal_summary）")
    doc_id: str
    status: ExtractionStatus
    data: Optional[dict] = Field(default=None, description="解析后的结构化数据")
    raw_response: str = Field(default="", description="LLM原始返回文本")
    source_chunks: list[str] = Field(default_factory=list, description="输入chunks来源标识")
    error: str = Field(default="", description="错误信息")
