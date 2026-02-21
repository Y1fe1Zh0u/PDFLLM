"""SQLite事实数据库

轻量封装sqlite3，存储和查询FactRecord。
"""

import json
import sqlite3
from pathlib import Path

from src.infra.config import settings
from src.infra.logger import get_logger
from src.infra.models import FactRecord

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    doc_id TEXT PRIMARY KEY,
    company_name TEXT,
    stock_code TEXT,
    deal_summary TEXT,
    acquisition_purpose TEXT,
    status TEXT,
    raw_responses TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


class FactDB:
    """事实数据库"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_fact(self, record: FactRecord) -> None:
        """保存或更新一条事实记录"""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO facts
                (doc_id, company_name, stock_code, deal_summary,
                 acquisition_purpose, status, raw_responses, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    record.doc_id,
                    record.company_name,
                    record.stock_code,
                    record.deal_summary.model_dump_json(),
                    record.acquisition_purpose.model_dump_json(),
                    record.status.value,
                    json.dumps(record.raw_responses, ensure_ascii=False),
                ),
            )
        logger.info(f"事实已保存: {record.doc_id}")

    def get_fact(self, doc_id: str) -> FactRecord | None:
        """查询单条事实记录"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM facts WHERE doc_id = ?", (doc_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_documents(self) -> list[dict]:
        """列出所有已处理文档的摘要"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, company_name, stock_code, status, created_at "
                "FROM facts ORDER BY created_at"
            ).fetchall()

        return [dict(row) for row in rows]

    def get_processed_doc_ids(self) -> set[str]:
        """获取所有已处理的doc_id（用于断点续跑）"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM facts WHERE status IN ('success', 'partial')"
            ).fetchall()
        return {row["doc_id"] for row in rows}

    def _row_to_record(self, row: sqlite3.Row) -> FactRecord:
        from src.infra.models import (
            AcquisitionPurpose,
            DealSummary,
            ExtractionStatus,
        )

        return FactRecord(
            doc_id=row["doc_id"],
            company_name=row["company_name"],
            stock_code=row["stock_code"],
            deal_summary=DealSummary.model_validate_json(row["deal_summary"]),
            acquisition_purpose=AcquisitionPurpose.model_validate_json(
                row["acquisition_purpose"]
            ),
            status=ExtractionStatus(row["status"]),
            raw_responses=json.loads(row["raw_responses"]) if row["raw_responses"] else {},
        )
