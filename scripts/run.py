"""CLI入口

用法:
    python scripts/run.py data/uploads/           # 批量处理目录
    python scripts/run.py data/uploads/report.pdf  # 单文件处理
    python scripts/run.py data/uploads/ --no-resume  # 不跳过已处理
    python scripts/run.py --list                   # 列出已处理文档
"""

import argparse
import json
import sys

from src.infra.db import FactDB
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="PDFLLM - 并购报告事实抽取")
    parser.add_argument("input", nargs="?", help="PDF文件或目录路径")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已处理的文档")
    parser.add_argument("--list", action="store_true", help="列出已处理的文档")
    parser.add_argument("--show", metavar="DOC_ID", help="显示指定文档的事实档案")

    args = parser.parse_args()

    if args.list:
        db = FactDB()
        docs = db.list_documents()
        if not docs:
            print("暂无已处理的文档")
            return
        for doc in docs:
            print(f"  {doc['doc_id']:40s} {doc['stock_code']:8s} {doc['company_name']:15s} [{doc['status']}]")
        print(f"\n共 {len(docs)} 个文档")
        return

    if args.show:
        db = FactDB()
        record = db.get_fact(args.show)
        if record is None:
            print(f"未找到文档: {args.show}")
            sys.exit(1)
        print(record.model_dump_json(indent=2, ensure_ascii=False))
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    stats = run_pipeline(args.input, resume=not args.no_resume)
    print(f"\n处理完成: 总计{stats['total']}, 成功{stats['success']}, 失败{stats['failed']}, 跳过{stats['skipped']}")


if __name__ == "__main__":
    main()
