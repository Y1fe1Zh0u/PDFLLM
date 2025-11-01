"""批量测试财报提取脚本"""
import sys
import json
from pathlib import Path
from datetime import datetime
import traceback

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_extractor import PDFExtractor


def test_single_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """测试单个PDF文件

    Returns:
        包含提取结果和统计信息的字典
    """
    print(f"\n{'='*60}")
    print(f"正在处理: {pdf_path.name}")
    print(f"{'='*60}")

    result = {
        "file_name": pdf_path.name,
        "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "status": "pending",
        "error": None,
        "stats": {},
    }

    try:
        # 提取
        extractor = PDFExtractor(str(pdf_path))
        data = extractor.extract_all()

        # 统计信息
        text_chunks = data.get("text_chunks", [])
        tables = data.get("tables", [])

        result["stats"] = {
            "总页数": len(text_chunks),
            "文本块数": len(text_chunks),
            "表格数": len(tables),
            "平均每页文本长度": sum(len(c["text"]) for c in text_chunks) / len(text_chunks) if text_chunks else 0,
        }

        # 表格详情
        result["stats"]["表格详情"] = []
        for i, table in enumerate(tables, 1):
            table_info = {
                "编号": i,
                "页码": table.get("page"),
                "行数": len(table.get("dataframe", [])) if table.get("dataframe") else 0,
                "准确度": table.get("accuracy", "N/A"),
            }
            result["stats"]["表格详情"].append(table_info)
            print(f"  表格 {i}: 第{table.get('page')}页, 行数={table_info['行数']}, 准确度={table_info['准确度']}")

        # 保存提取结果
        output_file = output_dir / f"{pdf_path.stem}_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 将 DataFrame 转换为可序列化格式
            serializable_data = {
                "document_id": data["document_id"],
                "source_path": data["source_path"],
                "text_chunks": data["text_chunks"],
                "tables": [
                    {
                        "table_id": t.get("table_id"),
                        "page": t.get("page"),
                        "type": t.get("type"),
                        "accuracy": t.get("accuracy"),
                        "data": t["dataframe"].to_dict() if hasattr(t.get("dataframe"), "to_dict") else t.get("dataframe"),
                    }
                    for t in tables
                ]
            }
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        result["status"] = "success"
        result["output_file"] = str(output_file)

        print(f"✅ 成功! 提取了 {len(tables)} 个表格")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"❌ 失败: {e}")

    return result


def main():
    """主函数：批量测试所有PDF"""
    # 设置路径
    upload_dir = Path("data/uploads")
    output_dir = Path("data/outputs/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有PDF文件
    pdf_files = list(upload_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"\n⚠️  在 {upload_dir} 中没有找到PDF文件!")
        print(f"\n请按照以下步骤操作:")
        print(f"1. 将你的10份财报PDF文件复制到: {upload_dir.absolute()}")
        print(f"2. 重新运行此脚本: python scripts/test_extraction.py")
        return

    print(f"\n找到 {len(pdf_files)} 份PDF文件")
    print(f"输出目录: {output_dir.absolute()}\n")

    # 批量处理
    all_results = []
    start_time = datetime.now()

    for pdf_file in pdf_files:
        result = test_single_pdf(pdf_file, output_dir)
        all_results.append(result)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 生成汇总报告
    print(f"\n{'='*60}")
    print("测试汇总报告")
    print(f"{'='*60}")

    success_count = sum(1 for r in all_results if r["status"] == "success")
    failed_count = len(all_results) - success_count

    total_tables = sum(r["stats"].get("表格数", 0) for r in all_results if r["status"] == "success")
    total_pages = sum(r["stats"].get("总页数", 0) for r in all_results if r["status"] == "success")

    summary = {
        "测试时间": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "总耗时(秒)": round(duration, 2),
        "处理文件数": len(all_results),
        "成功": success_count,
        "失败": failed_count,
        "成功率": f"{success_count/len(all_results)*100:.1f}%" if all_results else "0%",
        "总页数": total_pages,
        "总表格数": total_tables,
        "平均每份财报表格数": round(total_tables / success_count, 1) if success_count else 0,
        "详细结果": all_results,
    }

    print(f"✅ 成功: {success_count}/{len(all_results)}")
    print(f"❌ 失败: {failed_count}/{len(all_results)}")
    print(f"📊 总表格数: {total_tables}")
    print(f"📄 总页数: {total_pages}")
    print(f"⏱️  总耗时: {duration:.1f}秒")

    # 失败案例
    if failed_count > 0:
        print(f"\n失败的文件:")
        for r in all_results:
            if r["status"] == "failed":
                print(f"  - {r['file_name']}: {r['error']}")

    # 保存汇总报告
    report_file = output_dir / f"test_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📝 详细报告已保存至: {report_file}")

    # 打印下一步建议
    print(f"\n{'='*60}")
    print("下一步操作建议:")
    print(f"{'='*60}")
    print("1. 查看提取结果: ")
    print(f"   cd {output_dir}")
    print(f"   ls -lh")
    print("2. 检查提取的表格数据:")
    print(f"   cat {output_dir}/<文件名>_extracted.json")
    print("3. 如果表格数量偏少，考虑:")
    print("   - 增加 Camelot 提取器（处理有边框表格）")
    print("   - 调整提取参数")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
