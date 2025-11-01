"""并发批量测试财报提取脚本 - 优化版"""
import sys
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入统一的环境设置和配置
from src.utils.env_setup import setup_ghostscript_env, ensure_env_in_subprocess
from src.utils.config import settings
from src.utils.logger import setup_script_logger
from src.ingestion.pdf_extractor import PDFExtractor

# 设置环境（主进程）
setup_ghostscript_env()

# 设置日志
logger = setup_script_logger("extraction")


def test_single_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """测试单个PDF文件（用于并发执行）

    Returns:
        包含提取结果和统计信息的字典
    """
    # 在子进程中设置环境变量（关键！）
    ensure_env_in_subprocess()

    result = {
        "file_name": pdf_path.name,
        "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "status": "pending",
        "error": None,
        "stats": {},
        "start_time": None,
        "end_time": None,
    }

    try:
        start = datetime.now()
        result["start_time"] = start.strftime("%H:%M:%S")

        logger.info(f"开始处理: {pdf_path.name}")

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
                "行数": len(table.get("dataframe", [])) if table.get("dataframe") is not None and hasattr(table.get("dataframe"), "__len__") else 0,
                "准确度": table.get("accuracy", "N/A"),
            }
            result["stats"]["表格详情"].append(table_info)

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

        end = datetime.now()
        result["end_time"] = end.strftime("%H:%M:%S")
        duration = (end - start).total_seconds()

        logger.info(f"✅ {pdf_path.name} - 成功! 提取了 {len(tables)} 个表格 (耗时 {duration:.1f}秒)")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["end_time"] = datetime.now().strftime("%H:%M:%S")
        logger.error(f"❌ {pdf_path.name} - 失败: {e}")

    return result


def process_pdf_wrapper(args):
    """包装函数用于进程池"""
    pdf_path, output_dir = args
    return test_single_pdf(pdf_path, output_dir)


def main():
    """主函数：并发批量测试所有PDF"""
    # 使用配置中的路径
    upload_dir = Path(settings.upload_dir)
    output_dir = Path(settings.output_dir) / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有PDF文件
    pdf_files = list(upload_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"在 {upload_dir} 中没有找到PDF文件!")
        print(f"\n⚠️  在 {upload_dir} 中没有找到PDF文件!")
        print(f"\n请按照以下步骤操作:")
        print(f"1. 将你的财报PDF文件复制到: {upload_dir.absolute()}")
        print(f"2. 重新运行此脚本: python scripts/test_extraction_parallel.py")
        return

    # 使用配置中的并发数
    max_workers = min(cpu_count(), len(pdf_files), settings.max_workers)

    logger.info(f"找到 {len(pdf_files)} 份PDF文件，使用 {max_workers} 个worker并发处理")

    print(f"\n{'='*60}")
    print(f"并发提取测试")
    print(f"{'='*60}")
    print(f"📁 找到 {len(pdf_files)} 份PDF文件")
    print(f"🚀 并发数: {max_workers} (CPU核心数: {cpu_count()})")
    print(f"📂 输出目录: {output_dir.absolute()}")
    print(f"{'='*60}\n")

    # 批量处理 - 并发执行
    all_results = []
    start_time = datetime.now()

    # 准备参数
    tasks = [(pdf_file, output_dir) for pdf_file in pdf_files]

    # 使用进程池并发执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_pdf_wrapper, task): task[0] for task in tasks}

        # 收集结果（按完成顺序）
        for future in as_completed(futures):
            result = future.result()
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

    # 按文件名排序结果
    all_results.sort(key=lambda x: x["file_name"])

    summary = {
        "测试时间": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "总耗时(秒)": round(duration, 2),
        "并发数": max_workers,
        "处理文件数": len(all_results),
        "成功": success_count,
        "失败": failed_count,
        "成功率": f"{success_count/len(all_results)*100:.1f}%" if all_results else "0%",
        "总页数": total_pages,
        "总表格数": total_tables,
        "平均每份财报表格数": round(total_tables / success_count, 1) if success_count else 0,
        "平均处理速度(秒/份)": round(duration / len(all_results), 2) if all_results else 0,
        "详细结果": all_results,
    }

    print(f"✅ 成功: {success_count}/{len(all_results)} ({summary['成功率']})")
    print(f"❌ 失败: {failed_count}/{len(all_results)}")
    print(f"📊 总表格数: {total_tables}")
    print(f"📄 总页数: {total_pages}")
    print(f"⏱️  总耗时: {duration:.1f}秒")
    print(f"⚡ 平均速度: {summary['平均处理速度(秒/份)']}秒/份")

    # 详细结果表格
    print(f"\n详细结果:")
    print(f"{'文件名':<50} {'状态':<10} {'页数':<8} {'表格数':<8}")
    print("-" * 80)
    for r in all_results:
        status = "✅ 成功" if r["status"] == "success" else "❌ 失败"
        pages = r["stats"].get("总页数", 0)
        tables = r["stats"].get("表格数", 0)
        file_name = r["file_name"][:47] + "..." if len(r["file_name"]) > 50 else r["file_name"]
        print(f"{file_name:<50} {status:<10} {pages:<8} {tables:<8}")

    # 失败案例
    if failed_count > 0:
        print(f"\n失败的文件详情:")
        for r in all_results:
            if r["status"] == "failed":
                print(f"  ❌ {r['file_name']}")
                print(f"     错误: {r['error']}")

    # 保存汇总报告
    report_file = output_dir / f"test_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📝 详细报告已保存至: {report_file}")

    # 打印下一步建议
    print(f"\n{'='*60}")
    print("下一步操作建议:")
    print(f"{'='*60}")
    if success_count > 0:
        print("1. 查看提取结果:")
        print(f"   cd {output_dir}")
        print(f"   cat <文件名>_extracted.json | head -100")
        print("2. 分析表格质量:")
        avg_tables = total_tables / success_count if success_count else 0
        if avg_tables < 5:
            print("   ⚠️  平均每份财报表格数较少，建议:")
            print("   - 检查PDF是否为扫描件")
            print("   - 考虑增加 Camelot 提取器")
        elif avg_tables >= 5:
            print("   ✅ 表格提取数量正常")
        print("3. 下一步开发:")
        print("   - 索引模块: 对提取的数据建立向量索引")
        print("   - 检索模块: 实现语义检索")
    else:
        print("⚠️  所有文件提取失败，请检查:")
        print("1. PDF文件是否损坏或加密")
        print("2. 依赖是否正确安装")
        print("3. 查看错误日志")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
