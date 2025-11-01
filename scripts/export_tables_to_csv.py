"""将提取的表格导出为CSV文件

功能：
1. 读取JSON格式的提取结果
2. 将每个表格导出为独立的CSV文件
3. 生成表格索引文件（包含表格元数据）
4. 可选：将同一PDF的所有表格导出到Excel的多个sheet
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入配置和日志
from src.utils.config import settings
from src.utils.logger import setup_script_logger
from src.ingestion.table_classifier import TableClassifier

# 设置日志
logger = setup_script_logger("export")


def extract_company_name(document_id: str) -> str:
    """从文档ID中提取公司简称

    Args:
        document_id: 文档ID

    Returns:
        公司简称
    """
    # 提取股票代码和公司名称（通常在文档ID开头）
    # 例如: "000035中国天楹..." -> "000035-中国天楹"
    import re
    match = re.match(r'(\d+)([^发行购买]+)', document_id)
    if match:
        code, name = match.groups()
        # 截取公司名称前几个字
        name = name[:10] if len(name) > 10 else name
        return f"{code}-{name}"
    return document_id[:20]


def find_table_title(text_chunks: list, table_page: int) -> str:
    """从文本块中查找表格标题

    Args:
        text_chunks: 文本块列表
        table_page: 表格所在页码

    Returns:
        表格标题（如果找到），否则返回空字符串
    """
    # 在表格所在页查找可能的标题
    for chunk in text_chunks:
        if chunk.get('page') == table_page:
            text = chunk.get('text', '')
            lines = text.split('\n')

            # 查找可能的标题特征：
            # 1. 短行（少于30字符）
            # 2. 包含"表"、"一览"、"明细"、"情况"等关键词
            # 3. 或者是"释义"、"声明"等特殊标题
            title_keywords = ['表', '一览', '明细', '情况', '列表', '清单', '汇总', '统计',
                            '释义', '声明', '说明', '概况', '信息', '数据', '资料']

            # 排除的页眉特征
            exclude_keywords = ['公司', '股份有限公司', '报告书', '公告书', '摘要']

            candidates = []
            for line in lines:
                line = line.strip()
                # 跳过过长或过短的行
                if 2 <= len(line) <= 30:
                    # 检查是否包含标题关键词
                    has_title_keyword = any(kw in line for kw in title_keywords)
                    # 检查是否是页眉
                    is_header = all(kw in line for kw in exclude_keywords[:2])

                    if has_title_keyword and not is_header:
                        # 清理标题
                        line = line.replace('\n', '').replace('\r', '').strip()
                        candidates.append(line)

            # 返回第一个候选标题
            if candidates:
                return candidates[0]

            # 如果没找到合适的，返回空（后续用默认命名）
            return ""

    return ""


def generate_smart_filename(
    table_info: dict,
    classification: dict,
    context_title: str,
    page: int,
    company_code: str,
    existing_names: set
) -> str:
    """生成智能文件名

    Args:
        table_info: 表格信息
        classification: 分类结果
        context_title: 从上下文提取的标题
        page: 页码
        company_code: 公司代码
        existing_names: 已存在的文件名集合（用于处理重名）

    Returns:
        清理后的文件名
    """
    # 策略1: 优先使用上下文标题
    if context_title and len(context_title) > 3:
        base_name = context_title
    # 策略2: 使用分类结果生成标题
    elif classification.get('type') != 'other':
        category_map = {
            'balance_sheet': '资产负债表',
            'income_statement': '利润表',
            'cash_flow_statement': '现金流量表',
            'equity_statement': '所有者权益变动表',
            'usage': '募集资金使用情况',
            'source': '募集资金来源',
            'issuance': '发行方案'
        }
        base_name = category_map.get(
            classification.get('category'),
            classification.get('suggested_title', '')
        )
    # 策略3: 使用table_id
    else:
        base_name = table_info.get('table_id', 'table')

    # 清理文件名
    base_name = sanitize_filename(base_name)

    # 限制长度
    if len(base_name) > 50:
        base_name = base_name[:50]

    # 生成完整文件名
    filename = f"{base_name}_page{page}_{company_code}.csv"

    # 处理重名
    if filename in existing_names:
        counter = 2
        while f"{base_name}_{counter}_page{page}_{company_code}.csv" in existing_names:
            counter += 1
        filename = f"{base_name}_{counter}_page{page}_{company_code}.csv"

    return filename


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符

    Args:
        filename: 原始文件名

    Returns:
        清理后的文件名
    """
    # 移除文件系统不允许的字符
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']
    for char in illegal_chars:
        filename = filename.replace(char, '')

    # 限制长度
    if len(filename) > 80:
        filename = filename[:80]

    return filename.strip()


def export_tables_from_json(json_file: Path, output_dir: Path, export_format: str = "csv"):
    """从JSON文件导出表格

    Args:
        json_file: JSON文件路径
        output_dir: 输出目录
        export_format: 导出格式 ("csv" 或 "excel")

    Returns:
        导出的表格数量
    """
    # 读取JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    document_id = data.get("document_id", "unknown")
    tables = data.get("tables", [])
    text_chunks = data.get("text_chunks", [])

    if not tables:
        logger.warning(f"{json_file.name}: 没有表格数据")
        return 0

    # 提取公司代码
    company_code = extract_company_name(document_id)

    # 创建文档专属目录
    doc_dir = output_dir / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # 初始化分类器
    classifier = TableClassifier()

    # 表格索引信息
    table_index = []
    existing_names = set()  # 用于处理重名

    # 导出每个表格
    exported_count = 0
    for i, table in enumerate(tables, 1):
        try:
            table_id = table.get("table_id", f"table_{i}")
            page = table.get("page", "unknown")
            accuracy = table.get("accuracy", "N/A")
            table_data = table.get("data", {})

            if not table_data:
                logger.warning(f"表格 {table_id} (页{page}): 数据为空")
                continue

            # 转换为DataFrame
            df = pd.DataFrame(table_data)

            # 检查是否为空
            if df.empty:
                logger.warning(f"表格 {table_id} (页{page}): DataFrame为空")
                continue

            # 清理单元格内的换行符（根据配置决定是否清理）
            if settings.export_clean_newlines:
                df = df.map(lambda x: str(x).replace('\n', '').replace('\r', '').strip() if pd.notna(x) else x)

            # 表格分类和标题提取
            context_title = classifier.extract_title_from_context(text_chunks, page, i)
            classification = classifier.classify_table(df, context_text=context_title or "")

            # 生成智能文件名
            filename = generate_smart_filename(
                table_info={'table_id': table_id},
                classification=classification,
                context_title=context_title,
                page=page,
                company_code=company_code,
                existing_names=existing_names
            )
            existing_names.add(filename)

            # 导出CSV
            if export_format == "csv":
                csv_file = doc_dir / filename
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                exported_count += 1

                # 记录索引信息
                table_index.append({
                    "表格ID": table_id,
                    "标题": context_title or classification.get('suggested_title', ''),
                    "类型": classification.get('type', 'unknown'),
                    "类别": classification.get('category', 'unknown'),
                    "置信度": f"{classification.get('confidence', 0):.2f}",
                    "页码": page,
                    "准确度": accuracy,
                    "行数": len(df),
                    "列数": len(df.columns),
                    "文件名": filename
                })

                logger.debug(f"导出: {filename} (类型: {classification.get('type')})")

        except Exception as e:
            logger.error(f"表格 {table_id} 导出失败: {e}")
            continue

    # 导出表格索引
    if table_index:
        index_df = pd.DataFrame(table_index)
        index_file = doc_dir / "_table_index.csv"
        index_df.to_csv(index_file, index=False, encoding='utf-8-sig')
        logger.info(f"{document_id}: 导出 {exported_count}/{len(tables)} 个表格")

    return exported_count


def main():
    """主函数"""
    # 使用配置中的路径
    results_dir = Path(settings.output_dir) / "test_results"
    export_dir = Path(settings.output_dir) / "tables_csv"
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始导出表格，输入目录: {results_dir}, 输出目录: {export_dir}")

    print(f"\n{'='*60}")
    print(f"表格CSV导出工具")
    print(f"{'='*60}")
    print(f"📂 输入目录: {results_dir}")
    print(f"📂 输出目录: {export_dir}")
    print(f"{'='*60}\n")

    # 查找所有JSON文件
    json_files = list(results_dir.glob("*_extracted.json"))

    if not json_files:
        logger.warning(f"在 {results_dir} 中未找到提取结果文件")
        print(f"❌ 在 {results_dir} 中未找到提取结果文件")
        return

    logger.info(f"找到 {len(json_files)} 个JSON文件")

    print(f"📁 找到 {len(json_files)} 个JSON文件\n")

    # 统计信息
    total_tables = 0
    total_exported = 0
    start_time = datetime.now()

    # 处理每个文件
    for json_file in json_files:
        try:
            # 读取表格数量
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                table_count = len(data.get("tables", []))
                total_tables += table_count

            # 导出表格
            exported = export_tables_from_json(json_file, export_dir)
            total_exported += exported

        except Exception as e:
            logger.error(f"{json_file.name}: 处理失败 - {e}")

    # 生成汇总报告
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"导出完成，总表格数: {total_tables}, 成功导出: {total_exported}, 耗时: {duration:.1f}秒")

    print(f"\n{'='*60}")
    print(f"导出完成")
    print(f"{'='*60}")
    print(f"📊 总表格数: {total_tables}")
    print(f"✅ 成功导出: {total_exported}")
    print(f"❌ 失败/跳过: {total_tables - total_exported}")
    print(f"⏱️  耗时: {duration:.1f}秒")
    print(f"📂 输出位置: {export_dir.absolute()}")
    print(f"{'='*60}\n")

    # 生成全局索引
    print("📝 生成全局表格索引...")
    all_indexes = []
    for doc_dir in export_dir.iterdir():
        if doc_dir.is_dir():
            index_file = doc_dir / "_table_index.csv"
            if index_file.exists():
                df = pd.read_csv(index_file, encoding='utf-8-sig')
                df['文档ID'] = doc_dir.name
                all_indexes.append(df)

    if all_indexes:
        global_index = pd.concat(all_indexes, ignore_index=True)
        global_index_file = export_dir / "全局表格索引.csv"
        global_index.to_csv(global_index_file, index=False, encoding='utf-8-sig')
        print(f"✅ 全局索引已保存: {global_index_file}")

        # 显示统计
        print(f"\n📈 表格统计:")
        print(f"  总文档数: {global_index['文档ID'].nunique()}")
        print(f"  总表格数: {len(global_index)}")
        print(f"  平均准确度: {global_index['准确度'].mean():.2f}%")
        print(f"  平均行数: {global_index['行数'].mean():.1f}")
        print(f"  平均列数: {global_index['列数'].mean():.1f}")


if __name__ == "__main__":
    main()
