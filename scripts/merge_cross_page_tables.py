"""合并跨页表格

功能：
1. 检测连续页面上的相似表格
2. 基于列数和表头相似度判断是否为跨页表格
3. 合并跨页表格并更新索引
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from difflib import SequenceMatcher

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入配置和日志
from src.utils.config import settings
from src.utils.logger import setup_script_logger

# 设置日志
logger = setup_script_logger("merge")


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个字符串的相似度（0-1）"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, str(text1), str(text2)).ratio()


def check_header_similarity(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 0.7) -> bool:
    """检查两个表格的表头相似度

    Args:
        df1: 第一个表格
        df2: 第二个表格
        threshold: 相似度阈值（0-1）

    Returns:
        是否相似
    """
    if df1.empty or df2.empty:
        return False

    # 比较第一行（通常是表头）
    header1 = ' '.join([str(x) for x in df1.iloc[0].tolist()])
    header2 = ' '.join([str(x) for x in df2.iloc[0].tolist()])

    similarity = calculate_similarity(header1, header2)
    return similarity >= threshold


def looks_like_header(row: pd.Series) -> bool:
    """判断一行是否看起来像表头

    Args:
        row: DataFrame的一行

    Returns:
        是否像表头
    """
    # 表头特征关键词
    header_keywords = [
        '名称', '编号', '序号', '项目', '内容', '金额', '数量', '单位', '日期',
        '类型', '说明', '备注', '合计', '小计', '比例', '占比', '年度', '期间',
        'Name', 'No', 'Item', 'Amount', 'Date', 'Type', 'Total', 'Ratio'
    ]

    row_str = ' '.join([str(x) for x in row.tolist()])

    # 检查是否包含表头关键词
    keyword_count = sum(1 for keyword in header_keywords if keyword in row_str)

    # 如果包含2个以上关键词，可能是表头
    if keyword_count >= 2:
        return True

    # 检查是否大部分是数字（如果是，则不像表头）
    numeric_count = 0
    for val in row.tolist():
        val_str = str(val).strip()
        # 尝试转换为数字
        try:
            float(val_str.replace(',', '').replace('%', ''))
            numeric_count += 1
        except:
            pass

    # 如果超过一半是数字，不像表头
    if numeric_count > len(row) / 2:
        return False

    return False


def should_merge_tables(table1: dict, table2: dict, similarity_threshold: float = 0.7) -> tuple:
    """判断两个表格是否应该合并

    Args:
        table1: 第一个表格信息（包含page, data等）
        table2: 第二个表格信息
        similarity_threshold: 表头相似度阈值

    Returns:
        (是否应该合并, 合并类型)
        合并类型: "header_repeat" | "data_continuation" | None
    """
    # 1. 检查页码是否连续
    if table2["page"] - table1["page"] != 1:
        return False, None

    # 2. 检查列数是否相同
    df1 = pd.DataFrame(table1["data"])
    df2 = pd.DataFrame(table2["data"])

    if len(df1.columns) != len(df2.columns):
        return False, None

    # 3. 检查表头相似度
    has_similar_headers = check_header_similarity(df1, df2, similarity_threshold)

    if has_similar_headers:
        # 模式1: 重复表头模式
        return True, "header_repeat"

    # 4. 检查是否为数据延续模式
    # 如果第二个表格的第一行不像表头，且表头不相似，则可能是数据延续
    if not df2.empty:
        first_row_df2 = df2.iloc[0]
        if not looks_like_header(first_row_df2):
            # 模式2: 数据延续模式
            return True, "data_continuation"

    return False, None


def merge_tables(tables: list, merge_type: str = "header_repeat") -> pd.DataFrame:
    """合并多个表格

    Args:
        tables: 表格列表，每个表格包含 data 字段
        merge_type: 合并类型 ("header_repeat" 或 "data_continuation")

    Returns:
        合并后的DataFrame
    """
    dfs = []
    for i, table in enumerate(tables):
        df = pd.DataFrame(table["data"])

        # 第一个表格保留完整内容
        if i == 0:
            dfs.append(df)
            continue

        if not df.empty:
            # 重复表头模式：去掉后续表格的表头（第一行）
            if merge_type == "header_repeat":
                df = df.iloc[1:]
            # 数据延续模式：保留所有行
            elif merge_type == "data_continuation":
                pass  # 保留所有行

            dfs.append(df)

    # 合并所有表格
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def process_document_tables(json_file: Path, similarity_threshold: float = 0.7,
                            dry_run: bool = False) -> dict:
    """处理单个文档的表格合并

    Args:
        json_file: JSON文件路径
        similarity_threshold: 表头相似度阈值
        dry_run: 是否为试运行（不实际合并，只输出报告）

    Returns:
        合并统计信息
    """
    # 读取JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    document_id = data.get("document_id", "unknown")
    tables = data.get("tables", [])

    if not tables:
        return {"document_id": document_id, "merged_count": 0, "groups": []}

    print(f"\n{'='*60}")
    print(f"处理文档: {document_id}")
    print(f"原始表格数: {len(tables)}")
    print(f"{'='*60}")

    # 检测需要合并的表格组
    merge_groups = []
    current_group = [tables[0]]
    current_merge_type = None

    for i in range(1, len(tables)):
        prev_table = tables[i-1]
        curr_table = tables[i]

        should_merge, merge_type = should_merge_tables(prev_table, curr_table, similarity_threshold)

        if should_merge:
            current_group.append(curr_table)
            if current_merge_type is None:
                current_merge_type = merge_type

            type_label = "重复表头" if merge_type == "header_repeat" else "数据延续"
            print(f"✓ 检测到跨页表格({type_label}): {prev_table['table_id']}(页{prev_table['page']}) + {curr_table['table_id']}(页{curr_table['page']})")
        else:
            # 当前组结束
            if len(current_group) > 1:
                merge_groups.append((current_group, current_merge_type))
            current_group = [curr_table]
            current_merge_type = None

    # 检查最后一组
    if len(current_group) > 1:
        merge_groups.append((current_group, current_merge_type))

    print(f"\n发现 {len(merge_groups)} 个需要合并的表格组")

    if dry_run:
        print("\n[试运行模式] 不执行实际合并")
        return {
            "document_id": document_id,
            "original_count": len(tables),
            "merged_groups": len(merge_groups),
            "groups": merge_groups
        }

    # 执行合并
    merged_tables = []
    merged_indices = set()

    for group, merge_type in merge_groups:
        # 合并表格
        merged_df = merge_tables(group, merge_type)

        # 记录合并信息
        start_page = group[0]["page"]
        end_page = group[-1]["page"]
        table_ids = [t["table_id"] for t in group]

        merged_table = {
            "table_id": f"{group[0]['table_id']}_merged",
            "page": f"{start_page}-{end_page}",
            "type": "table",
            "accuracy": sum(t["accuracy"] for t in group) / len(group),
            "data": merged_df.to_dict(),
            "merged_from": table_ids,
            "merged_pages": [t["page"] for t in group],
            "merge_type": merge_type
        }

        merged_tables.append(merged_table)

        # 记录已合并的表格索引
        for t in group:
            merged_indices.add(tables.index(t))

        type_label = "重复表头" if merge_type == "header_repeat" else "数据延续"
        print(f"✅ 合并({type_label}): {' + '.join(table_ids)} → {merged_table['table_id']} (页{start_page}-{end_page})")

    # 保留未合并的表格
    final_tables = []
    for i, table in enumerate(tables):
        if i not in merged_indices:
            final_tables.append(table)

    # 添加合并后的表格
    final_tables.extend(merged_tables)

    # 按页码排序
    final_tables.sort(key=lambda x: int(str(x["page"]).split('-')[0]))

    # 更新JSON
    data["tables"] = final_tables
    data["merge_info"] = {
        "original_count": len(tables),
        "merged_count": len(merged_tables),
        "final_count": len(final_tables),
        "merge_timestamp": datetime.now().isoformat()
    }

    # 保存更新后的JSON
    output_file = json_file.parent / f"{json_file.stem}_merged.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n📄 合并后表格数: {len(final_tables)}")
    print(f"💾 已保存: {output_file.name}")

    return {
        "document_id": document_id,
        "original_count": len(tables),
        "merged_groups": len(merge_groups),
        "final_count": len(final_tables)
    }


def main():
    """主函数"""
    # 使用配置中的路径和参数
    results_dir = Path(settings.output_dir) / "test_results"
    similarity_threshold = settings.table_merge_similarity_threshold
    dry_run = settings.table_merge_dry_run

    logger.info(f"开始跨页表格合并，输入目录: {results_dir}, 相似度阈值: {similarity_threshold}, 试运行: {dry_run}")

    print(f"\n{'='*60}")
    print(f"跨页表格合并工具")
    print(f"{'='*60}")
    print(f"📂 输入目录: {results_dir}")
    print(f"🎯 表头相似度阈值: {similarity_threshold}")
    print(f"🔍 模式: {'试运行' if dry_run else '实际合并'}")
    print(f"{'='*60}")

    # 查找所有JSON文件
    json_files = list(results_dir.glob("*_extracted.json"))

    if not json_files:
        logger.warning(f"在 {results_dir} 中未找到提取结果文件")
        print(f"❌ 在 {results_dir} 中未找到提取结果文件")
        return

    logger.info(f"找到 {len(json_files)} 个JSON文件")

    print(f"\n📁 找到 {len(json_files)} 个JSON文件")

    # 统计信息
    total_stats = {
        "processed_docs": 0,
        "total_original": 0,
        "total_merged_groups": 0,
        "total_final": 0
    }

    # 处理每个文件
    for json_file in json_files:
        try:
            stats = process_document_tables(json_file, similarity_threshold, dry_run)

            total_stats["processed_docs"] += 1
            total_stats["total_original"] += stats.get("original_count", 0)
            total_stats["total_merged_groups"] += stats.get("merged_groups", 0)
            total_stats["total_final"] += stats.get("final_count", 0)

        except Exception as e:
            logger.error(f"{json_file.name}: 处理失败 - {e}")

    # 生成汇总报告
    logger.info(f"合并汇总 - 处理文档数: {total_stats['processed_docs']}, 检测到跨页组: {total_stats['total_merged_groups']}")
    print(f"\n{'='*60}")
    print(f"合并汇总")
    print(f"{'='*60}")
    print(f"📊 处理文档数: {total_stats['processed_docs']}")
    print(f"📋 原始表格总数: {total_stats['total_original']}")
    print(f"🔗 检测到的跨页组: {total_stats['total_merged_groups']}")
    if not dry_run:
        print(f"✅ 合并后表格总数: {total_stats['total_final']}")
        print(f"📉 减少表格数: {total_stats['total_original'] - total_stats['total_final']}")
    print(f"{'='*60}\n")

    if dry_run:
        print("💡 提示: 这是试运行模式，未执行实际合并")
        print("   检查上述检测结果，如果正确，修改脚本中 dry_run=False 后重新运行")


if __name__ == "__main__":
    main()
