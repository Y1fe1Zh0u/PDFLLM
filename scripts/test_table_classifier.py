"""测试表格分类和标题提取

运行此脚本测试表格分类器的效果
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.table_classifier import TableClassifier, classify_and_title_tables
from src.utils.logger import setup_script_logger
import pandas as pd

# 设置日志
logger = setup_script_logger("table_classifier_test")


def test_classifier():
    """测试分类器"""
    # 读取一个已提取的JSON文件
    results_dir = Path("data/outputs/test_results")
    json_files = list(results_dir.glob("*_extracted.json"))

    if not json_files:
        logger.error("没有找到提取结果文件")
        return

    # 测试第一个文件
    test_file = json_files[0]
    logger.info(f"测试文件: {test_file.name}")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tables = data.get('tables', [])
    text_chunks = data.get('text_chunks', [])

    logger.info(f"文档包含 {len(tables)} 个表格")

    # 运行分类
    enhanced_tables = classify_and_title_tables(tables, text_chunks)

    # 统计
    financial_count = 0
    fundraising_count = 0
    other_count = 0

    print(f"\n{'='*80}")
    print(f"表格分类结果 - {data.get('document_id', 'unknown')}")
    print(f"{'='*80}\n")

    for i, table in enumerate(enhanced_tables[:20], 1):  # 只显示前20个
        classification = table.get('classification', {})
        title = table.get('title', '')
        page = table.get('page', 0)
        table_type = classification.get('type', 'unknown')
        confidence = classification.get('confidence', 0)
        category = classification.get('category', 'unknown')

        # 统计
        if table_type == 'financial_report':
            financial_count += 1
            type_emoji = "💰"
        elif table_type == 'fundraising':
            fundraising_count += 1
            type_emoji = "💵"
        else:
            other_count += 1
            type_emoji = "📄"

        print(f"{type_emoji} 表格 {i} (页{page})")
        print(f"   标题: {title}")
        print(f"   类型: {table_type} ({category})")
        print(f"   置信度: {confidence:.2f}")
        print(f"   上下文标题: {table.get('context_title', '未找到')}")
        print()

    # 汇总
    print(f"{'='*80}")
    print(f"分类汇总")
    print(f"{'='*80}")
    print(f"💰 财务报表: {financial_count} 个")
    print(f"💵 募资相关: {fundraising_count} 个")
    print(f"📄 其他表格: {other_count} 个")
    print(f"{'='*80}\n")

    # 保存增强后的结果
    output_file = results_dir / f"{test_file.stem}_classified.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        enhanced_data = data.copy()
        # 只保存可序列化的部分
        enhanced_data['tables'] = [
            {
                **t,
                'data': t.get('data')  # 保持原始data
            }
            for t in enhanced_tables
        ]
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

    logger.info(f"增强后的结果已保存至: {output_file}")
    print(f"✅ 增强后的结果已保存至: {output_file.name}")


def test_specific_tables():
    """测试特定表格的分类"""
    classifier = TableClassifier()

    # 测试案例1：财务报表
    print("\n" + "="*60)
    print("测试案例 1: 模拟资产负债表")
    print("="*60)

    df_balance = pd.DataFrame({
        '项目': ['流动资产', '货币资金', '应收账款', '存货'],
        '本期金额': [1000000, 200000, 300000, 500000],
        '上期金额': [900000, 180000, 280000, 440000]
    })

    context = "合并资产负债表（2023年12月31日）以下单位为人民币元"

    result = classifier.classify_table(df_balance, context)
    print(f"类型: {result['type']}")
    print(f"类别: {result['category']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"建议标题: {result['suggested_title']}")

    # 测试案例2：募资表格
    print("\n" + "="*60)
    print("测试案例 2: 模拟募集资金使用表")
    print("="*60)

    df_fundraising = pd.DataFrame({
        '序号': [1, 2, 3],
        '项目名称': ['研发中心建设', '生产线扩建', '补充流动资金'],
        '投资金额（万元）': [5000, 8000, 2000],
        '募集资金拟投入（万元）': [5000, 8000, 2000]
    })

    context = "募集资金投资项目情况表"

    result = classifier.classify_table(df_fundraising, context)
    print(f"类型: {result['type']}")
    print(f"类别: {result['category']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"建议标题: {result['suggested_title']}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("表格分类和标题提取测试")
    print("="*80)

    # 测试实际数据
    print("\n📊 测试实际提取的数据...")
    test_classifier()

    # 测试特定案例
    print("\n🧪 测试特定案例...")
    test_specific_tables()

    print("\n✅ 测试完成！")
