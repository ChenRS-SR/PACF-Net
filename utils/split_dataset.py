"""
数据集划分脚本
根据任务ID将裁剪后的图片数据集分为 train、val、test 三个集合
"""

import pandas as pd
from pathlib import Path

# 任务列表 (按顺序编号为 1-12)
TASK_LIST = [
    ("task_20260122_202652", 1, 1407),  # task_id = 1
    ("task_20260122_220644", 7, 1120),  # task_id = 2
    ("task_20260122_231127", 7, 742),   # task_id = 3
    ("task_20260122_234125", 9, 920),   # task_id = 4
    ("task_20260123_012915", 15, 1738), # task_id = 5
    ("task_20260123_062549", 15, 710),  # task_id = 6
    ("task_20260123_073418", 20, 933),  # task_id = 7
    ("task_20260123_081221", 10, 812),  # task_id = 8
    ("task_20260123_155043", 7, 774),   # task_id = 9
    ("task_20260123_162714", 6, 840),   # task_id = 10
    ("task_20260124_072841", 1, 692),   # task_id = 11
    ("task_20260124_075811", 6, 922),   # task_id = 12
]

# 数据划分规则
TRAIN_TASKS = [1, 2, 3, 4, 5, 6, 7, 8]       # 前8组
VAL_TASKS = [9, 12]                          # 第9和12组
TEST_TASKS = [10, 11]                        # 第10和11组


def split_dataset():
    """数据集划分主函数"""
    
    # 路径配置
    base_dir = Path(r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop")
    output_dir = base_dir
    
    print("\n" + "="*70)
    print("数据集划分工具")
    print("="*70 + "\n")
    
    # 检查输出目录
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return False
    
    print(f"✓ 输出目录: {output_dir}\n")
    
    # 初始化三个数据集
    train_data = []
    val_data = []
    test_data = []
    all_data = []  # 保持原始顺序 (1-12)
    
    # 遍历每个任务
    for task_idx, (task_name, start_num, end_num) in enumerate(TASK_LIST, 1):
        task_id = task_idx
        csv_path = base_dir / task_name / "print_message.csv"
        
        if not csv_path.exists():
            print(f"⚠️  任务 {task_id} CSV 不存在: {csv_path}")
            continue
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 在第一列插入 task_id
            df.insert(0, 'task_id', task_id)
            
            print(f"✓ 加载任务 {task_id}: {task_name} ({len(df)} 行)")
            
            # 添加到 all_data (保持顺序)
            all_data.append(df)
            
            # 根据规则添加到对应数据集
            if task_id in TRAIN_TASKS:
                train_data.append(df)
            elif task_id in VAL_TASKS:
                val_data.append(df)
            elif task_id in TEST_TASKS:
                test_data.append(df)
            
        except Exception as e:
            print(f"❌ 读取任务 {task_id} 失败: {e}")
            return False
    
    # 合并数据
    print(f"\n{'='*70}")
    print("合并数据集...")
    print(f"{'='*70}\n")
    
    # 按照原始顺序 (1-12) 合并 all_data
    all_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    print(f"All   集: {len(all_df)} 行 (所有任务: 按顺序 1-12)")
    print(f"Train 集: {len(train_df)} 行 (任务: {TRAIN_TASKS})")
    print(f"Val   集: {len(val_df)} 行 (任务: {VAL_TASKS})")
    print(f"Test  集: {len(test_df)} 行 (任务: {TEST_TASKS})")
    
    # 保存CSV文件
    print(f"\n{'='*70}")
    print("保存数据集...")
    print(f"{'='*70}\n")
    
    try:
        # 保存 all_data.csv
        all_csv_path = output_dir / "all_data.csv"
        all_df.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存 all_data.csv: {all_csv_path}")
        
        # 保存 train.csv
        train_csv_path = output_dir / "train.csv"
        train_df.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存 train.csv: {train_csv_path}")
        
        # 保存 val.csv
        val_csv_path = output_dir / "val.csv"
        val_df.to_csv(val_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存 val.csv: {val_csv_path}")
        
        # 保存 test.csv
        test_csv_path = output_dir / "test.csv"
        test_df.to_csv(test_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存 test.csv: {test_csv_path}")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    
    # 显示数据统计
    print(f"\n{'='*70}")
    print("数据集统计")
    print(f"{'='*70}\n")
    
    print(f"All 集统计:")
    print(f"  总行数: {len(all_df)}")
    print(f"  任务数: {len(all_df['task_id'].unique())}")
    print(f"  包含任务: {sorted(all_df['task_id'].unique().tolist())}")
    
    print(f"\nTrain 集统计:")
    print(f"  总行数: {len(train_df)}")
    print(f"  任务数: {len(train_df['task_id'].unique())}")
    print(f"  包含任务: {sorted(train_df['task_id'].unique().tolist())}")
    
    print(f"\nVal 集统计:")
    print(f"  总行数: {len(val_df)}")
    print(f"  任务数: {len(val_df['task_id'].unique())}")
    print(f"  包含任务: {sorted(val_df['task_id'].unique().tolist())}")
    
    print(f"\nTest 集统计:")
    print(f"  总行数: {len(test_df)}")
    print(f"  任务数: {len(test_df['task_id'].unique())}")
    print(f"  包含任务: {sorted(test_df['task_id'].unique().tolist())}")
    
    print(f"\n总计: {len(all_df)} 行数据")
    
    print(f"\n{'='*70}")
    print("✓ 数据集划分完成！")
    print(f"{'='*70}\n")
    
    return True, all_df, train_df, val_df, test_df


def analyze_distribution(all_df, train_df, val_df, test_df, output_dir):
    """
    分析样本分布统计
    
    Args:
        all_df: 全部数据
        train_df: 训练集数据
        val_df: 验证集数据
        test_df: 测试集数据
        output_dir: 输出目录
    """
    
    # 要统计的参数类别
    class_params = ['flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class']
    
    # 数据集名称和数据
    datasets = {
        'all_data': all_df,
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    print(f"\n{'='*70}")
    print("样本分布统计")
    print(f"{'='*70}\n")
    
    # 对每个数据集进行统计
    for dataset_name, df in datasets.items():
        if df.empty:
            print(f"\n{dataset_name} 集: (空)")
            continue
        
        print(f"\n{'─'*70}")
        print(f"{dataset_name.upper()} 集 - 样本分布统计")
        print(f"{'─'*70}")
        print(f"总样本数: {len(df)}\n")
        
        # 统计每个参数类别
        for param in class_params:
            if param not in df.columns:
                print(f"⚠️  列 '{param}' 不存在")
                continue
            
            # 获取值的分布
            counts = df[param].value_counts().sort_index()
            
            print(f"  {param}:")
            for class_val, count in counts.items():
                percentage = (count / len(df)) * 100
                print(f"    class {class_val}: {count:5d} 样本 ({percentage:6.2f}%)")
            print()
    
    print(f"\n{'='*70}")
    print("✓ 统计完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    base_dir = Path(r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop")
    success, all_df, train_df, val_df, test_df = split_dataset()
    if not success:
        import sys
        sys.exit(1)
    
    # 分析样本分布
    analyze_distribution(all_df, train_df, val_df, test_df, base_dir)
