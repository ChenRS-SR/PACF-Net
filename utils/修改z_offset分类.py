import os
import pandas as pd
from pathlib import Path

# 目标目录
base_dir = r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01"

def classify_z_offset(z_offset):
    """
    根据 z_offset 值进行分类
    - 如果 z_offset < -0.04，返回 0
    - 如果 z_offset > 0.08，返回 2
    - 如果 -0.04 <= z_offset <= 0.08，返回 1
    """
    if z_offset < -0.04:
        return 0
    elif z_offset > 0.08:
        return 2
    else:
        return 1

def process_csv_files(base_dir):
    """
    处理所有子文件夹下的 print_message.csv 文件
    """
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    processed_count = 0
    total_rows_processed = 0
    
    for subdir in sorted(subdirs):
        csv_path = os.path.join(base_dir, subdir, 'print_message.csv')
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            print(f"❌ 跳过 {subdir}: print_message.csv 不存在")
            continue
        
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_path,encoding='gbk')
            
            # 检查必要的列是否存在
            if 'z_offset' not in df.columns or 'z_offset_class' not in df.columns:
                print(f"❌ {subdir}: 缺少 z_offset 或 z_offset_class 列")
                continue
            
            # 应用分类函数
            df['z_offset_class'] = df['z_offset'].apply(classify_z_offset)
            
            # 保存回原文件
            df.to_csv(csv_path, index=False)
            
            # 统计分类结果
            class_counts = df['z_offset_class'].value_counts().to_dict()
            total_rows = len(df)
            total_rows_processed += total_rows
            
            print(f"✅ {subdir}")
            print(f"   总行数: {total_rows}")
            print(f"   类别分布 - 0: {class_counts.get(0, 0)}, 1: {class_counts.get(1, 0)}, 2: {class_counts.get(2, 0)}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ {subdir}: 处理出错 - {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"处理完成！共处理 {processed_count} 个文件，总行数: {total_rows_processed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print(f"开始处理目录: {base_dir}\n")
    process_csv_files(base_dir)
