import pandas as pd
import os

def convert_xlsx_to_csv_pandas(directory_path):
    """
    使用pandas将指定目录下的所有xlsx文件转换为csv文件
    
    参数:
        directory_path: 要转换的xlsx文件所在目录路径
    """
    # 获取目录下所有文件
    files = os.listdir(directory_path)
    
    # 筛选出所有xlsx文件
    xlsx_files = [f for f in files if f.endswith('.xlsx')]
    
    if not xlsx_files:
        print("在指定目录下未找到xlsx文件")
        return
    
    print(f"找到 {len(xlsx_files)} 个xlsx文件，开始转换...")
    
    for xlsx_file in xlsx_files:
        try:
            # 构建完整文件路径
            xlsx_path = os.path.join(directory_path, xlsx_file)
            
            # 生成csv文件名（保持原文件名，只改扩展名）
            csv_file = xlsx_file.replace('.xlsx', '.csv')
            csv_path = os.path.join(directory_path, csv_file)
            
            # 读取Excel文件
            df = pd.read_excel(xlsx_path)
            
            # 保存为CSV文件，避免中文乱码使用utf-8-sig编码
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            print(f"✓ 成功转换: {xlsx_file} -> {csv_file}")
            
        except Exception as e:
            print(f"✗ 转换失败 {xlsx_file}: {str(e)}")
    
    print("所有文件转换完成！")

# 使用方法：将下面的路径替换为你的实际目录路径
if __name__ == "__main__":
    # 指定包含xlsx文件的目录路径
    target_directory = "./iron_daily/data"  # 可以改为你的具体目录路径，如 "C:/我的文档/Excel文件"
    convert_xlsx_to_csv_pandas(target_directory)
