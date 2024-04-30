#----------------Verson 4 可处理多个被试者的单数据与多标签----------------#
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime

# 读取CSV文件，不包含第一行和第二行
def read_csv_file(input_file_path):
    df_data = pd.DataFrame()
    df_data = pd.read_csv(input_file_path, usecols=lambda col: col not in ['gaze_object', 'gaze_object_class', 'gaze_object_class&name'])
    # 将最后一列的名称改为 'Timestamp'
    df_data.rename(columns={df_data.columns[-1]: 'Timestamp'}, inplace=True)
    df_data = resample_and_interpolate(df_data, 50)
   
    return df_data

# 将数据重采样到固定频率，并补上缺失值
def resample_and_interpolate(df, freq_hz):
    # 将时间戳列转换为日期时间格式
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

    # 设置采样频率
    desired_freq = str(int(1000 / freq_hz)) + 'L'

    # 设置采样频率为指定频率
    df_resampled = df.set_index('Timestamp').resample(desired_freq).mean(numeric_only=True)

    # 使用线性插值填充缺失值
    df_resampled = df_resampled.interpolate(method='linear')

    # 将时间戳转换为毫秒级UNIX时间戳
    df_resampled['Timestamp'] = df_resampled.index.astype('int64') // 10 ** 6
    # df_resampled['Timestamp'] = df_resampled.index.astype('int64') // 10**3

    # 返回固定频率的DataFrame
    return df_resampled

# 保存处理后的DataFrame为新的CSV文件
def save_to_csv(df, output_file_path):
    df.to_csv(output_file_path, index=False)

# 读取标签文件，返回标签信息的DataFrame
def read_label_files(label_file_path):
    df_label = pd.DataFrame()

    df_label = pd.read_csv(label_file_path, usecols=[0, 1], skiprows=[1,2,4,6,8,10,12,14], encoding='GBK')   #注意不是skiprows=[0]，否则就把标题Skip了
    #df = df.drop(df.index[-1])
    # df_label = df_label.append(df, ignore_index=True)
    print(df_label.shape)
    print(df_label.values)

    return df_label

# 根据标签文件为数据添加标签列
def add_label_column(df_data, df_label):
    # 新增标签列，并初始化为0
    df_data['Label'] = 0

    for _, label_row in df_label.iterrows():
        label_timestamp = label_row[1]
        label_category = label_row[0]

        if int(label_category) in range(1, 6):
            # 计算标签时间戳前后5秒的时间范围
            start_time = label_timestamp * 1000 - 5 * 1000
            end_time = label_timestamp *1000 + 5 * 1000

            # 根据时间范围给数据添加标签
            df_data.loc[(df_data['Timestamp'] >= start_time) &
                        (df_data['Timestamp'] <= end_time) & (df_data['Label'] == 0), 'Label'] = label_category

    return df_data

# 主函数
def process_one_person_data(input_file_path, label_file_path, output_file_path):
    # 处理CSV数据，不包含前两行
    df_data = read_csv_file(input_file_path)

    # 读取标签文件
    df_label = read_label_files(label_file_path)

    # 为数据添加标签列
    df_data = add_label_column(df_data, df_label)

    # 将处理后的DataFrame保存为新的CSV文件
    save_to_csv(df_data, output_file_path)


def main(data_folder_path):
    # Get a list of subfolders under data_folder_path
    subfolders = [f.name for f in os.scandir(data_folder_path) if f.is_dir()]

    for subfolder in subfolders:
        # Create the output folder for each subfolder
        output_folder = os.path.join('./Datasets-Test/train-EYE(50Hz)', subfolder)
        os.makedirs(output_folder, exist_ok=True)

        # Input file path
        input_file_path = os.path.join(data_folder_path, subfolder,'EyeData_'+str(subfolder)+'.csv')

        # Label folder path
        label_file_path = os.path.join(data_folder_path, subfolder, 'AnxietyGroundtruth_'+str(subfolder)+'.csv')

        # Output file path
        output_file_path = os.path.join(output_folder, 'output.csv')

        # Process data and save to output file
        process_one_person_data(input_file_path, label_file_path, output_file_path)


if __name__ == "__main__":
    data_folder_path = "./Data/"  # Data folder path containing subfolders with input.csv and Label folder
    main(data_folder_path)