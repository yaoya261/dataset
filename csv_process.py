#----------------Verson 1 可处理单个被试者的单数据----------------#
# import pandas as pd
# import numpy as np
# from datetime import datetime
#
# # 读取CSV文件，不包含第一行和第二行
# def read_csv_file(file_path):
#     df = pd.read_csv(file_path, header=None, skiprows=[0, 1])
#     return df
#
# # 添加13位毫秒级UNIX时间戳到第二列
# def add_milliseconds_unix_timestamp(df, initial_unix_timestamp, frequency_hz):
#     frequency_hz = float(frequency_hz)
#
#     # 计算每个样本的时间间隔（以毫秒为单位）
#     time_interval_ms = int(1000 / frequency_hz)
#
#     # 生成毫秒级时间戳的序列
#     timestamp_milliseconds = np.arange(initial_unix_timestamp * 1000,
#                                        initial_unix_timestamp * 1000 + len(df) * time_interval_ms,
#                                        time_interval_ms)
#
#     # 将13位毫秒级UNIX时间戳写入第二列
#     df.insert(1, 'Unix Timestamp (ms)', timestamp_milliseconds)
#
#     # 将时间戳转换为日期时间格式，用于更好的可视化
#     df['DateTime'] = pd.to_datetime(df['Unix Timestamp (ms)'], unit='ms')
#
#     return df
#
# # 保存处理后的DataFrame为新的CSV文件
# def save_to_csv(df, output_file_path):
#     df.to_csv(output_file_path, index=False)
#
# # 主函数
# def main(input_file_path, output_file_path):
#     # 读取CSV文件，获取初始时间戳和采样频率
#     df_info = pd.read_csv(input_file_path, header=None, nrows=2)
#     initial_unix_timestamp = df_info.iloc[0, 0]
#     frequency_hz = df_info.iloc[1, 0]
#
#     # 处理CSV数据，不包含前两行
#     df_data = read_csv_file(input_file_path)
#     df_data = add_milliseconds_unix_timestamp(df_data, initial_unix_timestamp, frequency_hz)
#
#     # 为第一列添加列名
#     df_data.columns = ['EDA'] + df_data.columns.tolist()[1:]
#
#     # 将处理后的DataFrame保存为新的CSV文件
#     save_to_csv(df_data, output_file_path)
#
# if __name__ == "__main__":
#     input_file_path = "./Data Process/EDA.csv"  # 输入CSV文件路径
#     output_file_path = "./Data Process/output.csv"  # 输出CSV文件路径
#     main(input_file_path, output_file_path)


#----------------Verson 2 可处理单个被试者的单数据与单标签----------------#
# import pandas as pd
# import numpy as np
# from datetime import datetime
#
# # 读取CSV文件，不包含第一行和第二行
# def read_csv_file(file_path):
#     df = pd.read_csv(file_path, header=None, skiprows=[0, 1])
#     return df
#
# # 添加13位毫秒级UNIX时间戳到第二列
# def add_milliseconds_unix_timestamp(df, initial_unix_timestamp, frequency_hz):
#     frequency_hz = float(frequency_hz)
#
#     # 计算每个样本的时间间隔（以毫秒为单位）
#     time_interval_ms = int(1000 / frequency_hz)
#
#     # 生成毫秒级时间戳的序列
#     timestamp_milliseconds = np.arange(initial_unix_timestamp * 1000,
#                                        initial_unix_timestamp * 1000 + len(df) * time_interval_ms,
#                                        time_interval_ms)
#
#     # 将13位毫秒级UNIX时间戳写入第二列
#     df.insert(1, 'Unix Timestamp (ms)', timestamp_milliseconds)
#
#     # 将时间戳转换为日期时间格式，用于更好的可视化
#     df['DateTime'] = pd.to_datetime(df['Unix Timestamp (ms)'], unit='ms')
#
#     return df
#
# # 保存处理后的DataFrame为新的CSV文件
# def save_to_csv(df, output_file_path):
#     df.to_csv(output_file_path, index=False)
#
# # 读取标签文件，返回标签信息的DataFrame
# def read_label_file(label_file_path):
#     df_label = pd.read_csv(label_file_path, skiprows=[1])
#     df_label = df_label.drop(df_label.index[-1])
#     return df_label
#
# # 根据标签文件为数据添加标签列
# def add_label_column(df_data, df_label):
#     # 初始化标签列为0
#     # df_data['Label'] = 0
#
#     for _, label_row in df_label.iterrows():
#         label_timestamp = label_row[1]
#         label_category = label_row[0]
#
#         if int(label_category) in range(1, 6):
#             # 计算标签时间戳前后5秒的时间范围
#             start_time = label_timestamp - 5 * 1000
#             end_time = label_timestamp + 5 * 1000
#
#             # 根据时间范围给数据添加标签
#             df_data.loc[(df_data['Unix Timestamp (ms)'] >= start_time) &
#                         (df_data['Unix Timestamp (ms)'] <= end_time), 'Label'] = label_category
#
#     return df_data
#
# # 主函数
# def main(input_file_path, label_file_path, output_file_path):
#     # 读取CSV文件，获取初始时间戳和采样频率
#     df_info = pd.read_csv(input_file_path, header=None, nrows=2)
#     initial_unix_timestamp = df_info.iloc[0, 0]
#     frequency_hz = df_info.iloc[1, 0]
#
#     # 处理CSV数据，不包含前两行
#     df_data = read_csv_file(input_file_path)
#     df_data = add_milliseconds_unix_timestamp(df_data, initial_unix_timestamp, frequency_hz)
#
#     # 读取标签文件
#     df_label = read_label_file(label_file_path)
#
#     # 为数据添加标签列
#     df_data = add_label_column(df_data, df_label)
#
#     # 为第一列添加列名
#     df_data.columns = ['EDA'] + df_data.columns.tolist()[1:]
#
#     # 将处理后的DataFrame保存为新的CSV文件
#     save_to_csv(df_data, output_file_path)
#
# if __name__ == "__main__":
#     input_file_path = "./Data Process/EDA.csv"  # 输入CSV文件路径
#     label_file_path = "./Data Process/label.csv"  # 标签CSV文件路径
#     output_file_path = "./Data Process/output.csv"  # 输出CSV文件路径
#     main(input_file_path, label_file_path, output_file_path)


#----------------Verson 3 可处理单个被试者的单数据与多标签----------------#
# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
#
# # 读取CSV文件，不包含第一行和第二行
# def read_csv_file(file_path):
#     df = pd.read_csv(file_path, header=None, skiprows=[0, 1])
#     return df
#
# # 添加13位毫秒级UNIX时间戳到第二列
# def add_milliseconds_unix_timestamp(df, initial_unix_timestamp, frequency_hz):
#     frequency_hz = float(frequency_hz)
#
#     # 计算每个样本的时间间隔（以毫秒为单位）
#     time_interval_ms = int(1000 / frequency_hz)
#
#     # 生成毫秒级时间戳的序列
#     timestamp_milliseconds = np.arange(initial_unix_timestamp * 1000,
#                                        initial_unix_timestamp * 1000 + len(df) * time_interval_ms,
#                                        time_interval_ms)
#
#     # 将13位毫秒级UNIX时间戳写入第二列
#     df.insert(1, 'Unix Timestamp (ms)', timestamp_milliseconds)
#
#     # 将时间戳转换为日期时间格式，用于更好的可视化
#     # df['DateTime'] = pd.to_datetime(df['Unix Timestamp (ms)'], unit='ms')
#
#     return df
#
# # 保存处理后的DataFrame为新的CSV文件
# def save_to_csv(df, output_file_path):
#     df.to_csv(output_file_path, index=False)
#
# # 读取标签文件，返回标签信息的DataFrame
# def read_label_files(label_folder_path):
#     df_label = pd.DataFrame()
#
#     # 列出目录中的所有文件
#     files = os.listdir(label_folder_path)
#
#     for file in files:
#         if file.endswith('.csv'):
#             file_path = os.path.join(label_folder_path, file)
#             df = pd.read_csv(file_path, skiprows=[1])   #注意不是skiprows=[0]，否则就把标题Skip了
#             df = df.drop(df.index[-1])
#             # df_label = df_label.append(df, ignore_index=True)
#             df_label = pd.concat([df_label, df], ignore_index=True)  # 使用pd.concat()进行替代未来即将弃用的append()
#
#     return df_label
#
# # 根据标签文件为数据添加标签列
# def add_label_column(df_data, df_label):
#     # 初始化标签列为0
#     df_data['Label'] = 0
#
#     for _, label_row in df_label.iterrows():
#         label_timestamp = label_row[1]
#         label_category = label_row[0]
#
#         if int(label_category) in range(1, 6):
#             # 计算标签时间戳前后5秒的时间范围
#             start_time = label_timestamp - 5 * 1000
#             end_time = label_timestamp + 5 * 1000
#
#             # 根据时间范围给数据添加标签
#             df_data.loc[(df_data['Unix Timestamp (ms)'] >= start_time) &
#                         (df_data['Unix Timestamp (ms)'] <= end_time), 'Label'] = label_category
#
#     return df_data
#
# # 主函数
# def main(input_file_path, label_folder_path, output_file_path):
#     # 读取CSV文件，获取初始时间戳和采样频率
#     df_info = pd.read_csv(input_file_path, header=None, nrows=2)
#     initial_unix_timestamp = df_info.iloc[0, 0]
#     frequency_hz = df_info.iloc[1, 0]
#
#     # 处理CSV数据，不包含前两行
#     df_data = read_csv_file(input_file_path)
#     df_data = add_milliseconds_unix_timestamp(df_data, initial_unix_timestamp, frequency_hz)
#
#     # 读取标签文件
#     df_label = read_label_files(label_folder_path)
#
#     # 为数据添加标签列
#     df_data = add_label_column(df_data, df_label)
#
#     # 为第一列添加列名
#     df_data.columns = ['EDA'] + df_data.columns.tolist()[1:]
#
#     # 将处理后的DataFrame保存为新的CSV文件
#     save_to_csv(df_data, output_file_path)
#
# if __name__ == "__main__":
#     input_file_path = "./Data Process/EDA.csv"  # 输入CSV文件路径
#     label_folder_path = "./Data Process/Label/"  # 标签CSV文件夹路径
#     output_file_path = "./Data Process/output.csv"  # 输出CSV文件路径
#     main(input_file_path, label_folder_path, output_file_path)


# #----------------Verson 4 可处理多个被试者的单数据与多标签----------------#
# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
#
# # 读取CSV文件，不包含第一行和第二行
# def read_csv_file(file_path):
#     df = pd.read_csv(file_path, header=None, skiprows=[0, 1])
#     return df
#
# # 添加13位毫秒级UNIX时间戳到第二列
# def add_milliseconds_unix_timestamp(df, initial_unix_timestamp, frequency_hz):
#     frequency_hz = float(frequency_hz)
#
#     # 计算每个样本的时间间隔（以毫秒为单位）
#     time_interval_ms = int(1000 / frequency_hz)
#
#     # 生成毫秒级时间戳的序列
#     timestamp_milliseconds = np.arange(initial_unix_timestamp * 1000,
#                                        initial_unix_timestamp * 1000 + len(df) * time_interval_ms,
#                                        time_interval_ms)
#
#     # 将13位毫秒级UNIX时间戳写入第二列
#     # df.insert(1, 'Timestamp', timestamp_milliseconds)
#     df.insert(len(df.columns), 'Timestamp', timestamp_milliseconds)
#
#     # 将时间戳转换为日期时间格式，用于更好的可视化
#     # df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='ms')
#
#     return df
#
# # 保存处理后的DataFrame为新的CSV文件
# def save_to_csv(df, output_file_path):
#     df.to_csv(output_file_path, index=False)
#
# # 读取标签文件，返回标签信息的DataFrame
# def read_label_files(label_folder_path):
#     df_label = pd.DataFrame()
#
#     # 列出目录中的所有文件
#     files = os.listdir(label_folder_path)
#
#     for file in files:
#         if file.endswith('.csv'):
#             file_path = os.path.join(label_folder_path, file)
#             df = pd.read_csv(file_path, usecols=[0, 1], skiprows=[1], encoding='GBK')   #注意不是skiprows=[0]，否则就把标题Skip了
#             df = df.drop(df.index[-1])
#             # df_label = df_label.append(df, ignore_index=True)
#             df_label = pd.concat([df_label, df], ignore_index=True)  # 使用pd.concat()进行替代未来即将弃用的append()
#
#     return df_label
#
# # 根据标签文件为数据添加标签列
# def add_label_column(df_data, df_label):
#     # 新增标签列，并初始化为0
#     df_data['Label'] = 0
#
#     for _, label_row in df_label.iterrows():
#         label_timestamp = label_row[1]
#         label_category = label_row[0]
#
#         if int(label_category) in range(1, 6):
#             # 计算标签时间戳前后5秒的时间范围
#             start_time = label_timestamp - 5 * 1000
#             end_time = label_timestamp + 5 * 1000
#
#             # 根据时间范围给数据添加标签
#             df_data.loc[(df_data['Timestamp'] >= start_time) &
#                         (df_data['Timestamp'] <= end_time) & (df_data['Label'] == 0), 'Label'] = label_category
#             # df_data.loc[(df_data['Timestamp'] >= start_time) &
#             #             (df_data['Timestamp'] <= end_time), 'Label'] = label_category
#
#     return df_data
#
# # 主函数
# def process_one_person_data(input_file_path, label_folder_path, output_file_path):
#     # 读取CSV文件，获取初始时间戳和采样频率
#     df_info = pd.read_csv(input_file_path, header=None, nrows=2)
#     initial_unix_timestamp = df_info.iloc[0, 0]
#     frequency_hz = df_info.iloc[1, 0]
#
#     # 处理CSV数据，不包含前两行
#     df_data = read_csv_file(input_file_path)
#     df_data = add_milliseconds_unix_timestamp(df_data, initial_unix_timestamp, frequency_hz)
#
#     # 读取标签文件
#     df_label = read_label_files(label_folder_path)
#
#     # 为数据添加标签列
#     df_data = add_label_column(df_data, df_label)
#
#     # 为第一列添加列名
#     # df_data.columns = ['HR'] + df_data.columns.tolist()[1:]
#     df_data.columns = ['X', 'Y', 'Z'] + df_data.columns.tolist()[3:]
#
#     # 将处理后的DataFrame保存为新的CSV文件
#     save_to_csv(df_data, output_file_path)
#
#
# def main(data_folder_path):
#     # Get a list of subfolders under data_folder_path
#     subfolders = [f.name for f in os.scandir(data_folder_path) if f.is_dir()]
#
#     for subfolder in subfolders:
#         # Create the output folder for each subfolder
#         output_folder = os.path.join('./Datasets-Test/train', subfolder)
#         os.makedirs(output_folder, exist_ok=True)
#
#         # Input file path
#         input_file_path = os.path.join(data_folder_path, subfolder, 'ACC.csv')
#
#         # Label folder path
#         label_folder_path = os.path.join(data_folder_path, subfolder, 'Label')
#
#         # Output file path
#         output_file_path = os.path.join(output_folder, 'output.csv')
#
#         # Process data and save to output file
#         process_one_person_data(input_file_path, label_folder_path, output_file_path)
#
#
# if __name__ == "__main__":
#     data_folder_path = "./Data Process/"  # Data folder path containing subfolders with input.csv and Label folder
#     main(data_folder_path)


#----------------Verson 5 加入重采样功能----------------#
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 读取CSV文件，不包含第一行和第二行
def read_csv_file(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=[0, 1])
    return df

# 添加13位毫秒级UNIX时间戳到第二列
def add_milliseconds_unix_timestamp(df, initial_unix_timestamp, frequency_hz):
    frequency_hz = float(frequency_hz)

    # 计算每个样本的时间间隔（以毫秒为单位）
    time_interval_ms = int(1000 / frequency_hz)

    # 生成毫秒级时间戳的序列
    timestamp_milliseconds = np.arange(initial_unix_timestamp * 1000,
                                       initial_unix_timestamp * 1000 + len(df) * time_interval_ms,
                                       time_interval_ms).astype('int64')

    # 将13位毫秒级UNIX时间戳写入第二列
    # df.insert(1, 'Timestamp', timestamp_milliseconds)
    df.insert(len(df.columns), 'Timestamp', timestamp_milliseconds)

    # 将时间戳转换为日期时间格式，用于更好的可视化
    # df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='ms')

    return df

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
    # df_resampled['Timestamp'] = df_resampled.index.astype('int64') // 10**3
    df_resampled['Timestamp'] = df_resampled.index.astype('int64') // 10**6

    # 返回固定频率的DataFrame
    return df_resampled

# 保存处理后的DataFrame为新的CSV文件
def save_to_csv(df, output_file_path):
    df.to_csv(output_file_path, index=False)

# 读取标签csv文件，返回标签信息的DataFrame
def read_label_files(file_path):
    df_label = pd.DataFrame()

    df_label = pd.read_csv(file_path, usecols=[0, 1], skiprows=[1,2,4,6,8,10,12,14], encoding='GBK')   #注意不是skiprows=[0]，否则就把标题Skip了
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
            #print("ok")
            start_time = label_timestamp * 1000 - 5 * 1000
            end_time = label_timestamp * 1000 + 5 * 1000

            # 根据时间范围给数据添加标签
            df_data.loc[(df_data['Timestamp'] >= start_time) &
                        (df_data['Timestamp'] <= end_time) & (df_data['Label'] == 0), 'Label'] = label_category
            # df_data.loc[(df_data['Timestamp'] >= start_time) &
            #             (df_data['Timestamp'] <= end_time), 'Label'] = label_category

    return df_data

# 主函数, 只是读取一个人的一个模态数据，即是一个csv文件
def process_one_person_data(input_file_path, label_file_path, output_file_path,str):
    # 读取CSV文件，获取初始时间戳和采样频率
    df_info = pd.read_csv(input_file_path, header=None, nrows=2)
    initial_unix_timestamp = df_info.iloc[0, 0]
    frequency_hz = df_info.iloc[1, 0]

    # 处理CSV数据，不包含前两行
    df_data = read_csv_file(input_file_path)
    df_data = add_milliseconds_unix_timestamp(df_data, initial_unix_timestamp, frequency_hz)
    df_data = resample_and_interpolate(df_data, 50)

    # 读取标签文件
    df_label = read_label_files(label_file_path)

    # 为数据添加标签列
    df_data = add_label_column(df_data, df_label)

    # 为第一列添加列名     #-------------需根据模态类型修改-------------#
    df_data.columns = [str] + df_data.columns.tolist()[1:]
    # df_data.columns = ['X', 'Y', 'Z'] + df_data.columns.tolist()[3:]

    # 将处理后的DataFrame保存为新的CSV文件
    save_to_csv(df_data, output_file_path)


def main(data_folder_path):
    # Get a list of subfolders under data_folder_path
    subfolders = [f.name for f in os.scandir(data_folder_path) if f.is_dir()]

    #01 02 subfolder
    for subfolder in subfolders:
        # Create the output folder for each subfolder
        output_folder_ACC = os.path.join('./Datasets-Test/train-ACC(50Hz)', subfolder)
        output_folder_EDA = os.path.join('./Datasets-Test/train-EDA(50Hz)', subfolder)
        #output_folder_EYE = os.path.join('./Datasets-Test/train-EYE(50Hz)', subfolder)
        output_folder_BVP = os.path.join('./Datasets-Test/train-BVP(50Hz)', subfolder)
        output_folder_HR = os.path.join('./Datasets-Test/train-HR(50Hz)', subfolder)
        output_folder_TEMP = os.path.join('./Datasets-Test/train-TEMP(50Hz)', subfolder)

        os.makedirs(output_folder_ACC, exist_ok=True)
        os.makedirs(output_folder_EDA, exist_ok=True)
        #os.makedirs(output_folder_EYE, exist_ok=True)
        os.makedirs(output_folder_BVP, exist_ok=True)
        os.makedirs(output_folder_HR, exist_ok=True)
        os.makedirs(output_folder_TEMP, exist_ok=True)


        # Input file path
        input_file_path_ACC = os.path.join(data_folder_path, subfolder, 'E4_elevator','ACC.csv')
        input_file_path_EDA = os.path.join(data_folder_path, subfolder, 'E4_elevator','EDA.csv')
        #input_file_path_EYE = os.path.join(data_folder_path, subfolder, 'E4_elevator','EYE.csv')
        input_file_path_BVP = os.path.join(data_folder_path, subfolder, 'E4_elevator','BVP.csv')
        input_file_path_HR = os.path.join(data_folder_path, subfolder, 'E4_elevator','HR.csv')
        input_file_path_TEMP = os.path.join(data_folder_path, subfolder, 'E4_elevator','TEMP.csv')

        # Label folder path
        label_file_path = os.path.join(data_folder_path, subfolder, 'AnxietyGroundtruth_'+str(subfolder)+'.csv')

        # Output file path
        output_file_path_ACC = os.path.join(output_folder_ACC, 'output.csv')
        output_file_path_EDA = os.path.join(output_folder_EDA, 'output.csv')
        #output_file_path_EYE = os.path.join(output_folder_EYE, 'output.csv')
        output_file_path_BVP = os.path.join(output_folder_BVP, 'output.csv')
        output_file_path_HR = os.path.join(output_folder_HR, 'output.csv')
        output_file_path_TEMP = os.path.join(output_folder_TEMP, 'output.csv')

        # Process data and save to output file
        process_one_person_data(input_file_path_ACC, label_file_path, output_file_path_ACC,'ACC')
        process_one_person_data(input_file_path_EDA, label_file_path, output_file_path_EDA,'EDA')
        #process_one_person_data(input_file_path_EYE, label_file_path, output_file_path_EYE)
        process_one_person_data(input_file_path_BVP, label_file_path, output_file_path_BVP,'BVP')
        process_one_person_data(input_file_path_HR, label_file_path, output_file_path_HR,'HR')
        process_one_person_data(input_file_path_TEMP, label_file_path, output_file_path_TEMP,'TEMP')


if __name__ == "__main__":
    data_folder_path = "./Data/"  # Data folder path containing subfolders with input.csv and Label folder
    main(data_folder_path)