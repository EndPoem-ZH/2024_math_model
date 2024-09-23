# 大数据驱动的地理综合问题

# pip install netCDF4 numpy matplotlib
import datetime
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## 问题1 描述降水量和土地利用

### 问题1.1 按时间求各地平均降水量，绘制热力图，查看空间分布情况

# ------------------------------------------------ #
# 函数输入：数据集名称、起始年份、终止年份
# 函数返回：二维数组masked_pre（国境内降水量）
def getPre(filename, year_begin, year_end):
    # 打开文件
    pre_set = nc.Dataset(filename)

    # 查看数据集信息
    # print(pre_set)

    # 提取年份和降水量数据

    time_var = pre_set.variables['time']
    time_units = time_var.units  # 假设时间单位是 "hours since 1961-01-01 00:00:00"
    time_values = time_var[:]  # 获取时间数值，单位为小时

    # 确定时间基准点
    base_time = datetime.datetime(1961, 1, 1)  # 基准时间为1961-01-01 00:00

    # 计算1990年1月1日到2020年1月1日的小时数
    start_date = datetime.datetime(year_begin, 1, 1)  # 题目中为1990-01-01 00:00
    end_date = datetime.datetime(year_end, 1, 1)    # 题目中为2020-01-01 00:00

    # 计算从基准时间开始的小时数
    start_hours = (start_date - base_time).total_seconds() / 3600  # 从1961-01-01到1990-01-01的小时数
    end_hours = (end_date - base_time).total_seconds() / 3600      # 从1961-01-01到2020-01-01的小时数

    time_mask = (time_values >= start_hours) & (time_values <= end_hours)
    time_indices = np.where(time_mask)[0]  # 获取对应的时间索引

    pre_vara = pre_set.variables['pre']  # 假设降水量变量名为'pre'
    pre_var = pre_vara[time_indices, :, :]  # 根据时间索引提取对应的数据

    # # 创建一个与经纬度维度相同大小的数组，用于存储大于五十的数据个数
    # count_greater_than_50 = np.zeros((144, 256), dtype=int)

    # # 遍历三维数组，统计每个经纬度位置大于五十的数据个数
    # for t in range(10958):
    #     mask = pre_var[t, :, :] > 50
    #     count_greater_than_50[mask] += 1

    # # 打印结果
    # print("Count of values greater than 50 at each latitude and longitude:")
    # print(count_greater_than_50)
    # 掩码(144*256, 在国界内为 True，国界外为 False)
    mask = pre_var >= 0

    # 获取国界内的降水量数据
    masked_pre = np.where(mask, pre_var, np.nan)  # 使用 NaN 替换国界外的降水量
    return masked_pre

# ------------------------------------------------ #
# 边缘裁剪，以便稍后和高程图对应
# 计算国界内的平均降水量，排除 NaN 值
def trim_NaN(matrix):
    # 找到所有不是 NaN 的行
    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    # 找到所有不是 NaN 的列
    valid_cols = ~np.all(np.isnan(matrix), axis=0)
    # 使用有效行和列创建新的数组
    trimmed_mat = matrix[valid_rows][:, valid_cols]
    return trimmed_mat
# ------------------------------------------------ #

# 从这里开始
if __name__ == '__main__':

    filename = 'database/CHM_PRE_0.25dg_19612022.nc'
    year_begin = 1990
    year_end = 2020
    masked_pre = getPre(filename, year_begin, year_end)

    mean_precipitation = np.nanmean(masked_pre, axis=0) * 365  # 计算年平均值 (latitude, longitude)

    # 初始化一个形状为 (144, 256) 的数组，用于存储大于50的降水量数据的个数
    count_greater_50 = np.zeros((144, 256), dtype=int)

    # 遍历每个时间步，统计大于50的降水量数据的个数
    for t in range(masked_pre.shape[0]):
        count_greater_50 += (masked_pre[t, :, :] > 50).astype(int)

    # 打印结果
    print("Count of precipitation values greater than 50 at each latitude and longitude:\n", count_greater_50)
    # 打印形状和示例数据
    print("Masked Precipitation Shape:", masked_pre.shape)
    print("Mean Precipitation Shape:", mean_precipitation.shape)
    trimmed_pre = trim_NaN(mean_precipitation)
    print("Trimmed Precipitation Shape:", trimmed_pre.shape)

    # 绘图
    # 绘制平均降水量图像
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_precipitation, cmap='Blues', aspect='auto', 
            norm=colors.Normalize(vmin=0, vmax=np.nanmax(mean_precipitation)))

    # plt.figure(1)
    # plt.imshow(mean_precipitation, cmap='Blues')
    plt.colorbar(label='Mean Precipitation (mm)')
    plt.title('Mean Precipitation from 1990 to 2020')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # 添加经纬度刻度
    plt.xticks(ticks=np.arange(0, 256, 32), labels=np.linspace(72, 136, num=256)[::32].astype(int))  # 假设256表示经度
    plt.yticks(ticks=np.arange(0, 144, 18), labels=np.linspace(18, 54, num=144)[::18].astype(int))  # 假设144表示纬度

    # 翻转 y 轴（纬度）
    plt.gca().invert_yaxis() 

    # 显示图像
    plt.show()



## 问题2



## 问题3



## 问题4