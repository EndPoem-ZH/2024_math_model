import datetime
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# 打开文件
pre_set = nc.Dataset('database/CHM_PRE_0.25dg_19612022.nc')

# 获取时间和降水量数据
time_var = pre_set.variables['time']
pre_vara = pre_set.variables['pre']  # 假设降水量变量名为 'pre'

# 基准时间 (假设时间单位为 "hours since 1961-01-01 00:00:00")
base_time = datetime.datetime(1961, 1, 1)
time_values = time_var[:]  # 获取时间数值，单位为小时

# 时间跨度 1990年1月1日到2020年12月31日
start_date = datetime.datetime(1990, 1, 1)
end_date = datetime.datetime(2020, 12, 31)
start_hours = (start_date - base_time).total_seconds() / 3600  # 从1961到1990的小时数
end_hours = (end_date - base_time).total_seconds() / 3600  # 从1961到2020的小时数

# 找到1990年到2020年的时间索引
time_mask = (time_values >= start_hours) & (time_values <= end_hours)
time_indices = np.where(time_mask)[0]  # 1990年到2020年之间的所有时间索引

# 创建每个年份的掩码
years = np.arange(1990, 2021)  # 1990到2020年
yearly_means = []  # 用于存储每年的全国平均降水量

# 计算每年的平均降水量
for year in years:
    # 计算每一年开始和结束的小时数
    year_start = datetime.datetime(year, 1, 1)
    year_end = datetime.datetime(year, 12, 31)
    
    start_hours_year = (year_start - base_time).total_seconds() / 3600
    end_hours_year = (year_end - base_time).total_seconds() / 3600
    
    # 找到每个年份对应的时间索引
    year_mask = (time_values >= start_hours_year) & (time_values <= end_hours_year)
    year_indices = np.where(year_mask)[0]
    
    # 提取该年份的降水量数据
    pre_var_year = pre_vara[year_indices, :, :]  # 提取该年的降水量数据
    
    # 掩码，确保国界内的有效降水量 (假设 >= 0 为有效数据)
    mask = pre_var_year >= 0
    masked_pre_year = np.where(mask, pre_var_year, np.nan)  # 使用 NaN 替换无效数据
    
    # 计算该年份的全国平均降水量，忽略空间维度上的 NaN 值
    mean_precipitation_year = np.nanmean(masked_pre_year)
    
    # 将每年的全国平均降水量保存起来
    yearly_means.append(mean_precipitation_year)

# 关闭 netCDF 数据集
pre_set.close()

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(years, yearly_means, marker='o', linestyle='-', color='b')
plt.title('Annual Mean Precipitation in China (1990-2020)')
plt.xlabel('Year')
plt.ylabel('Mean Precipitation (mm)')
plt.grid(True)
plt.xticks(ticks=years, rotation=45)  # x轴上显示每一年
plt.tight_layout()

# 显示折线图
plt.show()

