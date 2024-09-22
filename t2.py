import datetime
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

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

# 创建存储时间和月平均降水量的列表
monthly_means = []
monthly_dates = []

# 循环计算1990到2020年每个月的全国平均降水量
current_date = start_date

while current_date <= end_date:
    # 计算当前月份开始和结束的小时数
    next_month = current_date.replace(day=28) + datetime.timedelta(days=4)  # 跳到下个月的第1天
    next_month = next_month.replace(day=1)  # 将天设置为1
    
    start_hours_month = (current_date - base_time).total_seconds() / 3600
    end_hours_month = (next_month - base_time).total_seconds() / 3600
    
    # 找到对应月份的时间索引
    month_mask = (time_values >= start_hours_month) & (time_values < end_hours_month)
    month_indices = np.where(month_mask)[0]
    
    # 提取该月份的降水量数据
    pre_var_month = pre_vara[month_indices, :, :]  # 提取该月的降水量数据
    
    # 掩码，确保国界内的有效降水量 (假设 >= 0 为有效数据)
    mask = pre_var_month >= 0
    masked_pre_month = np.where(mask, pre_var_month, np.nan)  # 使用 NaN 替换无效数据
    
    # 计算该月的全国平均降水量，忽略空间维度上的 NaN 值
    mean_precipitation_month = np.nanmean(masked_pre_month)
    
    # 将每个月的平均降水量保存起来
    monthly_means.append(mean_precipitation_month)
    monthly_dates.append(current_date)  # 保存对应的月份
    
    # 更新当前日期为下个月
    current_date = next_month

# 关闭 netCDF 数据集
pre_set.close()

# 绘制月平均降水量的折线图
plt.figure(figsize=(10, 6))
plt.plot(monthly_dates, monthly_means, marker='o', linestyle='-', color='b', label='Monthly Mean Precipitation')

# 设置图形格式
plt.title('Monthly Mean Precipitation in China (1990-2020)')
plt.xlabel('Date')
plt.ylabel('Mean Precipitation (mm)')
plt.grid(True)
plt.legend()

# 格式化日期轴
plt.gca().xaxis.set_major_locator(MonthLocator(interval=12))  # 每年显示一次
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # 显示年-月
plt.gcf().autofmt_xdate()  # 旋转日期标签

# 显示图像
plt.tight_layout()
plt.show()
