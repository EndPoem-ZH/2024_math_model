import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为SimHei以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 打开文件
pre_set = nc.Dataset('database/CHM_PRE_0.25dg_19612022.nc')

# 获取时间和降水量数据
time_var = pre_set.variables['time']
pre_vara = pre_set.variables['pre']  # 假设降水量变量名为 'pre'

# 基准时间 (假设时间单位为 "hours since 1961-01-01 00:00:00")
base_time = datetime(1961, 1, 1)
time_values = time_var[:]  # 获取时间数值，单位为小时

# 时间跨度 1990年1月1日到2020年12月31日
start_date = datetime(1990, 1, 1)
end_date = datetime(2020, 12, 31)
start_hours = (start_date - base_time).total_seconds() / 3600  # 从1961到1990的小时数
end_hours = (end_date - base_time).total_seconds() / 3600  # 从1961到2020的小时数

# 找到1990年到2020年的时间索引
time_mask = (time_values >= start_hours) & (time_values <= end_hours)
time_indices = np.where(time_mask)[0]  # 1990年到2020年之间的所有时间索引

# 获取时间数据，转换为 datetime 对象
time_dates = [base_time + timedelta(hours=float(t)) for t in time_values[time_indices]]

# 创建一个 Pandas DataFrame 来存储月和降雨量
data = {'Date': time_dates}

# 提取对应时间段的降水量数据
pre_data = pre_vara[time_indices, :, :]
mask = pre_data >= 0  # 只保留有效降水量
masked_pre = np.where(mask, pre_data, np.nan)


total_precipitation = np.nansum(masked_pre, axis=(1, 2))  # 按时间计算全国平均降水量

data['Precipitation'] = total_precipitation

# 创建 DataFrme
df = pd.DataFrame(data)

# 提取年份和月份
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 根据月份绘制降雨量盒须图
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='Precipitation', data=df, color='black', fliersize=0, width=0.7, 
            boxprops=dict(color="gray"), medianprops=dict(color="black"))

# 设置标题和坐标轴标签
plt.title('1990年到2020年各月降雨量变化特征', fontsize=16)
plt.xlabel('月份', fontsize=14)
plt.ylabel('降雨量 (mm)', fontsize=14)

# 调整颜色和样式，类似示例中的黑白风格
plt.xticks(ticks=np.arange(0, 12), labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
plt.grid(True, linestyle='--', linewidth=0.5, color='gray')  # 添加网格线

# 调整盒须图的样式
plt.box(False)  # 去掉图表外框
plt.tight_layout()

# 显示图像
plt.show()
