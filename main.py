# 大数据驱动的地理综合问题

import datetime
import numpy as np
# pip install netCDF4
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# 打开文件
pre_set = nc.Dataset('database/CHM_PRE_0.25dg_19612022.nc')

# 查看数据集信息
# print(pre_set)

## 问题1 描述降水量和土地利用

### 问题1.1 按时间求各地平均降水量，绘制热力图，查看空间分布情况

# 提取年份和降水量数据

time_var = pre_set.variables['time']
time_units = time_var.units  # 假设时间单位是 "hours since 1961-01-01 00:00:00"
time_values = time_var[:]  # 获取时间数值，单位为小时

# 确定时间基准点
base_time = datetime.datetime(1961, 1, 1)  # 基准时间为1961-01-01 00:00

# 计算1990年1月1日到2020年1月1日的小时数
start_date = datetime.datetime(1990, 1, 1)  # 1990-01-01 00:00
end_date = datetime.datetime(2020, 1, 1)    # 2020-01-01 00:00

# 计算从基准时间开始的小时数
start_hours = (start_date - base_time).total_seconds() / 3600  # 从1961-01-01到1990-01-01的小时数
end_hours = (end_date - base_time).total_seconds() / 3600      # 从1961-01-01到2020-01-01的小时数

time_mask = (time_values >= start_hours) & (time_values <= end_hours)
time_indices = np.where(time_mask)[0]  # 获取对应的时间索引

pre_vara = pre_set.variables['pre']  # 假设降水量变量名为'pre'
pre_var = pre_vara[time_indices, :, :]  # 根据时间索引提取对应的数据


# 掩码(144*256, 在国界内为 True，国界外为 False)
mask = pre_var >= 0

# 获取国界内的降水量数据
masked_pre = np.where(mask, pre_var, np.nan)  # 使用 NaN 替换国界外的降水量

# 计算国界内的平均降水量，排除 NaN 值
mean_precipitation = np.nanmean(masked_pre, axis=0)  # 计算平均值 (latitude, longitude)

# 打印形状和示例数据
print("Masked Precipitation Shape:", masked_pre.shape)
print("Mean Precipitation Shape:", mean_precipitation.shape)

# 绘图
# 绘制平均降水量图像
plt.figure(figsize=(10, 6))
plt.imshow(mean_precipitation, cmap='Blues', aspect='auto', 
           norm=colors.Normalize(vmin=0, vmax=np.nanmax(mean_precipitation)))
plt.colorbar(label='Mean Precipitation (mm)')
plt.title('Mean Precipitation Over the Study Period')
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