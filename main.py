# 大数据驱动的地理综合问题

import numpy as np
# pip install netCDF4
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# 打开文件
pre_set = nc.Dataset('database/CHM_PRE_0.25dg_19612022.nc')

# 查看数据集信息
# print(pre_set)

## 问题1 


### 问题1.1 按时间求各地平均降水量，绘制热力图，查看空间分布情况

# 提取年份和降水量数据
times_var = pre_set.variables['time'][:]  # 获取年份数据
pre_var = pre_set.variables['pre'][:]  # 获取降水量数据 (time, latitude, longitude)

# 掩码(144*256, 在国界内为 True，国界外为 False)
mask = pre_var != -99.9

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
plt.imshow(mean_precipitation, cmap='viridis', aspect='auto', norm=colors.Normalize(vmin=0, vmax=np.nanmax(mean_precipitation)))
plt.colorbar(label='Mean Precipitation (mm)')
plt.title('Mean Precipitation Over the Study Period')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 添加经纬度刻度
plt.xticks(ticks=np.arange(0, 256, 32), labels=np.linspace(-180, 180, num=256)[::32].astype(int))  # 假设256表示经度
plt.yticks(ticks=np.arange(0, 144, 18), labels=np.linspace(-90, 90, num=144)[::18].astype(int))  # 假设144表示纬度

# 翻转 y 轴（纬度）
plt.gca().invert_yaxis() 

# 显示图像
plt.show()




## 问题2



## 问题3



## 问题4