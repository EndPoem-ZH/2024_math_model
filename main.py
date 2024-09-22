# 大数据驱动的地理综合问题

import numpy as np
# pip install netCDF4
import netCDF4 as nc

## 打开文件
pre_set = nc.Dataset('database/CHM_PRE_0.25dg_19612022.nc')

# 查看数据集信息
print('数据集信息：')
print(pre_set)

# # 访问变量
# print('降雨量信息：')
# precipitation = pre_set['pre']  # 替换为你的变量名
# print(precipitation)

# # 查看变量信息
# variable_info = pre_set.variables['station_number']
# print(variable_info)
# data = variable_info[:]
# print(data)

# 提取年份和降水量数据
years_var = pre_set.variables['years'][:]  # 获取年份数据
pre_var = pre_set.variables['pre'][:]  # 获取降水量数据 (time, latitude, longitude)

# 我们需要提取1990-2020年的数据，从年份的第29年到第59年
start_year = years_var[29]  
end_year = years_var[59]    

# 找到对应的年份索引
year_indices = np.where((years_var >= start_year) & (years_var <= end_year))[0]

# 提取特定年份的降水量（假设前两维是时间和空间）
extracted_pre = pre_var[year_indices, :, :]  # 对应年份的降水量

# 输出结果
for i, year_index in enumerate(year_indices):
    print(f"年份: {years_var[year_index]}")
    print("相应的降水量数据:")
    print(extracted_pre[i])  # 输出对应年份的降水量数据
# 可视化

## 问题1

## 问题2



## 问题3



## 问题4