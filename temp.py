import os
import numpy as np
import rasterio

# 定义数据文件夹
data_dir = 'database\\temp'

# 收集所有的年份
years = [str(year) for year in range(1979, 2019)]
tif_files = []

# 遍历年份文件夹，收集tif文件路径
for year in years:
    year_dir = os.path.join(data_dir, f"{year}_avg")
    for filename in os.listdir(year_dir):
        if filename.endswith('.tif'):
            tif_files.append(os.path.join(year_dir, filename))

# 假设所有的tif文件有相同的shape，读取第一个以获得shape
with rasterio.open(tif_files[0]) as src:
    height, width = src.read(1).shape

# 初始化三维数组
num_days = len(tif_files)
temp_data = np.empty((num_days, height, width))

# 读取所有的tif文件并填充到三维数组中
for i, tif_file in enumerate(tif_files):
    with rasterio.open(tif_file) as src:
        temp_data[i] = src.read(1)  # 读取第一个波段的数据

# 当前的 temp_data 是一个 (time, latitude, longitude) 的三维数组
print(temp_data.shape)  # (num_days, height, width)