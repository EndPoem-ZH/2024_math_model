# 求土地利用率

import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

# 文件路径
tif_file = 'database\land use & cover\cropland-1900.tif'

# 打开 TIFF 文件
with rasterio.open(tif_file) as dataset:
    # 查看一些元数据
    print('元数据:')
    print(dataset.meta)

    # 读取图像数据（这里读取第一个波段）
    band1 = dataset.read(1)
    
    # 打印数据形状
    print('数据形状:', band1.shape)

    # 绘制数据
    plt.figure(1)
    plt.imshow(band1, cmap='gray')
    plt.colorbar()
    plt.title('Band 1')
    plt.show(block=False)

# 定义读取影像数据并计算面积的函数
def read_tif_and_calculate_area(tif_file):
    with rasterio.open(tif_file) as src:
        data = src.read(1)  # 读取第一层
        area = np.sum(data)
        return area

# 可选土地类型
land_types = ['cropland', 'forest', 'grass', 'shrub', 'wetland']

# 存储各土地类型每年的面积
yearly_area = {land_type: {} for land_type in land_types}

# 读取所有土地类型的面积
for land_type in land_types:
    for year in range(1990, 2021):
        tif_filename = f'database/land use & cover/{land_type}-{year}.tif'  # 注意调整路径分隔符
        if os.path.exists(tif_filename):
            yearly_area[land_type][year] = read_tif_and_calculate_area(tif_filename)

# 计算每种土地类型的变化率
yearly_change_rate = {land_type: {} for land_type in land_types}
for land_type in land_types:
    years = list(yearly_area[land_type].keys())
    for i in range(1, len(years)):
        prev_year = years[i - 1]
        curr_year = years[i]
        if yearly_area[land_type][prev_year] != 0:  # 确保不除以0
            change_rate = ((yearly_area[land_type][curr_year] - yearly_area[land_type][prev_year]) / yearly_area[land_type][prev_year]) * 100
            yearly_change_rate[land_type][curr_year] = change_rate

# 绘制面积折线图 (Figure 1)
plt.figure(figsize=(12, 6))
for land_type in land_types:
    years = list(yearly_area[land_type].keys())
    plt.plot(years, list(yearly_area[land_type].values()), marker='o', label=land_type)

plt.title('Yearly Area of Different Land Types (1990-2020)')
plt.xlabel('Year')
plt.ylabel('Area')
plt.xticks(range(1990, 2021), rotation=45)
plt.grid()
plt.legend(title='Land Use Type')
plt.tight_layout()  # 自动调整布局
# plt.savefig('land_area_plot.png')  # 保存图像
plt.show(block=False)

# 绘制变化率折线图 (Figure 2)
plt.figure(figsize=(12, 6))
for land_type in land_types:
    years = list(yearly_change_rate[land_type].keys())
    plt.plot(years, list(yearly_change_rate[land_type].values()), marker='o', label=land_type)

plt.title('Yearly Change Rate of Different Land Types (1990-2020)')
plt.xlabel('Year')
plt.ylabel('Change Rate (%)')
plt.xticks(range(1990, 2021), rotation=45)
plt.grid()
plt.legend(title='Land Use Type')
plt.tight_layout()  # 自动调整布局
# plt.savefig('land_change_rate_plot.png')  # 保存图像
plt.show()
