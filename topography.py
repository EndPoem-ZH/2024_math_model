import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import os

# 打开 TIFF 文件
def open_tiff(tif_file):
    with rasterio.open(tif_file) as dataset:
        # 查看一些元数据
        print('元数据:')
        print(dataset.meta)

        # 读取图像数据（这里读取第一个波段）
        band1 = dataset.read(1)
        
        # 转换为 NumPy 数组，以便于处理
        np_array = np.array(band1)

        # 找到包含有效值的行
        valid_rows = np.any(np_array > -32768, axis=1)

        # 找到包含有效值的列
        valid_cols = np.any(np_array > -32768, axis=0)

        # 使用有效行和列创建新的数组
        trimmed_dem = np_array[valid_rows][:, valid_cols]

        # 定义目标形状
        target_shape = (144, 256)
        # 计算缩放因子
        zoom_factors = (target_shape[0] / trimmed_dem.shape[0], target_shape[1] / trimmed_dem.shape[1])
        # 使用在 scipy.ndimage 中的 zoom 函数进行缩放
        trimmed_scaled_dem = zoom(trimmed_dem, zoom_factors)

        # 打印数据形状
        print('数据形状:', trimmed_scaled_dem.shape)

        # 绘制数据
        plt.imshow(trimmed_scaled_dem, cmap='gray')
        plt.colorbar()
        plt.title('Scaled DEM of China')
        plt.show()
    

# 文件路径

# (5019, 7062) 
tif_file = 'database\Geo\TIFF\chinadem_geo.tif'
open_tiff(tif_file)
