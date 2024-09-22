import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

# 打开 TIFF 文件
def open_tiff(tif_file):
    with rasterio.open(tif_file) as dataset:
        # 查看一些元数据
        print('元数据:')
        print(dataset.meta)

        # 读取图像数据（这里读取第一个波段）
        band1 = dataset.read(1)
        
        # 打印数据形状
        print('数据形状:', band1.shape)

        # 绘制数据
        plt.imshow(band1, cmap='gray')
        plt.colorbar()
        plt.title('Band 1')
        plt.show()
    

# 文件路径

# (5019, 7062) 
# 'nodata': -32768.0
tif_file = 'database\Albers_105\TIFF\chdem_105.tif'
# 'nodata': -3.4028234663852886e+38
tif_file2 = 'database\Albers_105\TIFF\chdem_Aspect.tif'
# 'nodata': 255.0
tif_file3 = 'database\Albers_105\TIFF\chdem_hillshade.tif'
# 'nodata': -3.4028234663852886e+38
tif_file4 = 'database\Albers_105\TIFF\chdem_Slope.tif'


# (5019, 7062) 
# 'nodata': -32768.0
# tif_file = 'database\Geo\TIFF\chinadem_geo.tif'

open_tiff(tif_file)
open_tiff(tif_file2)
open_tiff(tif_file3)
open_tiff(tif_file4)