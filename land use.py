# 求土地形状

import rasterio
import matplotlib.pyplot as plt

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
    plt.imshow(band1, cmap='gray')
    plt.colorbar()
    plt.title('Band 1')
    plt.show()
