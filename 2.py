import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from rasterio.transform import xy

# 读取GeoTIFF文件
tif_file = 'E:/jie/github/repositories/2024_math_model/database/Geo/TIFF/chinadem_geo.tif'
with rasterio.open(tif_file) as dataset:
    # 提取地形数据（海拔）
    elevation = dataset.read(1)
    transform = dataset.transform

# 创建掩膜，将境外（无效值或负值）设置为NaN
elevation_masked = np.where(elevation < 0, np.nan, elevation)

# 创建经典的地形颜色渐变，从绿色到白色
terrain_cmap = LinearSegmentedColormap.from_list('terrain_cmap', 
                                                 ['green', 'yellow', 'brown', 'white'], N=256)

# 获取图像的行和列数
nrows, ncols = elevation.shape

# 获取左上角和右下角像素的经纬度，用于设定图像边界
top_left_lon, top_left_lat = xy(transform, 0, 0)  # 左上角
bottom_right_lon, bottom_right_lat = xy(transform, nrows, ncols)  # 右下角

# 生成纬度和经度的刻度
lon = np.linspace(top_left_lon, bottom_right_lon, ncols)
lat = np.linspace(top_left_lat, bottom_right_lat, nrows)

# 显示地形数据，横纵坐标用经纬度表示
plt.figure(figsize=(10, 8))
plt.imshow(elevation_masked, cmap=terrain_cmap, extent=[lon.min(), lon.max(), lat.min(), lat.max()])
plt.colorbar(label='Elevation (meters)')
plt.title('China DEM (1km resolution)')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.show()
