import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from main import getPre

# 原始数组大小：(5019, 7062) 
tif_file = 'database\Geo\TIFF\chinadem_geo.tif'

# ------------------------------------------------ #
# 函数功能：打开TIFF文件
# 函数输入: tif文件名
# 函数返回: 二维数组数据
def open_tiff(tif_file):
    with rasterio.open(tif_file) as dataset:
        # 查看一些元数据
        print('元数据:')
        print(dataset.meta)

        # 读取图像数据（这里读取第一个波段）
        band1 = dataset.read(1)
        return band1
# ------------------------------------------------ #
# 函数功能：裁剪二维数组边缘的无效值
# 函数输入：NumPy数组、无效值invalid_num
# 函数返回：裁剪的数组trimmed_arr
def trim_np_arr(np_array, invalid_num):
    # 找到包含有效值的行
    valid_rows = np.any(np_array > invalid_num, axis=1)
    # 找到包含有效值的列
    valid_cols = np.any(np_array > invalid_num, axis=0)
    # 使用有效行和列创建新的数组
    trimmed_arr = np_array[valid_rows][:, valid_cols]
    return trimmed_arr
# ------------------------------------------------ #
# 函数功能：变换二维数组的大小
# 函数输入：二维数组matrix、目标大小target_shape
# 函数返回：变换后的数组scaled_mat
def zoom_arr(matrix, target_shape):
    # 定义目标形状
    target_shape = (144, 256)
    # 计算缩放因子
    zoom_factors = (target_shape[0] / matrix.shape[0], target_shape[1] / matrix.shape[1])
    # 使用在 scipy.ndimage 中的 zoom 函数进行缩放
    scaled_mat = zoom(matrix, zoom_factors)
    return scaled_mat
# ------------------------------------------------ #

### 从这里开始

# 获取数据
dem_data = open_tiff(tif_file)

# 转为NumPy数组
np_data = np.array(dem_data)

# 裁剪数据边缘
trimmed_dem = trim_np_arr(np_data, -32768)

# 缩放至与降水量相同的分辨率
# 均为(144, 256)
trimmed_scaled_dem = zoom_arr(trimmed_dem, (144, 256))

# 展示：查看数据形状，已缩放为(144, 256)
print('数据形状 trimmed_scaled_dem:', trimmed_scaled_dem.shape)

# 将数组转为float，并将"无数据"的部分(即国界外，值为-32768.0)替换为NaN
float_dem = trimmed_scaled_dem.astype(float)
float_dem[float_dem < - 1000] = np.nan

# 展示：查看此时的高程图
plt.figure(1)
plt.imshow(float_dem, cmap='gray')
plt.title('float dem inborder')
plt.show(block=False)


# 打开降水量文件，得到1990-2020降水量数据
filename = 'database/CHM_PRE_0.25dg_19612022.nc'
year_begin = 1990
year_end = 2020
# 降水量数据masked_pre为(144, 256)的二维数组
masked_pre = getPre(filename, year_begin, year_end)
# 计算降水量年平均值 (latitude, longitude)
mean_precipitation = np.nanmean(masked_pre, axis=0) * 365  

# 接下来准备绘制降水量关于海拔的散点图

# 创建布尔掩码，找出有效（非NaN）数据
valid_mask_dem = ~np.isnan(float_dem)
valid_mask_precip = ~np.isnan(mean_precipitation)

# 找到同时有效的数据索引
valid_mask = valid_mask_dem & valid_mask_precip

# 过滤有效数据
x = float_dem[valid_mask]
y = mean_precipitation[valid_mask]

# 创建散点图
plt.figure(2, figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.title("Scatter Plot of Mean Precipitation vs. Trimmed Scaled DEM (Without NaN)")
plt.xlabel("Trimmed Scaled DEM/m")
plt.ylabel("Mean Precipitation/mm·year^{-1}")
plt.grid()
plt.show()

