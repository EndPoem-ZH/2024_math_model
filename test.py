import numpy as np

# pip install netCDF4
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def create_mask(array):
    """
    创建一个mask数组，判断输入数组中的每个值是否等于-99.9

    参数:
    array: 三维NumPy数组，形状为(22645, 144, 256)

    返回:
    mask: 布尔型三维数组，形状与输入数组相同
    """
    # 检查输入数组的形状
    if array.shape != (22645, 144, 256):
        raise ValueError("输入数组的形状必须为(22645, 144, 256)")
    
    # 创建mask，等于-99.9的元素对应False，其他元素对应True
    mask = array != -99.9
    return mask


# 打开文件
pre_set = nc.Dataset('database/CHM_PRE_0.25dg_19612022.nc')

# 提取年份和降水量数据
times_var = pre_set.variables['time'][:]  # 获取年份数据
pre_var = pre_set.variables['pre'][:]  # 获取降水量数据 (time, latitude, longitude)
# 示例使用
# 随机生成一个三维数组用于测试
# example_array = np.random.random((22645, 144, 256)) * 200 - 100  # 生成范围在[-100, 100)的随机数
mask = create_mask(pre_var)

print(mask)  # 输出mask

