from matplotlib.colors import ListedColormap
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from main import getPre
# 分段线性回归
import pwlf
# 随机森林
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为SimHei以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
# 函数功能：预处理：读取TIFF数据集，
# 函数输入：TIFF数据集文件名tif_file, 无数据的数值no_data和有效数据界limit
# 函数返回：转换、裁剪、缩放
# 说明：no_data表示未采集数据的数值表示（比如海拔数据no_data = -32768），
#      limit表示有效的数据界（比如我国最低海拔limit_min = -154.31）
def tiff_preprocess(tif_file, no_data, limit_min, limit_max):
    dem_data = open_tiff(tif_file)

    # 转为NumPy数组
    np_data = np.array(dem_data)

    # 裁剪数据边缘
    trimmed_dem = trim_np_arr(np_data, no_data)

    # 缩放至与降水量相同的分辨率
    # 均为(144, 256)
    trimmed_scaled_dem = zoom_arr(trimmed_dem, (144, 256))

    # 展示：查看数据形状，已缩放为(144, 256)
    print('数据形状 trimmed_scaled_dem:', trimmed_scaled_dem.shape)

    # 将数组转为float，并将"无数据"的部分(即国界外)替换为NaN
    float_dem = trimmed_scaled_dem.astype(float)
    float_dem[(float_dem < limit_min) | (float_dem > limit_max)] = np.nan
    return float_dem
# ------------------------------------------------ #

### 从这里开始

# 原始数组大小：(5019, 7062) 

# 高程图
tif_dem = 'database\Geo\TIFF\chinadem_geo.tif'
dem_data = tiff_preprocess(tif_dem, -32768, -154.31, 8848.86)

# 坡度图
tif_slope = 'database\Albers_105\TIFF\chdem_Slope.tif'
slope_data = tiff_preprocess(tif_slope, -3.4e38, -10, 59.79)


# 高程图颜色映射：自定义地形颜色梯度（绿色 -> 棕色 -> 白色）
elevation_cmap = ListedColormap(["darkgreen", "green", "lightgreen", "yellow", "brown", "white"])

# 展示：查看高程图
plt.figure(figsize=[6, 4])  # 调整图形尺寸
plt.imshow(dem_data, cmap=elevation_cmap)  # 使用自定义的地形颜色映射
plt.title('中国数字高程图 (0.25°)', fontsize=14)  # 设置中文标题
plt.colorbar(label='高度 (米)', orientation='vertical', shrink=0.8)  # 调整颜色条尺寸和标签
plt.xlabel('经度', fontsize=12)
plt.ylabel('纬度', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
plt.show(block=False)

# 坡度图颜色映射：浅色 -> 深色（平缓 -> 陡峭）
slope_cmap = plt.cm.get_cmap('magma')

# 展示：查看坡度图
plt.figure(figsize=[6, 4])  # 调整图形尺寸
plt.imshow(slope_data, cmap=slope_cmap)  # 使用“magma”颜色映射，渐变清晰
plt.title('中国数字坡度图 (0.25°)', fontsize=14)  # 设置中文标题
plt.colorbar(label='坡度 (度)', orientation='vertical', shrink=0.8)  # 设置颜色条并缩小以适应图形
plt.xlabel('经度', fontsize=12)
plt.ylabel('纬度', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # 同样添加网格线
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

# 创建布尔掩码，筛选有效（非NaN）数据
valid_mask_dem = ~np.isnan(dem_data)
valid_mask_slope = ~np.isnan(slope_data)
valid_mask_precip = ~np.isnan(mean_precipitation)

# 找到同时有效的数据索引
valid_mask = valid_mask_dem & valid_mask_precip
valid_mask_eps = valid_mask & valid_mask_slope

# 过滤有效数据
x = dem_data[valid_mask]
y = mean_precipitation[valid_mask]
s = slope_data[valid_mask_slope]
x_min = min(x)
x_max = max(x)
y_max = max(y)
s_max = max(s)

# ------------------------------------------------ #
# 定义损失函数
def compute_cost(w, b, x, y):
    total_cost = 0
    M = len(x)
    # 逐点计算平方损失误差，然后求平均数
    for i in range(M):
        total_cost += (y[i] - w * x[i] - b) ** 2
    return total_cost / M
# ------------------------------------------------ #
# 定义算法拟合函数
def average(data):
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num
 # ------------------------------------------------ #
# 定义核心拟合函数
def fit(x, y):
    M = len(x)
    x_bar = average(x)

    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0

    for i in range(M):
        sum_yx += y[i] * (x[i] - x_bar)
        sum_x2 += x[i] ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))

    for i in range(M):
        sum_delta += (y[i] - w * x[i])
    b = sum_delta / M

    return w, b
# ------------------------------------------------ #

# 创建散点图
plt.figure()
plt.scatter(x, y, s=1)
# 绘制拟合曲线
w, b = fit(x,y)
print("w is: ", w)
print("b is: ", b)
cost = compute_cost(w, b, x,y)
print("cost is: ", cost)
# 针对每一个x，计算出预测的y值
pred_y = w * x + b
plt.plot(x, pred_y, c='r',label='线性拟合')

# 补充题图等信息
plt.title("平均降水量与海拔的散点图")
plt.xlabel("海拔/米")
plt.ylabel("平均降水量/毫米·年^(-1)")
plt.grid()
plt.show(block=False)

# 创建分段线性拟合对象
my_pwlf = pwlf.PiecewiseLinFit(x, y)
# 设置节点（可以根据需要调整节点的数量和位置）
breaks = [1500,4000,x_max]
my_pwlf.fit_with_breaks(breaks)
# res = my_pwlf.fitfast(3) # 3表示最多可以有3个段

# 查看拟合信息
print(my_pwlf.intercepts)
print(my_pwlf.slopes)
print(my_pwlf.fit_breaks)
print(my_pwlf.ssr)

# # 生成预测值
x_fit = np.linspace(0, x_max, 100)
y_fit = my_pwlf.predict(x_fit)

# 预测值应大于0
fit_mask = y_fit >= 0
x_fit_m = x_fit[fit_mask]
y_fit_m = y_fit[fit_mask]


# 绘图
plt.figure()
plt.scatter(x, y, color='blue', label='原始数据', s=1)
plt.plot(x_fit_m, y_fit_m, color='red', label='分段线性拟合')
plt.title('降水量与海拔的分段线性回归')
plt.xlabel('海拔/米')
plt.ylabel('降水量/毫米·天')
plt.legend()
plt.grid()
plt.show(block=False)

# 散点图
plt.figure()
x_eps = dem_data[valid_mask_eps]
y_eps = mean_precipitation[valid_mask_eps]
s_eps = slope_data[valid_mask_eps]
ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
ax.set_title('降水量关于海拔和坡度的散点图')  # 设置图名称
ax.scatter(x_eps, y_eps, s_eps, c = 'r', s=1)   # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
 
ax.set_xlabel('海拔/m')  # 设置x坐标轴
ax.set_ylabel('降水量/mm·年^(-1)')  # 设置y坐标轴
ax.set_zlabel('坡度/°')  # 设置z坐标轴
plt.show(block=False)


# # 随机森林
# # 划分训练集和测试集
# # X包括海拔、温度和坡度，y为暴雨下的日降水量(>16mm)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 构建随机森林回归模型
# rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 设置决策树的数量为100

# # 训练模型
# rf.fit(X_train, y_train)

# # 预测结果
# y_pred = rf.predict(X_test)

# # 模型评估
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print('均方误差:', mse)
# print('平均绝对误差 (MAE):', mae)
# print('判定系数 (R²):', r2)



# 定义函数：使用IQR法剔除异常值
def remove_outliers_iqr(X, y):
    # 计算目标变量（降雨量）的四分位数
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1  # 四分位距

    # 计算上限和下限，用于判断异常值
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 创建布尔掩码，筛选出正常范围内的数据
    mask = (y >= lower_bound) & (y <= upper_bound)

    # 返回剔除异常值后的特征和目标变量
    return X[mask], y[mask]

# 打开降水量文件，得到1990-2020降水量数据
filename = 'database/CHM_PRE_0.25dg_19612022.nc'
year_begin = 1990
year_end = 2020
masked_pre = getPre(filename, year_begin, year_end)
mean_precipitation = np.nanmean(masked_pre, axis=0) * 365  

# 创建布尔掩码，筛选有效（非NaN）数据
valid_mask_dem = ~np.isnan(dem_data)
valid_mask_slope = ~np.isnan(slope_data)
valid_mask_precip = ~np.isnan(mean_precipitation)
valid_mask = valid_mask_dem & valid_mask_precip & valid_mask_slope

# 过滤有效数据
X = np.column_stack((dem_data[valid_mask], slope_data[valid_mask]))  # 海拔和坡度作为特征
y = mean_precipitation[valid_mask]  # 降雨量作为目标变量

# 使用IQR法剔除异常值 
X_filtered, y_filtered = remove_outliers_iqr(X, y)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

# 创建随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测测试集的降雨量
y_pred = rf_model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'均方误差 (MSE): {mse:.2f}')
print(f'平均绝对误差 (MAE): {mae:.2f}')
print(f'决定系数 (R²): {r2:.2f}')

# 绘制实际降雨量 vs 预测降雨量的散点图
plt.figure()

# 绘制散点图并调整点的透明度和大小
plt.scatter(y_test, y_pred, alpha=0.6, s=10, color='blue', label="数据点")

# 绘制参考线：y = x (理想情况)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label="理想参考线")

# 设置坐标轴范围，去除空白
plt.xlim([min(y_test) - 100, max(y_test) + 100])
plt.ylim([min(y_pred) - 100, max(y_pred) + 100])

# 设置图表标题和标签
plt.xlabel('实际降雨量 (mm/年)', fontsize=12)
plt.ylabel('预测降雨量 (mm/年)', fontsize=12)
plt.title('随机森林模型：实际 vs 预测降雨量 (剔除异常值)', fontsize=14)

# 显示模型评估结果
mse_text = f'MSE: {mse:.2f}'
mae_text = f'MAE: {mae:.2f}'
r2_text = f'R²: {r2:.2f}'
plt.text(0.05, 0.9, mse_text, transform=plt.gca().transAxes, fontsize=12, color='black')
plt.text(0.05, 0.85, mae_text, transform=plt.gca().transAxes, fontsize=12, color='black')
plt.text(0.05, 0.8, r2_text, transform=plt.gca().transAxes, fontsize=12, color='black')

# 显示网格线和图例
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# 显示图像
plt.show()



