import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 数据准备
# 假设降水数据为 NetCDF 格式
precip_data = xr.open_dataset('database/CHM_PRE_0.25dg_19612022.nc')
# 假设高程数据为 GeoTIFF 格式
dem_data = rasterio.open('E:/jie/github/repositories/2024_math_model/database/Geo/TIFF/chinadem_geo.tif')

# 2. 特征提取
# 将降水数据转为 DataFrame
precip_df = precip_data.to_dataframe().reset_index()
precip_df = precip_df.rename(columns={'time': 'date', 'precipitation_variable': 'precipitation'})

# 提取 DEM 数据并转换为 DataFrame
dem_array = dem_data.read(1)  # 读取高程数据
height_data = dem_array.flatten()
lon, lat = np.meshgrid(np.arange(dem_data.bounds.left, dem_data.bounds.right, dem_data.res[0]),
                       np.arange(dem_data.bounds.bottom, dem_data.bounds.top, dem_data.res[1]))
lon_flat = lon.flatten()
lat_flat = lat.flatten()

# 创建 DEM DataFrame
dem_df = pd.DataFrame({
    'longitude': lon_flat,
    'latitude': lat_flat,
    'elevation': height_data
})

# 合并降水和DEM数据
combined_df = pd.merge(precip_df, dem_df, on=['longitude', 'latitude'], how='inner')

# 3. 模型训练
X = combined_df[['elevation']]  # 特征（地形特征）
y = combined_df['precipitation']  # 目标（降水量）

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 结果评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'均方误差 (MSE): {mse}')

# 可视化预测结果
plt.scatter(y_test, y_pred)
plt.xlabel('实际降水量')
plt.ylabel('预测降水量')
plt.title('实际降水量与预测降水量')
plt.show()
