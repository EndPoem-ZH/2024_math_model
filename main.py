# 大数据驱动的地理综合问题

# pip install netCDF4
import netCDF4 as nc

## 打开文件
pre_set = nc.Dataset('database/CHM_PRE_0.25dg_19612022.nc')

# 查看数据集信息
print('数据集信息：')
print(pre_set)

# 访问变量
print('降雨量信息：')
precipitation = pre_set['pre']  # 替换为你的变量名
print(precipitation)

# 可视化


## 问题1



## 问题2



## 问题3



## 问题4