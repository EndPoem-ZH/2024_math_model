import netCDF4 as nc

# 打开 netCDF 文件
file_path = 'database/CHM_PRE_0.25dg_19612022.nc'
pre_set = nc.Dataset(file_path, mode='r')  # 以只读方式打开文件

# 打印文件的基本信息
print(pre_set)

# 获取所有变量的名称和相关信息
print("文件中所有变量的名称：")
for var in pre_set.variables:
    print(var)

# 获取每个变量的详细信息
for var in pre_set.variables:
    print(f"\n变量名称: {var}")
    print(f"维度: {pre_set.variables[var].dimensions}")
    print(f"数据类型: {pre_set.variables[var].dtype}")
    print(f"属性: {pre_set.variables[var].ncattrs()}")
