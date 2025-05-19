import os
import pandas as pd
import netCDF4 as nc
import numpy as np
from ..NcOp import NcFileIO
import scipy.stats as stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 自定义单位映射函数
def _get_units(var_name):
    units_map = {
        'mean_sea_surface': 'm',
        'swh_ku': 'm',
        'sig0_ku': 'dB',
        'wind_speed_alt': 'm/s',
        # 添加其他变量的单位...
    }
    return units_map.get(var_name, 'unitless')

# 修改后的数据导出部分
def export_to_netcdf(gridded_data, grid_lon, grid_lat, filename, resolution):
    """将插值后的所有数据导出到单个NetCDF文件"""
    with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
        # ====== 定义维度 ======
        lon_dim = ds.createDimension('lon', len(grid_lon))
        lat_dim = ds.createDimension('lat', len(grid_lat))

        # ====== 创建坐标变量 ======
        # 经度
        lon_var = ds.createVariable('lon', 'f4', ('lon',))
        lon_var[:] = grid_lon
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'Longitude'

        # 纬度
        lat_var = ds.createVariable('lat', 'f4', ('lat',))
        lat_var[:] = grid_lat
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'Latitude'

        # ====== 添加全局属性 ======
        ds.title = "Merged Satellite Gridded Data"
        ds.history = f"Created {pd.Timestamp.now().isoformat()}"
        ds.source = "Multi-source L2P altimetry data"
        ds.resolution = f"{resolution} degree"
        ds.Conventions = "CF-1.8"

        # ====== 写入所有数据变量 ======
        for var_name, data in gridded_data.items():
            # 跳过坐标变量
            if var_name in ['lon', 'lat']:
                continue

            # 创建变量
            var = ds.createVariable(
                varname=var_name,
                datatype='f4',
                dimensions=('lat', 'lon'),  # 注意维度顺序
                zlib=True,  # 启用压缩
                complevel=4,
                fill_value=-9999.0
            )

            # 添加变量属性
            var.long_name = var_name.replace('_', ' ').title()
            var.units = _get_units(var_name)  # 需要自定义单位获取函数
            var.grid_mapping = "latitude_longitude"

            # 处理缺失值
            masked_data = np.where(np.isnan(data), 0, data)
            var[:] = masked_data

        print(f"成功导出合并文件: {filename}")

# 质量控制
def qul_control(data, lable, qual_lable, qual_arrays):
    qual_arr = []
    print(lable)
    for qual in qual_lable:
        if(lable in qual) & (lable != 'alt'):
            qual_arr = qual_arrays[qual]
        else:
            qual_arr = data<data.max()
    return qual_arr

    # flag = True
    # for i in data_list:
    #     if flag:
    #         flag = False
    #         qual_arr=(data_array[i] < data_array[i].max())
    #     else:
    #         qual_arr = (qual_arr) & (data_array[i]<data_array[i].max())
    # return qual_arr

#创建输出目录
def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"文件夹创建成功: {path}")
    except Exception as e:
        print(f"创建失败: {e}")
