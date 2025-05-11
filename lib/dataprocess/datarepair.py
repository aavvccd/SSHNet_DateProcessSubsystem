import os
import pandas as pd
import netCDF4 as nc
import numpy as np
from NcOp import NcFileIO
import scipy.stats as stats
import pandas as pd
from gensim.utils import is_empty
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
def export_to_netcdf(gridded_data, grid_lon, grid_lat, filename):
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



def qul_control(data_array, data_list):
    qual_arr = []
    for i in data_list:
        if is_empty(qual_arr):
            qual_arr = data_array[i]<2147483647
        else:
            qual_arr = (qual_arr) & (data_array[i]<2147483647)
    return qual_arr


resolution = 1 #生成网格的分辨率
intermathord = 'linear'

# 创建存储网格数据的字典
gridded_data = {}

#建立输出文件夹
data_out_dirname = 'dataresult'

#创建输出目录
def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"文件夹创建成功: {path}")
    except Exception as e:
        print(f"创建失败: {e}")


file_rootdir = 'ncdata' #文件目录
dirs=os.listdir(file_rootdir) #获取文件夹中的所有cycle

# nc_file = nc.Dataset('gridded_data.nc', 'w')

create_folder(data_out_dirname)

for i in dirs:
    file_datadir = os.path.join(file_rootdir,i)
    nc_filedata = NcFileIO.read_netcdf4(file_datadir,'r')

    data_arrays = { #数据标签
        'lon': [],
        'lat': [],
        'surface_type': [],
        'mean_sea_surface': [],
        'range_ku': [],
        'swh_ku': [],
        'sig0_ku': [],
        'alt': [],
        'wind_speed_model_u': [],
        'wind_speed_model_v': [],
        'wind_speed_alt': [],
        'wind_speed_rad': [],
        'agc_ku': [],
        'agc_numval_ku': [],  # 注意：原数据标签中此处缺少逗号，已修正
        'rad_water_vapor': [],
        'off_nadir_angle_wf_ku': [],
        # 'waveforms_20hz_ku': []  # 二维数据可用嵌套列表（如 [[v1, v2, ...], ...]）
    }
    data_lable_list = list(data_arrays.keys())

    for nc_data in nc_filedata:
        for lable in data_lable_list:
            data_arrays[lable].append(nc_data.variables[lable][:])
    for lable in data_lable_list:
        data_arrays[lable] = np.concatenate(data_arrays[lable])

    qual = qul_control(data_arrays, data_lable_list)
    mask = (data_arrays['mean_sea_surface'] != 2) & (data_arrays['mean_sea_surface'] != 3) & qual

    for lable in data_lable_list:
        data_arrays[lable] = data_arrays[lable][mask]

    data_arrays['lon'] = np.where(data_arrays['lon']>180, data_arrays['lon']-360,data_arrays['lon'])
    lon_min, lon_max = np.floor(data_arrays['lon'].min()), np.ceil(data_arrays['lon'].max())  # 设置网格坐标界限
    lat_min, lat_max = np.floor(data_arrays['lat'].min()), np.ceil(data_arrays['lat'].max())
    grid_lon = np.arange(lon_min, lon_max + resolution, resolution)  # 设置网格数组
    grid_lat = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat) #生成二维网格
    points = np.column_stack((data_arrays['lon'], data_arrays['lat'])) #生成用于插值的

    #cydle子目录创建
    data_out_dirname = os.path.join(data_out_dirname, i)
    create_folder(data_out_dirname)

    for lable in data_lable_list:
        grid_data = griddata(
            points, data_arrays[lable],
            (grid_lon2d, grid_lat2d),
            method=intermathord,
            fill_value=np.nan
        )
        # 存储插值结果到字典
        gridded_data[lable] = grid_data

        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        #绘制海洋数据
        contour = ax.contourf(
            grid_lon2d,
            grid_lat2d,
            grid_data,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            levels=20,
            zorder=0  # 确保数据层在底层
        )
        # 添加陆地覆盖层（关键步骤）
        ax.add_feature(
            cfeature.LAND,
            facecolor='white',  # 与背景同色
            edgecolor='none',  # 隐藏边界线
            zorder=1  # 覆盖在数据层之上
        )
        # 添加海岸线参考
        ax.add_feature(
            cfeature.COASTLINE.with_scale('50m'),
            edgecolor='gray',
            linewidth=0.5,
            zorder=2  # 显示在最顶层
        )

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.6)
        cbar.set_label(lable)

        # 添加标题和网格
        plt.title(f'Interpolated {lable} Data (Resolution: {resolution}°)')
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

        output_path = os.path.join(data_out_dirname, f'{lable}_interpolation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 生成文件名
    output_path = os.path.join(
         data_out_dirname,
          f"merged_grid_{i}.nc"  # 示例: merged_grid_cycle001.nc
    )
    export_to_netcdf(gridded_data, grid_lon, grid_lat, output_path)




















