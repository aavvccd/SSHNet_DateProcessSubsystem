import os
import pandas as pd
import netCDF4 as nc
import numpy as np
from lib.NcOp import NcFileIO
from lib.dataprocess import datarepair
from lib.dataprocess import filter
import scipy.stats as stats
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from concurrent.futures import ProcessPoolExecutor  # 多进程并行处理库

def plot(data_out_dirname, resolution, lable, grid_lon2d, grid_lat2d, grid_data):
    """可视化并保存插值结果图片
    Args:
        data_out_dirname: 输出目录路径
        resolution: 网格分辨率
        lable: 数据标签名称（用于标题和文件名）
        grid_lon2d: 二维经度网格
        grid_lat2d: 二维纬度网格
        grid_data: 待可视化的二维网格数据
    """
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

        #绘制海洋数据
        # mask = not np.isnan(grid_data)
        # val_mx = int(grid_data[mask].max())
        # val_mi = int(grid_data[mask].min())

    contour = ax.contourf(
            grid_lon2d,
            grid_lat2d,
            grid_data,
            transform=ccrs.PlateCarree(),
            cmap='turbo',
            levels=256,
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
            cfeature.COASTLINE.with_scale('110m'),
            edgecolor='gray',
            linewidth=0.5,
            zorder=2  # 显示在最顶层
        )

        # 添加颜色条
    cbar = plt.colorbar(contour, ax=ax, shrink=0.6)
    cbar.set_label(lable)

        # 添加标题和网格
    plt.title(f'Interpolated {lable} Data (Resolution: {resolution}°)')
    ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)

    output_path = os.path.join(data_out_dirname, f'{lable}_interpolation.png')
    plt.savefig(output_path, dpi=900, bbox_inches='tight')
    plt.close()

def process_label(args):
    """并行处理单个数据标签的核心函数
    Args:
        args: 包含以下参数的元组:
            - lable: 数据标签名称
            - data_array: 原始数据数组
            - points: 插值坐标点集合
            - grid_lon2d: 目标经度网格
            - grid_lat2d: 目标纬度网格
            - intermathord: 插值方法
            - data_out_dirname: 输出目录
            - resolution: 网格分辨率
    
    Returns:
        tuple: (数据标签名称, 处理后的网格数据)
    """
    # 解包传入的参数
    lable, data_array, points, grid_lon2d, grid_lat2d, intermathord, data_out_dirname, resolution = args
    
    # 根据标签类型选择插值方法
    if lable != 'surface_type':
        method = intermathord  # 大多数数据用线性插值
    else:
        method = 'nearest'    # 表面类型数据用最近邻插值
    
    # 执行网格插值
    grid_data = griddata(
        points, data_array,
        (grid_lon2d, grid_lat2d),
        method=method,
        fill_value=np.nan
    )
    
    # 应用滑动窗口滤波
    grid_data = filter.sliding_window_filter(grid_data, 3, 'gaussian')
    
    # 对表面类型数据进行取整处理
    if lable == 'surface_type':
        grid_data = np.ceil(grid_data)
    
    # 生成并保存可视化结果
    plot(data_out_dirname, resolution, lable, grid_lon2d, grid_lat2d, grid_data)
    
    return (lable, grid_data)

if __name__ == '__main__':
    # 主程序入口（必须的Windows多进程保护）
    # ------------------ 初始化配置 ------------------
    file_rootdir = r"D:\Download\satellite\HY-2B"  # 原始数据根目录
    data_out_dirname = 'dataresult'                # 输出目录名称
    resolution = 1                                 # 网格分辨率（单位：度）
    intermathord = 'linear'                        # 默认插值方法
    
    # ------------------ 目录准备 ------------------
    dirs = os.listdir(file_rootdir)                # 获取所有周期目录
    datarepair.create_folder(data_out_dirname)     # 创建输出根目录

    # ------------------ 处理每个周期 ------------------
    for cycle in dirs:
        # 构造当前周期的数据路径
        file_datadir = os.path.join(file_rootdir, cycle)
        
        # 读取NetCDF文件数据
        nc_filedata = NcFileIO.read_netcdf4(file_datadir, 'r')
        
        # ------------------ 数据预处理 ------------------
        # 初始化数据容器
        data_arrays = { 
            'lon': [], 'lat': [], 'surface_type': [], 'mean_sea_surface': [],
            'range_ku': [], 'swh_ku': [], 'sig0_ku': [], 'alt': [],
            'wind_speed_model_u': [], 'wind_speed_model_v': [], 'wind_speed_alt': [],
            'wind_speed_rad': [], 'agc_ku': [], 'agc_numval_ku': [],
            'rad_water_vapor': [], 'off_nadir_angle_wf_ku': [], 'geoid': []
        }
        data_lable_list = list(data_arrays.keys())
        
        # 合并多个文件的数据
        for nc_data in nc_filedata:
            for lable in data_lable_list:
                data_arrays[lable].append(nc_data.variables[lable][:])
        
        # 拼接多维数组
        for lable in data_lable_list:
            data_arrays[lable] = np.concatenate(data_arrays[lable])
        
        # ------------------ 数据质量控制 ------------------
        qual = datarepair.qul_control(data_arrays, data_lable_list)
        mask = (data_arrays['mean_sea_surface'] != 2) & (data_arrays['mean_sea_surface'] != 3) & qual
        for lable in data_lable_list:
            data_arrays[lable] = data_arrays[lable][mask]  # 应用质量掩码
        
        # 经度坐标修正（180°转-180°表示）
        data_arrays['lon'] = np.where(
            data_arrays['lon'] > 180,
            data_arrays['lon'] - 360,
            data_arrays['lon']
        )
        
        # ------------------ 网格生成 ------------------
        # 计算坐标范围
        lon_min, lon_max = np.floor(data_arrays['lon'].min()), np.ceil(data_arrays['lon'].max())
        lat_min, lat_max = np.floor(data_arrays['lat'].min()), np.ceil(data_arrays['lat'].max())
        
        # 生成一维网格坐标
        grid_lon = np.arange(lon_min, lon_max + resolution, resolution)
        grid_lat = np.arange(lat_min, lat_max + resolution, resolution)
        
        # 生成二维网格矩阵
        grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
        
        # 构造插值点集合
        points = np.column_stack((data_arrays['lon'], data_arrays['lat']))
        
        # ------------------ 并行处理部分 ------------------
        # 创建当前周期的输出目录
        cycle_out_dir = os.path.join(data_out_dirname, cycle)
        datarepair.create_folder(cycle_out_dir)
        
        # 准备多进程参数列表
        args_list = []
        for lable in data_lable_list:
            args = (
                lable, data_arrays[lable], points, 
                grid_lon2d, grid_lat2d, intermathord,
                cycle_out_dir, resolution
            )
            args_list.append(args)  # 每个标签对应一个参数包
        
        # 使用进程池并行处理
        gridded_data = {}
        with ProcessPoolExecutor() as executor:
            # 提交所有任务，map保持输入输出顺序一致
            results = executor.map(process_label, args_list)
            
            # 收集处理结果
            for result in results:
                lable, grid_data = result
                gridded_data[lable] = grid_data
        
        # ------------------ 结果输出 ------------------
        output_path = os.path.join(cycle_out_dir, f"merged_grid_{cycle}.nc")
        datarepair.export_to_netcdf(gridded_data, grid_lon, grid_lat, output_path, resolution)
        print("YR巨献！！！")