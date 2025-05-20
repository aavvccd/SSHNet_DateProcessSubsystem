import os
import gc
import netCDF4 as nc
import numpy as np
from lib.NcOp import NcFileIO
from lib.dataprocess import datarepair
from lib.dataprocess import oldfilter
import scipy.stats as stats
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from lib.dataprocess.waveform import WaveformReshaper



file_rootdir = r"E:\dlproject\documents\ncdata"  # 文件目录
data_out_root_dirname = r'E:\github\dataresult'  # 输出目录
resolution = 1  # 生成网格的分辨率
intermathord = 'linear'
  # 创建存储网格数据的字典
dirs = os.listdir(file_rootdir)  # 获取文件夹中的所有cycle
datarepair.create_folder(data_out_root_dirname)





for i in dirs:
    gridded_data = {}
    file_datadir = os.path.join(file_rootdir, i)
    nc_filedata = NcFileIO.read_netcdf4(file_datadir, 'r')

    # cydle子目录创建
    data_out_dirname = os.path.join(data_out_root_dirname, i)
    datarepair.create_folder(data_out_dirname)


    data_arrays = {  # 数据标签
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
        'agc_numval_ku': [],
        'rad_water_vapor': [],
        'off_nadir_angle_wf_ku': [],
        'geoid': [],
        # 'waveforms_20hz_ku': []  # 二维数据可用嵌套列表（如 [[v1, v2, ...], ...]）
    }

    time = []
    waveforms_20hz_ku = []

    qual_data_arrays = {
        'qual_alt_1hz_range_ku': [],
        'qual_alt_1hz_swh_ku': [],
        'qual_alt_1hz_sig0_ku': [],
        # 可添加更多
    }
    data_lable_list = list(data_arrays.keys())
    qual_lable_list = list(qual_data_arrays.keys())

    for nc_data in nc_filedata:
        waveforms_20hz_ku.append(nc_data.variables['waveforms_20hz_ku'][:])
        time.append(nc_data.variables['time'][:])
        for lable in data_lable_list:
            data_arrays[lable].append(nc_data.variables[lable][:])
        for lable in qual_lable_list:
            qual_data_arrays[lable].append(nc_data.variables[lable][:])
    for lable in data_lable_list:
        data_arrays[lable] = np.concatenate(data_arrays[lable])
    for lable in qual_lable_list:
        qual_data_arrays[lable] = np.concatenate(qual_data_arrays[lable])
        qual_data_arrays[lable] = np.where(qual_data_arrays[lable]==1,False,True)
    waveforms_20hz_ku = np.concatenate(waveforms_20hz_ku)
    time = np.concatenate(time)


    print('处理波形数据')
    # 初始化波形处理器
    reshaper = WaveformReshaper(waveforms_20hz_ku.shape[0],20,128)
    reshaped_data = reshaper.reshape_to_3d(waveforms_20hz_ku)
    # print("三维数据形状:", reshaped_data.shape)
    reshaper.import_to_nc(data_out_dirname,reshaped_data,time) #导出波形到文件
    print('导出波形数据')
    del reshaper
    del reshaped_data


    lon_all = data_arrays['lon']
    lat_all = data_arrays['lat']
    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
    lon_min, lon_max = np.floor(lon_all.min()), np.ceil(lon_all.max())
    lat_min, lat_max = np.floor(lat_all.min()), np.ceil(lat_all.max())
    grid_lon = np.arange(lon_min, lon_max + resolution, resolution)
    grid_lat = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)



    for lable in data_lable_list:
        if (lable == 'lat') or (lable == 'lon'):
            continue
        base_mask = (data_arrays['surface_type'] != 2) & (data_arrays['surface_type'] != 3)
        if (lable != 'surface_type'):
            qual = datarepair.qul_control(data_arrays[lable], lable, qual_lable_list, qual_data_arrays)
            mask = base_mask & qual
        else:
            mask = base_mask

        mask_lon = lon_all[mask]
        mask_lat = lat_all[mask]

        current_data = data_arrays[lable][mask]
        points = np.column_stack((mask_lon, mask_lat))

        if lable != 'surface_type':
            grid_data = griddata(
                points, current_data,
                (grid_lon2d, grid_lat2d),
                method=intermathord,
                fill_value=np.nan
            )
        else:
            grid_data = griddata(
                points, current_data,
                (grid_lon2d, grid_lat2d),
                method='nearest',
                fill_value=np.nan
            )
        # 存储插值结果到字典
        grid_data = oldfilter.sliding_window_filter(grid_data, 3, 'gaussian') #滤波
        if lable == 'surface_type':
            grid_data = np.ceil(grid_data)
        gridded_data[lable] = grid_data
        datarepair.plot(data_out_dirname, resolution, lable, grid_lon2d, grid_lat2d, grid_data)

    #生成文件名
    output_path = os.path.join(
        data_out_dirname,
        f"merged_grid_{i}.nc"  # 示例: merged_grid_cycle001.nc
    )
    datarepair.export_to_netcdf(gridded_data, grid_lon, grid_lat, output_path, resolution)
    del nc_filedata  # 删除文件数据对象
    del data_arrays  # 删除原始数据集合
    del qual_data_arrays  # 删除质量控制数据
    del waveforms_20hz_ku  # 删除波形原始数据
    del time  # 删除时间数据
    del grid_lon, grid_lat, grid_lon2d, grid_lat2d  # 删除网格坐标
    del gridded_data  # 删除插值后的网格数据
    del data_out_dirname, file_datadir #重置目录
    gc.collect()  # 强制垃圾回收
    # print('YR倾情巨献!!!')
    print(f'Cycle {i} 处理完成，内存已清理')