import netCDF4 as nc
import os
import pandas as pd
from tabulate import tabulate
import numpy as np
from datetime import datetime, timedelta

"""
检查给定路径是否为文件夹。
参数:
path (str): 需要检查的路径。
返回:
bool: 如果路径存在且为文件夹，则返回True，否则返回False。
"""
def is_folder(path):
    if os.path.exists(path):
        return os.path.isdir(path)
    else:
        return False


"""
读取单个或多个NetCDF文件，并返回一个包含数据集对象的列表。
参数:
file_path (str): NetCDF文件的路径或包含NetCDF文件的文件夹路径。
mode (str): 打开NetCDF文件的模式（例如'r'为只读）。
返回:
list: 包含NetCDF数据集对象的列表。
"""

def read_netcdf4(file_path, mode):
    if is_folder(file_path):
        all_data = []
        for file in os.listdir(file_path):
            path = os.path.join(file_path, file)
            # print(path)
            if path.endswith('.nc'):
                data = nc.Dataset(path, mode)
                all_data.append(data)
            elif is_folder(path):
                sub_data = read_netcdf4(path, mode)
                all_data.extend(sub_data)
        return all_data
    elif file_path.endswith('.nc'):
        return nc.Dataset(file_path, mode)
    else:
        return []

"""
列出NetCDF数据集中的所有变量及其大小。
参数:
data: NetCDF数据集对象或包含多个数据集对象的列表。
"""

def list_variables(data):
    table_data = []
    for variable in data.variables:
        table_data.append([variable, len(data[variable][:])])
    print(tabulate(table_data, headers=['Variable', 'Size'], tablefmt='tabs'))



"""
描述NetCDF数据集中指定变量的属性，包括数据类型、非空值计数、最小值、最大值、平均值。
参数:
data: NetCDF数据集对象。
decimal_places (int): 浮点数显示的小数位数，默认为6。
"""

def describe_variable(data, decimal_places=6):
    table_data = []
    for variable in data.variables:
        val_min = np.min(data[variable][:])
        val_max = np.max(data[variable][:])
        val_mean = np.mean(data[variable][:])
        data_type = str(data[variable][:].dtype)
        table_data.append([
            variable,
            data[variable].shape,
            np.count_nonzero(data[variable][:]),
            f"{val_min:.{decimal_places}f}",
            f"{val_max:.{decimal_places}f}",
            f"{val_mean:.{decimal_places}f}",
            data_type
        ])
    print(tabulate(table_data, headers=['Variable','Shape', 'Not-null', 'Min', 'Max', 'Mean','Type'], tablefmt='fancy_grid', floatfmt=f".{decimal_places}f"))


"""
从NetCDF数据集中获取指定变量的数据。
参数:
data: NetCDF数据集对象。
variable_name (str): 要获取的变量名。
返回:
numpy.ndarray: 变量的数据，如果变量不存在则返回None。
"""

def get_variable(data, variable_name):
    if variable_name in data.variables:
        return data[variable_name][:]
    else:
        return None
