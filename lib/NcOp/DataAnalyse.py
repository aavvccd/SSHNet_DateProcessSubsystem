import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def calculate_time_series_statistics(df, variable_name, stats=None):
    """
    计算DataFrame中指定列的时间序列统计量。
    参数:
    df (pd.DataFrame): DataFrame对象，其中时间列已转换为datetime类型。
    variable_name (str): 需要计算统计量的列名称。
    stats (list): 需要计算的统计量列表，默认为['mean', 'std']。
    返回:
    dict: 包含计算结果的字典。
    """
    if stats is None:
        stats = ['mean', 'std']
    stats_results = {}
    if 'mean' in stats:
        stats_results['mean'] = df[variable_name].mean()
    if 'std' in stats:
        stats_results['std'] = df[variable_name].std()
    return stats_results

def perform_trend_analysis(df, variable_name, time_period):
    """
    对DataFrame中指定列进行趋势分析。
    参数:
    df (pd.DataFrame): DataFrame对象，其中时间列已转换为datetime类型。
    variable_name (str): 需要进行趋势分析的列名称。
    time_period: 时间周期，用于筛选数据，格式为[开始时间, 结束时间]。
    返回:
    numpy.poly1d: 趋势线的多项式对象。
    """
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    if time_period:
        df = df[(df['time'] >= time_period[0]) & (df['time'] <= time_period[1])]
    return np.polyfit(df.index, df[variable_name], 1)


def perform_correlation_analysis(df, variable_name1, variable_name2):
    """
    对DataFrame中两个指定列进行相关性分析。
    参数:
    df (pd.DataFrame): DataFrame对象。
    variable_name1 (str): 第一个列名称。
    variable_name2 (str): 第二个列名称。
    返回:
    float: 相关系数。
    """
    return np.corrcoef(df[variable_name1], df[variable_name2])[0, 1]

def plot_variable_time_series(df, variable_name, time_range=None):
    """
    绘制DataFrame中指定列的时间序列图。
    参数:
    df (pd.DataFrame): DataFrame对象，其中时间列已转换为datetime类型。
    variable_name (str): 需要绘制时间序列的列名称。
    time_range: 时间范围，用于筛选数据，格式为[开始时间, 结束时间]。
    """
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    if time_range:
        df = df[(df['time'] >= time_range[0]) & (df['time'] <= time_range[1])]
    plt.plot(df['time'], df[variable_name])
    plt.title(variable_name)
    plt.xlabel('Time')
    plt.ylabel(variable_name)
    plt.show()
