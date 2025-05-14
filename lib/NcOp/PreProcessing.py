import pandas as pd
from scipy import interpolate
import numpy as np
import warnings
# 禁用特定弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cgitb")

def apply_quality_control(df, criteria):
    """
    对DataFrame中的列应用质量控制标准。
    参数:
    df: DataFrame对象。
    criteria: 一个字典，包含列名作为键，以及相应的质量控制标准作为值。
              每个质量控制标准也是一个字典，可以包含：
              - 'min_value': 列值的最小阈值。
              - 'max_value': 列值的最大阈值。
              - 'std_dev_threshold': 基于标准差的异常值阈值。
              - 'median_filter': 中位数滤波窗口大小。
              - 'percentile_threshold': 基于百分位数的异常值阈值。
              - 'interpolate': 是否对缺失数据进行插值。
              - 'data_type': 预期的数据类型。
    """
    for column, crit in criteria.items():
        if column in df.columns:  # 检查列是否存在于DataFrame中
            series = df[column]
            # 应用最小值和最大值标准
            if 'min_value' in crit and 'max_value' in crit:
                series = series.where((series >= crit['min_value']) & (series <= crit['max_value']), pd.NA)

            # 应用基于标准差的异常值标准
            if 'std_dev_threshold' in crit:
                mean_val = series.mean()
                std_val = series.std()
                series = series.where(np.abs(series - mean_val) < crit['std_dev_threshold'] * std_val, pd.NA)

            # 应用中位数滤波
            if 'median_filter' in crit:
                window_size = crit['median_filter']
                for i in range(window_size, len(series) - window_size):
                    series[i] = series[i - window_size:i + window_size + 1].median()

            # 应用基于百分位数的异常值标准
            if 'percentile_threshold' in crit:
                lower, upper = series.quantile([crit['percentile_threshold'] / 100, 1 - crit['percentile_threshold'] / 100])
                series = series.where((series >= lower) & (series <= upper), pd.NA)

            # 对缺失数据进行插值
            if 'interpolate' in crit and crit['interpolate']:
                series = series.interpolate()

            # 检查数据类型
            if 'data_type' in crit:
                if not np.issubdtype(series.dtype, crit['data_type']):
                    raise ValueError(f"数据类型不匹配，预期为 {crit['data_type']}，实际为 {series.dtype}")

            df[column] = series
    return df

def fill_missing_values(df, variable_name, method='nearest'):
    """
    填充DataFrame中指定列的缺失值。

    参数:
    df: DataFrame对象。
    variable_name: 需要填充缺失值的列名称。
    method: 用于填充缺失值的方法，默认为'nearest'。
            可选方法包括：
            - 'nearest': 使用最近邻填充。
            - 'linear': 使用线性插值。
            - 'polynomial': 使用多项式插值。
            - 'spline': 使用样条插值。
            - 'time_series': 使用时间序列插值（适用于时间序列数据）。
    返回:
    填充后的列数据。
    """
    series = df[variable_name]

    if method == 'nearest':
        # 最近邻填充
        df[variable_name] = series.fillna(method='ffill').fillna(method='bfill')
    elif method == 'linear':
        # 线性插值
        mask = ~series.isna()
        x_valid = np.arange(len(series))[mask]
        y_valid = series[mask]
        x_missing = np.arange(len(series))[series.isna()]
        interp = interpolate.interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
        df[variable_name] = interp(df.index)
    elif method == 'polynomial':
        # 多项式插值
        mask = ~series.isna()
        x_valid = np.arange(len(series))[mask]
        y_valid = series[mask]
        x_missing = np.arange(len(series))[series.isna()]
        interp = interpolate.interp1d(x_valid, y_valid, kind='polynomial', fill_value="extrapolate")
        df[variable_name] = interp(df.index)
    elif method == 'spline':
        # 样条插值
        mask = ~series.isna()
        x_valid = np.arange(len(series))[mask]
        y_valid = series[mask]
        x_missing = np.arange(len(series))[series.isna()]
        interp = interpolate.interp1d(x_valid, y_valid, kind='spline', fill_value="extrapolate")
        df[variable_name] = interp(df.index)
    elif method == 'time_series':
        # 时间序列插值（需要时间索引）
        time_index = df.index  # 假设索引是时间索引
        mask = ~series.isna()
        x_valid = time_index[mask]
        y_valid = series[mask]
        x_missing = time_index[series.isna()]
        interp = interpolate.interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
        df[variable_name] = interp(x_missing)
    else:
        raise ValueError("Unsupported method. Choose 'nearest', 'linear', 'polynomial', 'spline', or 'time_series'.")

    return df


def remove_outliers(df, variable_name, method='z-score'):
    """
    从DataFrame中指定列移除异常值，并将异常值替换为NaN。
    返回一个新的DataFrame，原DataFrame不被修改。

    参数:
    df (pd.DataFrame): 输入的DataFrame对象。
    variable_name (str): 需要移除异常值的列名称。
    method (str): 用于识别和移除异常值的方法，默认为'z-score'。
                  可选方法包括：
                  - 'z-score': 使用Z分数方法，移除距离均值超过3倍标准差的点。
                  - 'iqr': 使用四分位距方法，移除低于Q1-1.5*IQR或高于Q3+1.5*IQR的点。
                  - 'percentile': 使用百分位数方法，移除低于5%或高于95%百分位数的点。
                  - 'mad': 使用平均绝对偏差方法，移除与中位数偏差超过3倍MAD的点。

    返回:
    pd.DataFrame: 一个新的DataFrame，其中异常值被替换为NaN。
    """
    df_new = df.copy()  # 创建原DataFrame的副本
    series = df_new[variable_name]

    if method == 'z-score':
        z_scores = np.abs((series - series.mean()) / series.std())
        mask = z_scores < 3
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (series >= lower_bound) & (series <= upper_bound)
    elif method == 'percentile':
        lower_percentile = 5
        upper_percentile = 95
        lower_bound = series.quantile(lower_percentile / 100)
        upper_bound = series.quantile(upper_percentile / 100)
        mask = (series >= lower_bound) & (series <= upper_bound)
    elif method == 'mad':
        median = series.median()
        deviation = np.abs(series - median)
        mad = deviation.median()
        mask = np.abs(series - median) < 3 * mad
    else:
        raise ValueError("Unsupported method. Choose 'z-score', 'iqr', 'percentile', or 'mad'.")

    # 将异常值设置为NaN
    df_new.loc[~mask, variable_name] = np.nan

    return df_new


def drop_rows_with_na(df):
    """
    删除DataFrame中包含空值的行。
    参数:
    df (pd.DataFrame): 输入的DataFrame。
    返回:
    pd.DataFrame: 删除包含空值行后的DataFrame。
    """
    # 删除包含空值的行
    clean_df = df.dropna()
    return clean_df