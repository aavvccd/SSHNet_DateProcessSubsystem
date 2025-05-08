"""
__init__.py - 数据分析包初始化文件
此文件用于初始化数据分析包，包含数据预处理、数据分析和NetCDF文件I/O操作的模块。通过包含此文件，Python将目录视为包含模块的包，可以导入和作为一个单元使用。
属性：
    DataAnalyse (模块)：数据分析功能的模块。
    NcFileIO (模块)：NetCDF文件输入/输出操作的模块。
    PreProcessing (模块)：数据预处理功能的模块。
示例：
    >>> from NcOp import DataAnalyse, NcFileIO, PreProcessing
    >>> # 使用DataAnalyse模块中的函数
    >>> # 使用NcFileIO模块中的函数
    >>> # 使用PreProcessing模块中的函数
"""

# 定义包的公共接口
__all__ = ["DataAnalyse", "NcFileIO", "PreProcessing"]