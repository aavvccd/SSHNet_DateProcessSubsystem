import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import datetime

from holoviews import output


class WaveformReshaper:
    def __init__(self, data_shape=2813, target_hz=20, samples_per_hz=128):
        """
        波形数据三维化处理器

        参数:
            target_hz: 目标频率数量 (20Hz)
            samples_per_hz: 每个Hz的采样点数 (128)
        """
        self.target_hz = target_hz
        self.samples_per_hz = samples_per_hz
        self.expected_2d_shape = (data_shape, target_hz * samples_per_hz)  # (2813, 2560)
        self.expected_3d_shape = (data_shape, target_hz, samples_per_hz)  # (2813, 20, 128)

    def validate_input(self, data_2d):
        """验证输入数据形状"""
        if data_2d.shape != self.expected_2d_shape:
            raise ValueError(f"输入数据形状需为 {self.expected_2d_shape}，当前形状为 {data_2d.shape}")

    def reshape_to_3d(self, data_2d):
        """
        执行三维化转换

        参数:
            data_2d: 原始二维数组 (2813, 2560)

        返回:
            三维数组 (2813, 20, 128)，结构说明:
            - 第一维度: 样本索引 (0-2812)
            - 第二维度: 频率索引 (0-19 对应 1-20Hz)
            - 第三维度: 采样点索引 (0-127)
        """
        self.validate_input(data_2d)

        # 三维化重塑
        data_3d = data_2d.reshape((
            self.expected_3d_shape[0],  # 2813
            self.target_hz,  # 20
            self.samples_per_hz  # 128
        ))

        # 验证输出形状
        if data_3d.shape != self.expected_3d_shape:
            raise RuntimeError(f"三维化失败，输出形状为 {data_3d.shape}")

        return data_3d

    def get_frequency_slice(self, data_3d, hz_index):
        """
        获取指定频率的所有数据

        参数:
            hz_index: 频率索引 (0-19)

        返回:
            (2813, 128) 的二维数组，包含指定频率的所有样本数据
        """
        if not 0 <= hz_index < self.target_hz:
            raise ValueError(f"频率索引需在 0-{self.target_hz - 1} 范围内")
        return data_3d[:, hz_index, :]

    def plot_frequency_waveform(self, data_reshaped, sample_idx=0, hz_idx=0):
        """绘制指定样本和频率的波形"""
        plt.figure(figsize=(10, 4))
        plt.plot(data_reshaped[sample_idx, hz_idx])
        plt.title(f"sample {sample_idx} - {hz_idx + 1}Hz wave (128 Sampling points)")
        plt.xlabel("Sample point sequence number")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def import_to_nc(self, out_path,reshaped_data,time):
        output_path = "waveform_3d_netcdf4.nc"
        output_path = os.path.join(out_path,output_path)
        ds = nc.Dataset(output_path, 'w', format='NETCDF4')


        ds.title = "20Hz三维波形数据集"
        ds.description = "包含20个频率成分的波形数据，每个频率含128个采样点"
        ds.author = 'yr'
        ds.institution = 'nn'
        ds.source = "Processed from raw waveforms_20hz_ku data"

        # ================== 定义维度 ==================
        time_dim = ds.createDimension('time', time.shape[0])  # 2813个时间点
        freq_dim = ds.createDimension('frequency', reshaped_data.shape[1])  # 20个频率
        sample_dim = ds.createDimension('sample_point', reshaped_data.shape[2])  # 128个采样点

        time_var = ds.createVariable('time', np.float32, ('time',))
        time_var.units = f"days since 2000-1-1 00:00:00.0"
        time_var.calendar = "standard"
        time_var.long_name = "Time"
        time_var[:] = time

        # 频率坐标 (1-20Hz)
        freq_var = ds.createVariable('frequency', np.float32, ('frequency',))
        freq_var.units = "Hz"
        freq_var.long_name = "Frequency component"
        freq_var[:] = np.arange(1, 21)  # 1到20Hz

        # 采样点坐标
        sample_var = ds.createVariable('sample_point', np.float32, ('sample_point',))
        sample_var.units = "index"
        sample_var.long_name = "Sampling point index within frequency"
        sample_var[:] = np.arange(128)

        waveform_var = ds.createVariable(
            'waveform',
            np.float32,
            ('time', 'frequency', 'sample_point'),
            zlib=True,  # 启用压缩
            complevel=4,  # 压缩级别(1-9)
            chunksizes=(100, 5, 32)  # 优化读取性能的块大小
        )

        waveform_var.units = "V"
        waveform_var.long_name = "Waveform amplitude"
        waveform_var.resolution = "16-bit ADC"
        waveform_var[:] = reshaped_data  # 写入数据

        # ================== 关闭文件 ==================
        ds.close()
        print(f"成功生成NetCDF4文件: {output_path}")

