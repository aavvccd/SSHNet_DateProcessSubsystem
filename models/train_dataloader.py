import numpy as np
import torch
from torch.utils.data import Dataset
from NcOp import NcFileIO as ncio
class OceanDataProcess(Dataset):
    def __init__(self,
                 merged_path,
                 ssh_path,
                 sst_path,
                 wind_path,
                 patch_size=16,
                 norm_params=None):
        self.patch_size = patch_size
        self.norm_params = norm_params if norm_params is not None else {}
        merged_data, merged_lon, merged_lat = self.load_merged_data(merged_path)
        ocean_mask = (merged_data[0] == 0)
        alt = merged_data[5]
        range_ku = merged_data[2]
        sat_ssh = alt - range_ku
        sat_ssh = sat_ssh / 100
        ssh_data, ssh_lat_mask = self.load_and_mask_data(ssh_path, "ssh", merged_lat)
        sst_data, sst_lat_mask = self.load_and_mask_data(sst_path, "sst", merged_lat)
        wind_data, wind_lat_mask = self.load_and_mask_data(wind_path, "wind", merged_lat)
        common_lat_mask = ssh_lat_mask & sst_lat_mask & wind_lat_mask
        common_lat_indices = np.where(common_lat_mask)[0]
        merged_common = merged_data[:, common_lat_indices, :]
        ocean_mask_common = ocean_mask[common_lat_indices, :]
        sat_ssh_common = sat_ssh[common_lat_indices, :]
        ssh_common = ssh_data[:, common_lat_indices, :]
        sst_common = sst_data[:, common_lat_indices, :]
        wind_common = wind_data[:, common_lat_indices, :]
        train_sat = np.concatenate([
            merged_common,
            sat_ssh_common[np.newaxis, :, :],
            sst_common,
            wind_common
        ], axis=0)
        train_real = ssh_common
        if not self.norm_params:
            self.norm_params = {
                'sat_means': [],
                'sat_stds': [],
            }
            for i in range(15):
                channel_data = merged_common[i]
                valid_data = channel_data[ocean_mask_common]
                if len(valid_data) > 0:
                    mean_val = np.nanmean(valid_data)
                    std_val = np.nanstd(valid_data)

                    if std_val < 1e-9:
                        std_val = 1.0
                else:
                    mean_val = 0.0
                    std_val = 1.0
                merged_common[i] = (channel_data - mean_val) / std_val
                self.norm_params['sat_means'].append(mean_val)
                self.norm_params['sat_stds'].append(std_val)
            valid_sst = sst_common[0][ocean_mask_common]
            if len(valid_sst) > 0:
                sst_mean = np.nanmean(valid_sst)
                sst_std = np.nanstd(valid_sst)
                if sst_std < 1e-9:
                    sst_std = 1.0
            else:
                sst_mean = 0.0
                sst_std = 1.0
            sst_common[0] = (sst_common[0] - sst_mean) / sst_std
            self.norm_params['sat_means'].append(sst_mean)
            self.norm_params['sat_stds'].append(sst_std)
            for i in range(2):  # u和v分量
                wind_channel = wind_common[i]
                valid_wind = wind_channel[ocean_mask_common]
                if len(valid_wind) > 0:
                    wind_mean = np.nanmean(valid_wind)
                    wind_std = np.nanstd(valid_wind)
                    if wind_std < 1e-9:
                        wind_std = 1.0
                else:
                    wind_mean = 0.0
                    wind_std = 1.0
                wind_common[i] = (wind_channel - wind_mean) / wind_std
                self.norm_params['sat_means'].append(wind_mean)
                self.norm_params['sat_stds'].append(wind_std)
        else:
            for i in range(15):
                mean_val = self.norm_params['sat_means'][i]
                std_val = self.norm_params['sat_stds'][i]
                merged_common[i] = (merged_common[i] - mean_val) / std_val
            sst_mean = self.norm_params['sat_means'][15]
            sst_std = self.norm_params['sat_stds'][15]
            sst_common[0] = (sst_common[0] - sst_mean) / sst_std
            for i in range(2):
                wind_mean = self.norm_params['sat_means'][16 + i]
                wind_std = self.norm_params['sat_stds'][16 + i]
                wind_common[i] = (wind_common[i] - wind_mean) / wind_std

        train_sat = np.concatenate([
            merged_common,
            sat_ssh_common[np.newaxis, :, :],
            sst_common,
            wind_common
        ], axis=0)
        train_sat = np.nan_to_num(train_sat, nan=0.0)
        train_real = np.nan_to_num(train_real, nan=0.0)
        self.ocean_mask = ocean_mask_common
        self.train_real_patches, self.train_sat_patches, self.mask_patches = self.create_ocean_patches(
            train_real, train_sat, ocean_mask_common
        )
        # print(f"纬度交集范围: {common_lat_indices[0]}-{common_lat_indices[-1]} "
        #       f"({len(common_lat_indices)}个纬度点)")
        # print(f"创建了 {len(self)} 个包含海洋的 {patch_size}x{patch_size} 大小的图像块")
        # print(f"输入特征维度: {train_sat.shape} → 输出真值维度: {train_real.shape}")

    def load_merged_data(self, path):
        """加载merged_grid_0110.nc数据"""
        with ncio.read_netcdf4(path, 'r') as nc:
            lon = nc.variables['lon'][:]
            lat = nc.variables['lat'][:]

            variables = [
                'surface_type', 'mean_sea_surface', 'range_ku', 'swh_ku',
                'sig0_ku', 'alt', 'wind_speed_model_u', 'wind_speed_model_v',
                'wind_speed_alt', 'wind_speed_rad', 'agc_ku', 'agc_numval_ku',
                'rad_water_vapor', 'off_nadir_angle_wf_ku', 'geoid'
            ]
            data = np.empty((len(variables), len(lat), len(lon)), dtype=np.float32)
            for i, var in enumerate(variables):
                data[i] = nc.variables[var][:]

            return data, lon, lat
    def load_and_mask_data(self, path, data_type, base_lat):
        with ncio.read_netcdf4(path, 'r') as nc:
            if data_type == 'ssh' or data_type == 'sst':
                data_lat = nc.variables['latitude'][:]
                data_var = nc.variables['ssh' if data_type == 'ssh' else 'sst'][:]
            else:  # wind
                data_lat = nc.variables['lat'][:]
                u_wind = nc.variables['eastward_wind'][:]
                v_wind = nc.variables['northward_wind'][:]
                data_var = np.stack([u_wind, v_wind], axis=0)
            if len(data_var.shape) == 2:
                data_var = data_var[np.newaxis, :, :]
            min_lat = max(base_lat.min(), data_lat.min())
            max_lat = min(base_lat.max(), data_lat.max())
            lat_mask = (base_lat >= min_lat) & (base_lat <= max_lat)
            return data_var, lat_mask
    def create_ocean_patches(self, real_data, sat_data, ocean_mask):
        c_real, h, w = real_data.shape
        c_sat, _, _ = sat_data.shape
        num_h = h // self.patch_size
        num_w = w // self.patch_size
        real_patches = []
        sat_patches = []
        mask_patches = []
        for i in range(num_h):
            for j in range(num_w):
                h_start = i * self.patch_size
                w_start = j * self.patch_size
                patch_mask = ocean_mask[h_start:h_start + self.patch_size,
                             w_start:w_start + self.patch_size]

                if np.mean(patch_mask) > 0.1:
                    real_patch = real_data[:, h_start:h_start + self.patch_size,
                                 w_start:w_start + self.patch_size]
                    sat_patch = sat_data[:, h_start:h_start + self.patch_size,
                                w_start:w_start + self.patch_size]

                    real_patches.append(real_patch)
                    sat_patches.append(sat_patch)
                    mask_patches.append(patch_mask)
        real_tensor = torch.tensor(np.array(real_patches), dtype=torch.float32)
        sat_tensor = torch.tensor(np.array(sat_patches), dtype=torch.float32)
        mask_tensor = torch.tensor(np.array(mask_patches), dtype=torch.bool)

        return real_tensor, sat_tensor, mask_tensor

    def get_norm_params(self):
        return self.norm_params

    def __getitem__(self, index):
        return self.train_sat_patches[index], self.train_real_patches[index], self.mask_patches[index]

    def __len__(self):
        return self.train_sat_patches.shape[0]
