import numpy as np
def sliding_window_filter(matrix,window_size,filter_type='mean'):
    rows,cols = matrix.shape
    filteredMatrix = np.full([rows,cols],np.nan)
    halfWindowSize = int(np.floor(window_size/2))
    for i in range(0,rows):
        for j in range(0,cols):
            r1 = max(1,i-halfWindowSize)
            r2 = min(rows,i+halfWindowSize)
            c1 = max(1,j-halfWindowSize)
            c2 = min(cols,j+halfWindowSize)
            windows = matrix[r1:r2+1,c1:c2+1]
            windows = np.resize(windows,[window_size,window_size])
            if not np.isnan(windows).all():
                if filter_type == 'mean':
                    filteredMatrix[i][j] = np.mean(windows)
                elif filter_type == 'median':
                    filteredMatrix[i][j] = np.median(windows)
                elif filter_type == 'gaussian':
                    sigma = 1
                    x = np.arange(-halfWindowSize, halfWindowSize + 1)
                    y = np.arange(-halfWindowSize, halfWindowSize + 1)
                    X, Y = np.meshgrid(x, y)
                    G = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
                    G = G / np.sum(G)
                    filteredMatrix[i, j] = np.sum(windows * G)
                elif filter_type == 'max':
                    filteredMatrix[i,j] = np.max(windows)
                elif filter_type == 'min':
                    filteredMatrix[i,j] = np.min(windows)
                elif filter_type == 'laplacian':
                    LaplacianKernel = [[1,-2,1],[-2,4,-2],[1,-2,1]]
                    filteredMatrix[i][j] = np.sum(windows*LaplacianKernel)
                elif filter_type == 'rms':
                    filteredMatrix[i][j] = np.sqrt(np.mean(windows*windows))
                elif filter_type == 'bilateral':
                    sigma_spatial = 1
                    sigma_intensity = 1
                    x = np.arange(-halfWindowSize, halfWindowSize + 1)
                    y = np.arange(-halfWindowSize, halfWindowSize + 1)
                    X, Y = np.meshgrid(x, y)
                    spatialWeights = np.exp(-(X**2 + Y**2) / (2 * sigma_spatial**2))
                    intensityWeights = np.exp(-(windows - matrix[i][j])**2 / (2 * sigma_intensity**2))
                    bilateralWeights = spatialWeights * intensityWeights
                    bilateralWeights = bilateralWeights / np.sum(bilateralWeights)
                    filteredMatrix[i][j] =np.sum(windows*bilateralWeights)
                else:
                    print('no type')
    return filteredMatrix

# if __name__ == '__main__':
#     x = [[0,0,0],[1,1,1],[1,1,1]]
#     x = np.array(x)
#     x = sliding_window_filter(x,3,'gaussian')
#     print(x)
#
#
#     data_arrays = { #数据标签
#         'lon': [],
#         'lat': [],
#         'surface_type': [],
#         'mean_sea_surface': [],
#         'range_ku': [],
#         'swh_ku': [],
#         'sig0_ku': [],
#         'alt': [],
#         'wind_speed_model_u': [],
#         'wind_speed_model_v': [],
#         'wind_speed_alt': [],
#         'wind_speed_rad': [],
#         'agc_ku': [],
#         'agc_numval_ku': [],  # 注意：原数据标签中此处缺少逗号，已修正
#         'rad_water_vapor': [],
#         'off_nadir_angle_wf_ku': [],
#         'geoid':[],
#         # 'waveforms_20hz_ku': []  # 二维数据可用嵌套列表（如 [[v1, v2, ...], ...]）
#     }