import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
def sliding_window_filter(matrix, window_size, filter_type='mean', **kwargs):
    """
    参数：
        matrix: 输入矩阵 (可能包含NaN值)
        window_size: 奇数窗口尺寸
        filter_type: 滤波器类型，支持以下类型：
            'mean' - 均值滤波
            'median' - 中值滤波
            'gaussian' - 高斯滤波
            'max'/'min' - 最大/最小值滤波
            'laplacian' - 拉普拉斯边缘检测
            'rms' - 均方根值
            'bilateral' - 双边滤波
            'sobel' - Sobel梯度滤波
            'anisotropic' - 各向异性扩散
            'morph_open' - 形态学开运算
            'morph_close' - 形态学闭运算
            'local_gradient' - 局部梯度幅值
            'ocean_ridge' - 海脊增强滤波
        **kwargs: 滤波器专用参数
    返回：
        滤波后的矩阵
    """
    # 输入校验
    rows, cols = matrix.shape
    if window_size % 2 == 0:
        raise ValueError("窗口尺寸必须为奇数")
    half_window = window_size // 2
    # 边缘填充
    pad_width = ((half_window, half_window), (half_window, half_window))
    padded_matrix = np.pad(matrix, pad_width, mode='reflect')  # 或使用 mode='reflect'

    # 生成滑动窗口视图
    windows = sliding_window_view(padded_matrix, (window_size, window_size))
    windows = windows.reshape(rows, cols, window_size, window_size)
    filtered = np.full((rows, cols), np.nan)

    # 公共参数处理
    sigma = kwargs.get('sigma', 1.0)
    iterations = kwargs.get('iterations', 1)
    struct_element = kwargs.get('struct_element', np.ones((3, 3)))
    # 滤波器分派
    if filter_type == 'mean':
        filtered = np.nanmean(windows, axis=(2, 3))
    elif filter_type == 'median':
        filtered = np.nanmedian(windows, axis=(2, 3))

    elif filter_type == 'gaussian':
        x = np.arange(-half_window, half_window + 1)
        X, Y = np.meshgrid(x, x)
        gaussian_kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        gaussian_kernel /= gaussian_kernel.sum()
        filtered = _apply_kernel(windows, gaussian_kernel)

    elif filter_type in ('max', 'min'):
        func = np.nanmax if filter_type == 'max' else np.nanmin
        filtered = func(windows, axis=(2, 3))

    elif filter_type == 'laplacian':
        laplacian_kernel = np.array([[1, -2, 1],
                                     [-2, 4, -2],
                                     [1, -2, 1]])
        if window_size != 3:
            raise ValueError("拉普拉斯滤波需要窗口尺寸为3")
        filtered = _apply_kernel(windows, laplacian_kernel)

    elif filter_type == 'rms':
        filtered = np.sqrt(np.nanmean(windows ** 2, axis=(2, 3)))

    elif filter_type == 'bilateral':
        filtered = _bilateral_filter(windows, half_window, sigma_s=sigma, sigma_r=0.1)

    elif filter_type == 'sobel':
        sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        filtered = _apply_kernel(windows, sobel_kernel)

    elif filter_type == 'anisotropic':
        filtered = _anisotropic_diffusion(matrix, iterations=iterations, kappa=50, gamma=0.1)

    elif filter_type in ('morph_open', 'morph_close'):
        func = ndimage.binary_opening if filter_type == 'morph_open' else ndimage.binary_closing
        filtered = func(np.where(np.isnan(matrix), 0, matrix), structure=struct_element)
        filtered = filtered.astype(float)
        filtered[np.isnan(matrix)] = np.nan

    elif filter_type == 'local_gradient':
        grad_x = _apply_kernel(windows, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8)
        grad_y = _apply_kernel(windows, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8)
        filtered = np.hypot(grad_x, grad_y)

    elif filter_type == 'ocean_ridge':
        # (组合拉普拉斯和高斯)
        gauss = sliding_window_filter(matrix, window_size, 'gaussian', sigma=sigma)
        laplacian = sliding_window_filter(gauss, 3, 'laplacian')
        filtered = np.clip(laplacian * 2, -1, 1)  # 增强线性特征

    else:
        raise ValueError(f"不支持的滤波类型: {filter_type}")

    return filtered


def _apply_kernel(windows, kernel):
    """通用卷积核应用函数"""
    kernel = kernel.reshape(1, 1, *kernel.shape)
    valid = ~np.isnan(windows)
    window_data = np.where(np.isnan(windows), 0, windows)
    result = np.sum(window_data * kernel, axis=(2, 3))
    valid_count = np.sum(valid * np.abs(kernel), axis=(2, 3))  # 权重有效性计数
    result[valid_count < 0.5] = np.nan  # 有效权重不足时返回NaN
    return result


def _bilateral_filter(windows, half_window, sigma_s=1.0, sigma_r=0.1):
    """双边滤波"""
    rows, cols = windows.shape[:2]
    filtered = np.zeros((rows, cols))

    # 预计算空间权重
    x = np.arange(-half_window, half_window + 1)
    X, Y = np.meshgrid(x, x)
    spatial_weights = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma_s ** 2))

    for i in range(rows):
        for j in range(cols):
            window = windows[i, j]
            center = window[half_window, half_window]
            if np.isnan(center):
                filtered[i, j] = np.nan
                continue

            # 强度权重
            intensity_weights = np.exp(-(window - center) ** 2 / (2 * sigma_r ** 2))
            weights = spatial_weights * intensity_weights
            weights[np.isnan(window)] = 0
            sum_weights = np.sum(weights)

            if sum_weights > 1e-6:
                filtered[i, j] = np.nansum(window * weights) / sum_weights
            else:
                filtered[i, j] = np.nan
    return filtered


def _anisotropic_diffusion(img, iterations=10, kappa=50, gamma=0.1):
    """各向异性扩散滤波"""
    img = img.copy()
    delta = np.zeros_like(img)

    for _ in range(iterations):
        grad_n = np.roll(img, -1, axis=0) - img
        grad_s = np.roll(img, 1, axis=0) - img
        grad_e = np.roll(img, -1, axis=1) - img
        grad_w = np.roll(img, 1, axis=1) - img

        # 计算扩散系数
        c_n = np.exp(-(grad_n / kappa) ** 2)
        c_s = np.exp(-(grad_s / kappa) ** 2)
        c_e = np.exp(-(grad_e / kappa) ** 2)
        c_w = np.exp(-(grad_w / kappa) ** 2)

        # 更新图像
        img += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)

    return img


if __name__ == '__main__':
    # 测试
    bathymetry = np.array([
        [np.nan, 200, 180, 200, np.nan],
        [210, 205, 190, 195, 200],
        [220, np.nan, 185, 190, 205],
        [215, 200, 195, 200, 210]
    ], dtype=float)

    # 使用滤波
    ridge_enhanced = sliding_window_filter(bathymetry, 3, 'anisotropic')
    print("结果:\n", ridge_enhanced)
