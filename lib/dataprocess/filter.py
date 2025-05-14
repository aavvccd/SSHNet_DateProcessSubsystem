import numpy as np
def sliding_window_filter(matrix,window_size,filter_type='mean'):
    rows,cols = matrix.shape
    filteredMatrix = np.full([rows,cols],np.nan)
    halfWindowSize = np.floor(window_size/2)
    for i in range(0,rows):
        for j in range(0,cols):
            r1 = max(1,i-halfWindowSize)
            r2 = min(rows,i+halfWindowSize)
            c1 = max(1,j-halfWindowSize)
            c2 = min(cols,j+halfWindowSize)
            windows = matrix[r1:r2,c1:c2]
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
