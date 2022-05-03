import numpy as np
def get_param_range(subset_size, exp_decay, method, data):
    if method=='BGD':
        g_range = [0.001]
        b_range = np.arange(180, 200, 1) * .005
        return g_range, b_range
    if data=='IMDBLargeCLinear':
        g_range = [0.0002]
        b_range = np.arange(20, 40, 1) * 0.005
    elif data in [ 'IMDBCLinear','IMDBLargeC5']:
        g_range = [0.0001]
        b_range = np.arange(180, 200, 1) * .005
    elif data in ['IMDBC5']:
        g_range = [0.001]
        b_range = np.arange(180, 200, 1) * .005
    elif data in ['Brazilnew']:
        g_range = [0.01]
        b_range = np.arange(20, 40, 1) * .005
    elif data in ['stackn']:
        g_range = [0.0001]
        b_range = np.arange(180, 200, 1) * .005
    elif data in ['taxi']:
        g_range = [0.0001]
        b_range = np.arange(20, 40, 1) * .005
    else:
        g_range = [0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.3, 0.35]
        b_range = [0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.9, 0.95]
        if subset_size < 1:
            g_range = [0.000035, 0.009, 0.01, 0.013, 0.015, 0.017, 0.018, 0.019, 0.02, 0.025, 0.03]
            b_range = np.arange(0, 19) * .01
    return g_range, b_range


