import numpy as np

def gen_rand_param(seed):
    np.random.seed(seed)

    rand_param_dict = {}

    rand_param_dict['seed'] = seed
    rand_param_dict['gE'] = np.random.uniform(0.1,16.0)
    rand_param_dict['gP'] = np.random.uniform(0.1,24.0)
    rand_param_dict['gS'] = np.random.uniform(0.1,8.0)
    rand_param_dict['bS'] = np.random.uniform(0.1,48.0)
    rand_param_dict['WEE'] = np.random.uniform(0.1,32.0)
    rand_param_dict['WEP'] = np.random.uniform(0.1,24.0)
    rand_param_dict['WES'] = np.random.uniform(0.1,64.0)
    rand_param_dict['WPE'] = np.random.uniform(0.1,56.0)
    rand_param_dict['WPP'] = np.random.uniform(0.1,32.0)
    rand_param_dict['WPS'] = np.random.uniform(0.1,16.0)
    rand_param_dict['WSE'] = np.random.uniform(0.1,64.0)
    rand_param_dict['WSP'] = np.random.uniform(0.1,16.0)

    return rand_param_dict
