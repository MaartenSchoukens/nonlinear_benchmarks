import numpy as np

#adapted from https://github.com/forgi86/pytorch-ident/blob/master/torchid/metrics.py
from nonlinear_benchmarks.utilities import Input_output_data

def RMSE(y_true, y_model, n_init=0, time_axis=0):
    """ Computes the Root Mean Square Error (RMSE) (also called RMS).

    The RMSE index is computed separately on each channel as:
    RMSE = np.sqrt(np.mean((y_model - y_true)**2, axis=time_axis))

    Parameters
    ----------
    y_true : np.array | Input_output_data
        Array of true values. 
    y_model : np.array | Input_output_data
        Array of predicted values. 
    n_init : int
        Number of samples that need to be removed due to usage in state initialization
        n_init will be set to y_true.state_initialization_window_length if isinstance(y_true, Input_output_data)
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    RMSE_val : np.array
        Array of RMSE values.

    """
    if isinstance(y_true, Input_output_data):
        n_init = y_true.state_initialization_window_length
        n_init = 0 if n_init==None else n_init
        y_true = y_true.y
    if isinstance(y_model, Input_output_data):
        y_model = y_model.y
    assert y_true.shape == y_model.shape, f'y_true and y_model should have the same shape, currently y_true.shape={y_true.shape} and y_model.shape={y_model.shape}'

    RMSE_val = np.sqrt(np.mean((y_model[n_init:] - y_true[n_init:])**2, axis=time_axis))
    return RMSE_val

def NRMSE(y_true, y_model, n_init=0, time_axis=0, std_tolerance=1e-10):
    """ Computes the Normalized Root Mean Square Error (NRMSE) (also called NRMS)

    The NRMSE index is computed separately on each channel as:
    NRMSE = RMSE/standard deviation of y_true per channel
          = RMSE/np.std(y_true, axis=time_axis)
          = np.sqrt(np.mean((y_model - y_true)**2, axis=time_axis))/np.std(y_true, axis=time_axis)

    Parameters
    ----------
    y_true : np.array | Input_output_data
        Array of true values.
    y_model : np.array | Input_output_data
        Array of predicted values.
    n_init : int
        Number of samples that need to be removed due to usage in state initialization
        n_init will be set to y_true.state_initialization_window_length if isinstance(y_true, Input_output_data)
    time_axis : int
        Time axis. All other axes define separate channels.
    std_tolerance : float
        The tolerance that the std should be large otherwise you might divide by zero


    Returns
    -------
    NRMSE_val : np.array
        Array of r_squared value.

    """
    
    RMSE_val = RMSE(y_true, y_model, n_init=n_init, time_axis=time_axis)
    if isinstance(y_true, Input_output_data):
        n_init = y_true.state_initialization_window_length
        n_init = 0 if n_init==None else n_init
        y_true = y_true.y
    std = np.std(y_true[n_init:], axis=time_axis)
    assert np.all(std>std_tolerance), 'the standard deviation of y_true is almost zero! std={std}'
    NRMSE_val = RMSE_val/std
    return NRMSE_val


def R_squared(y_true, y_model, n_init=0, time_axis=0):
    """ Computes the coefficient of determination R^2.

    The R^2 is computed separately on each channel. Given by
    R^2 = 1 - Mean Squared Error / Variance of y_test
        = 1 - MSE/np.var(y_true, axis=time_axis)
        = 1 - np.mean((y_model - y_true)**2, axis=time_axis)/np.var(y_true, axis=time_axis)
        = 1 - NRMSE^2 = 1 - (RMSE/Std(y_true))^2

    Parameters
    ----------
    y_true : np.array | Input_output_data
        Array of true values.  If must be at least 2D.
    y_model : np.array | Input_output_data
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    r_squared_val : np.array
        Array of r_squared value.
    """

    return 1.0 - NRMSE(y_true, y_model, n_init=0, time_axis=time_axis)**2


def MAE(y_true, y_model, n_init=0, time_axis=0):
    """ Computes the Mean Absolute value Error (MAE)

    The MAE is computed separately on each channel as:
    MSE = np.mean(np.abs(y_true - y_model), axis=time_axis)

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_model : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    MSE_val : np.array
        Array of mean absolute value errors.
    """
    if isinstance(y_true, Input_output_data):
        n_init = y_true.state_initialization_window_length
        n_init = 0 if n_init==None else n_init
        y_true = y_true.y
    if isinstance(y_model, Input_output_data):
        y_model = y_model.y
    assert y_true.shape == y_model.shape, f'y_true and y_model should have the same shape, currently y_true.shape={y_true.shape} and y_model.shape={y_model.shape}'

    MSE_val = np.mean(np.abs(y_true[n_init:] - y_model[n_init:]), axis=time_axis)
    return MSE_val


def fit_index(y_true, y_model, n_init=0, time_axis=0):
    """ Computes the per-channel fit index.

    The fit index is commonly used in System Identification. See the definition in the System Identification Toolbox
    or in the paper 'Nonlinear System Identification: A User-Oriented Road Map',
    https://arxiv.org/abs/1902.00683, page 31.
    The fit index is computed separately on each channel.
    fit = 100*(1 - np.sqrt(np.mean((y_true-y_model)**2))/np.sqrt(np.mean((y_true-np.mean(y_true))**2) 
        = 100*(1 - RMSE/Standard deviation of y_true)
        = 100*(1 - NRMSE)
        = 100*(1 - sqrt(1-R^2))

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_model : np.array
        Array of predicted values.  It must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    fit_index_val : np.array
        Array of fit index.

    """

    fit_index_val = 100*(1 - NRMSE(y_true, y_model, n_init=n_init, time_axis=time_axis))
    return fit_index_val


if __name__ == '__main__':
    N = 200
    for ny in [(), (2,), (12,2)]:
        SNR = 10
        y_true_std = 0.1
        y_true = np.random.randn(N, *ny)*y_true_std
        noise = np.random.randn(N, *ny)/SNR*y_true_std
        y_model = y_true + noise
        
        RMSE_val = RMSE(y_model, y_true)
        NRMSE_val = NRMSE(y_model, y_true)
        R_squared_val = R_squared(y_true, y_model)
        MAE_val = MAE(y_true, y_model)
        fit_index_val = fit_index(y_true, y_model)

        print(f"RMSE: {RMSE_val}")
        print(f"NRMSE: {NRMSE_val}")
        print(f"R-squared: {R_squared_val}")
        print(f'MAE: {MAE_val}')
        print(f"fit index: {fit_index_val}")