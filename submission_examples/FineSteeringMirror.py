import nonlinear_benchmarks
import numpy as np

# Load data: 
# Keep this part fixed, though you can split the train set further in a train and validation set. 
# Do not use the test set to make any decision about the model (parameters, hyperparameters, structure, ...)
trains, tests = nonlinear_benchmarks.FineSteeringMirror()
n = tests[0].state_initialization_window_length

# Train model:
# Modify this part such that you can train your model starting from the training data.
from simple_model import train_model, apply_model
model = train_model(trains)

# Apply model on test data:
# Only use u and y[:n] returning y_models (keep n fixed to the value provided by the benchmark dataset)
# Modify this part such that you can simulate the response of your model to the test input.

y_tests_model = []
for test in tests:
    y_tests_model.append(apply_model(model, test.u, test.y[:n]))
    
# Compute RMSE in meters and print results:
# Keep this functionally unchanged
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE

all_RMSEs = []
all_NRMSEs = []
for i, test, prediction in zip(range(len(tests)), tests, y_tests_model):
    test_RMSE = 1e6 * RMSE(test.y, prediction, n_init=n)
    test_NRMSE = NRMSE(test.y, prediction, n_init=n)
    
    # Average over all periods and realizations
    test_RMSE_mean = np.mean(test_RMSE, axis=(1,2))
    test_NRMSE_mean = np.mean(test_NRMSE, axis=(1,2))
    
    all_RMSEs.append(test_RMSE_mean)
    all_NRMSEs.append(test_NRMSE_mean)
        
for test, test_RMSE, test_NRMSE in zip(tests, all_RMSEs, all_NRMSEs):
    print(f"{test.name}:")
    print(f"  RMSE to submit: {np.mean(test_RMSE):.3e} µm")
    
    # Useful per-output RMSEs and relative errors:    
    for output_idx, (RMSE_output, NRMSE_output) in enumerate(zip(test_RMSE, test_NRMSE)):
        print(f"  output {output_idx + 1}: {RMSE_output:.3e} µm ({100*NRMSE_output:.2f}%)")
    print()