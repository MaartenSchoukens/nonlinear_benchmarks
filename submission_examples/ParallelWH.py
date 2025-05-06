import nonlinear_benchmarks

# Load data: 
# Keep this part fixed, though you can split the train set further in a train and validation set. 
# Do not use the test set to make any decision about the model (parameters, hyperparameters, structure, ...)
trains, tests = nonlinear_benchmarks.ParWH()
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


# Compute RMSE in mm (milimeters) and print results: 
# Keep this functionally unchanged
from nonlinear_benchmarks.error_metrics import RMSE

all_RMSEs = []
for i, test, prediction in zip(range(len(tests)), tests, y_tests_model):
    test_RMSE = 1000*RMSE(test.y[n:], prediction[n:])
    all_RMSEs.append(test_RMSE)
    print(f'test set {i+1}, {test_RMSE = :.2f} mv')



print('RMSE to submit = [', *(f"{x:.2f}" for x in all_RMSEs), ']') # report this number during submission

