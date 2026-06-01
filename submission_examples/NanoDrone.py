"""
Submission template for the NanoDrone benchmark.
─────────────────────────────────────────────────────────────────────────────
EVALUATION PROTOCOL (from Busetto et al., 2026)
─────────────────────────────────────────────────────────────────────────────
The model is evaluated on the Melon test trajectory (3 runs).
At each time step t, the model receives the true state y_t and the full
future input sequence {u_{t+k}}_{k=0}^{H-1}, and must predict the next H=50
states open-loop (no state corrections):

    ŷ_{t+1:t+H} = M(y_t, u_{t:t+H-1})

Errors are averaged over all valid start times T = {0,...,T_test - H - 1},
and then averaged across the 3 Melon runs. Four metrics are reported:

    MAE_p,h   — Euclidean position error at horizon h          [m]
    MAE_v,h   — Euclidean linear velocity error at horizon h   [m/s]
    MAE_R,h   — Geodesic orientation error at horizon h        [rad]
    MAE_ω,h   — Euclidean angular velocity error at horizon h  [rad/s]

For submission, report the cumulative simulation error (sum over h=1..50):

    SimErr_p  = Σ_{h=1}^{50} MAE_p,h     [m]
    SimErr_v  = Σ_{h=1}^{50} MAE_v,h     [m/s]
    SimErr_R  = Σ_{h=1}^{50} MAE_R,h     [rad]
    SimErr_ω  = Σ_{h=1}^{50} MAE_ω,h     [rad/s]

─────────────────────────────────────────────────────────────────────────────
PREDICTION FORMAT
─────────────────────────────────────────────────────────────────────────────
The model must predict the 13-dimensional state vector:

    y = [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]

where orientation is represented as a unit quaternion [qx, qy, qz, qw]
(scalar-last convention), consistent with the benchmark dataset.

The submitter is free to use any internal orientation representation
(e.g. rotation vectors, Euler angles) during training and inference,
as long as predictions are converted back to unit quaternions before
computing the metrics below.

Prediction columns in the output DataFrame:
    {x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz}_pred_h{1..50}
"""

import numpy as np
import pandas as pd
import nonlinear_benchmarks
from nonlinear_benchmarks.nanodrone_utils import quat_conj, quat_mul
from nonlinear_benchmarks.nanodrone_error_metrics import compute_errors, print_results


# ── Load benchmark data ────────────────────────────────────────────────────

# train: 9 Input_output_data  (square/random/chirp × run1/run2/run3)
# test:  3 Input_output_data  (melon × run1/run2/run3)
#
# test[i].u  shape (N,  4): [m1_rads, m2_rads, m3_rads, m4_rads]
# test[i].y  shape (N, 13): [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
train, test = nonlinear_benchmarks.NanoDrone()

sampling_time = train[0].sampling_time  # 0.01 s  (100 Hz)
max_horizon = 50  # 0.5 s ahead

# ── Train your model ───────────────────────────────────────────────────────
# Use ONLY the training data. You may split further into train/validation.
# Do NOT use the test data to make any modelling decision.
#
# from my_model import train_model, apply_model
# model = train_model(train)

# ── Evaluate on the 3 Melon test runs ─────────────────────────────────────
state_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw', 'wx', 'wy', 'wz']

all_metrics = []

for test_run in test:
    print(f"\nProcessing {test_run.name} ...")

    N = len(test_run.u)
    t = np.arange(N) * sampling_time

    # Ground truth columns from the benchmark data
    df_dict = {
        't': t,
        'x': test_run.y[:, 0], 'y': test_run.y[:, 1], 'z': test_run.y[:, 2],
        'vx': test_run.y[:, 3], 'vy': test_run.y[:, 4], 'vz': test_run.y[:, 5],
        'qx': test_run.y[:, 6], 'qy': test_run.y[:, 7],
        'qz': test_run.y[:, 8], 'qw': test_run.y[:, 9],
        'wx': test_run.y[:, 10], 'wy': test_run.y[:, 11], 'wz': test_run.y[:, 12],
    }

    # Generate multi-horizon predictions
    # apply_model must return an array of shape (N, 13) for each horizon h,
    # with columns in state_cols order. Orientation must be unit quaternions.
    # The model may only use test_run.u as input — no future state feedback.
    for h in range(1, max_horizon + 1):
        # y_pred_h = apply_model(model, test_run, h)  # shape (N, 13)
        y_pred_h = np.zeros((N, 13))
        y_pred_h[:, 9] = 1.0  # qw = 1  →  identity quaternion [0,0,0,1]

        for col_idx, col in enumerate(state_cols):
            df_dict[f'{col}_pred_h{h}'] = y_pred_h[:, col_idx]

    df_pred = pd.DataFrame(df_dict)

    # Compute and display metrics for this run
    metrics = compute_errors(df_pred, max_horizon=max_horizon)
    all_metrics.append(metrics)
    print_results(metrics, label=test_run.name)

    # Optionally save the prediction CSV
    # df_pred.to_csv(f'predictions_{test_run.name}.csv', index=False)

# ── Average across the 3 Melon runs ───────────────────────────────────────
avg_metrics = {
    key: {h: np.mean([m[key][h] for m in all_metrics])
          for h in range(1, max_horizon + 1)}
    for key in ['pos', 'vel', 'rot', 'omega']
}

print_results(avg_metrics, label='AVERAGE across melon runs (submit these)')