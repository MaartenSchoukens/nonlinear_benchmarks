import numpy as np

# === Quaternion utilities ===
def quat_conj(q):
    """ q = [x, y, z, w] """
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_mul(q1, q2):
    """ Hamilton product, both q=[x,y,z,w] """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_geodesic_error(q_true, q_pred):
    """Geodesic distance e_R = 2 atan2(||v||, w)."""
    qr = quat_mul(quat_conj(q_true), q_pred)
    v = qr[:3]
    w = qr[3]
    return 2 * np.arctan2(np.linalg.norm(v), w)

# === Core metric function ===
def compute_errors(df, max_horizon):
    """
    Returns dict:
      {
        'pos': {h: MAE_pos_h},
        'omega': {h: MAE_omega_h},
        'rot': {h: MAE_rot_h}
      }
    """
    errs_pos = {}
    errs_vel = {}
    errs_omega = {}
    errs_rot = {}

    for h in range(1, max_horizon + 1):

        # Align truth and predicted horizon=h
        true_pos = df[['x','y','z']].shift(-h+1).dropna().values
        pred_pos = np.vstack([df[f'{s}_pred_h{h}'].dropna().values for s in ['x','y','z']]).T
        min_len = min(len(true_pos), len(pred_pos))
        pos_err = np.linalg.norm(true_pos[:min_len] - pred_pos[:min_len], axis=1)
        errs_pos[h] = pos_err.mean()

        true_vel = df[['vx','vy','vz']].shift(-h+1).dropna().values
        pred_vel = np.vstack([df[f'{s}_pred_h{h}'].dropna().values for s in ['vx','vy','vz']]).T
        min_len = min(len(true_vel), len(pred_vel))
        vel_err = np.linalg.norm(true_vel[:min_len] - pred_vel[:min_len], axis=1)
        errs_vel[h] = vel_err.mean()

        true_omega = df[['wx','wy','wz']].shift(-h+1).dropna().values
        pred_omega = np.vstack([df[f'{s}_pred_h{h}'].dropna().values for s in ['wx','wy','wz']]).T
        min_len = min(len(true_omega), len(pred_omega))
        omega_err = np.linalg.norm(true_omega[:min_len] - pred_omega[:min_len], axis=1)
        errs_omega[h] = omega_err.mean()

        # Orientation
        true_q = df[['qx','qy','qz','qw']].shift(-h+1).dropna().values
        pred_q = df[[f'qx_pred_h{h}', f'qy_pred_h{h}', f'qz_pred_h{h}', f'qw_pred_h{h}']].dropna().values
        min_len = min(len(true_q), len(pred_q))
        rot_err = np.array([
            quat_geodesic_error(true_q[i], pred_q[i])
            for i in range(min_len)
        ])
        errs_rot[h] = rot_err.mean()

    return {
        "pos": errs_pos,
        "vel": errs_vel,
        "omega": errs_omega,
        "rot": errs_rot
    }

# ============================================================
# === Compute cumulative simulation error (sum over h=1..50)
# ============================================================
def compute_simerr(metric_dict):
    sim_p = sum(metric_dict["pos"][h] for h in range(1, 51))
    sim_v = sum(metric_dict["vel"][h] for h in range(1, 51))
    sim_R = sum(metric_dict["rot"][h] for h in range(1, 51))
    sim_w = sum(metric_dict["omega"][h] for h in range(1, 51))
    return sim_p, sim_v, sim_R, sim_w

def print_results(metrics, label='Model', max_horizon=50):
    """Print a formatted summary table at key horizons plus SimErr."""
    H_report = [1, 10, 50]
    print(f"\n{'=' * 68}")
    print(f"  {label}")
    print(f"{'=' * 68}")
    print(f"{'Metric':<22} {'h=1':>10} {'h=10':>10} {'h=50':>10} {'SimErr':>10}")
    print(f"{'-' * 68}")
    rows = [
        ('Position    [m]', 'pos'),
        ('Lin. vel.   [m/s]', 'vel'),
        ('Orientation [rad]', 'rot'),
        ('Ang. vel.   [r/s]', 'omega'),
    ]
    for name, key in rows:
        vals = [metrics[key][h] for h in H_report]
        sim = sum(metrics[key][h] for h in range(1, max_horizon + 1))
        print(f"{name:<22} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {sim:>10.4f}")
    print(f"{'=' * 68}")

    sim_p, sim_v, sim_R, sim_w = compute_simerr(metrics, max_horizon)
    print(f"\nNumbers to submit:")
    print(f"  SimErr_pos   = {sim_p:.6f}   [m]")
    print(f"  SimErr_vel   = {sim_v:.6f}   [m/s]")
    print(f"  SimErr_rot   = {sim_R:.6f}   [rad]")
    print(f"  SimErr_omega = {sim_w:.6f}   [rad/s]")
