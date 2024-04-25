import numpy as np

def oracle(bandit, t):
    r0 = bandit.pull(t, int(0))
    r1 = bandit.pull(t, int(1))
    return max(r0, r1)

def no_intervention_heartstep(all_bandits, T=200):
    R_active = []
    Y_active = []
    R_inactive = []
    Y_inactive = []
    for b in all_bandits:
        r_b = []
        y_b = []
        for t in range(T):
            noise = np.random.randn() * b.sig
            r, y = b.pull(a=0, n_notif=b.ts_r_notifications, noise=noise, prev_y=b.yprev_ts_r)
            b.ts_r_notifications += 0
            r_b.append(r)
            y_b.append(y)
            b.yprev_ts_r = y
        if b.patient_type == 'active':
            R_active.append(r_b)
            Y_active.append(y_b)
        else:
            R_inactive.append(r_b)
            Y_inactive.append(y_b)
    results = {"R_active": R_active, "R_inactive": R_inactive, "Y_active": Y_active, "Y_inactive": Y_inactive}
    return results

def all_intervention_heartstep(all_bandits, T=200):
    R_active = []
    Y_active = []
    R_inactive = []
    Y_inactive = []
    for b in all_bandits:
        r_b = []
        y_b = []
        for t in range(T):
            noise = np.random.randn() * b.sig
            r, y = b.pull(a=1, n_notif=b.ts_r_notifications, noise=noise, prev_y=b.yprev_ts_r)
            b.ts_r_notifications += 1
            r_b.append(r)
            y_b.append(y)
            b.yprev_ts_r = y
        if b.patient_type == 'active':
            R_active.append(r_b)
            Y_active.append(y_b)
        else:
            R_inactive.append(r_b)
            Y_inactive.append(y_b)
    results = {"R_active": R_active, "R_inactive": R_inactive, "Y_active": Y_active, "Y_inactive": Y_inactive}
    return results

def random_intervention_heartstep(all_bandits, T=200):
    R_active = []
    Y_active = []
    R_inactive = []
    Y_inactive = []
    for b in all_bandits:
        r_b = []
        y_b = []
        for t in range(T):
            a = np.random.choice([0, 1])
            noise = np.random.randn() * b.sig
            r, y = b.pull(a=a, n_notif=0, noise=noise, prev_y=b.yprev_ts_r)
            b.ts_r_notifications += a
            r_b.append(r)
            y_b.append(y)
            b.yprev_ts_r = y
        if b.patient_type == 'active':
            R_active.append(r_b)
            Y_active.append(y_b)
        else:
            R_inactive.append(r_b)
            Y_inactive.append(y_b)
    results = {"R_active": R_active, "R_inactive": R_inactive, "Y_active": Y_active, "Y_inactive": Y_inactive}
    return results

def reset_education_bandits(all_bandits):
    for b in all_bandits:
        b.reset()
    return all_bandits

def convert_keys_to_int(d):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        new_dict[new_key] = v
    return new_dict


def bits_list(n, val = None):
    bits = [int(x) for x in bin(n)[2:]]
    if val is None or len(bits) >= val:
        return bits
    else:
        return [0] * (val-len(bits)) + bits


# class MetricWrapper(tf.keras.metrics.Metric):
#     def __init__(self, base_metric_class, **kwargs):
#         super(MetricWrapper, self).__init__(name=base_metric_class.name, **kwargs)
#         self.metric_class = base_metric_class
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         true, pred = get_target(y_true, y_pred)
#         self.metric_class.update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)
#
#     def result(self):
#         return self.metric_class.result()
#
#     def reset_states(self):
#         self.metric_class.reset_states()


def SR(pi, Phi, R, W, G, choose_step):
    V_pi_hat = []
    V_pi_star = []
    diffs = []
    N = Phi.shape[0]
    T = Phi.shape[1]
    for i in range(N):
        for t in range(T):
            # Hide the notification index while choosing the best action!
            a = pi.choose_mle(choose_step=choose_step, phi=Phi[i, t], w=W[i], group=G[i])
            V_pi_hat.append(R[i, t, a])
            V_pi_star.append(np.max(R[i, t]))
            diff = np.max(R[i, t]) - R[i, t, a]
            diffs.append(diff)
    return np.mean(V_pi_star) - np.mean(V_pi_hat)

def is_mse(pred, true, weight):
    mse = 0
    for p, t, w in zip(pred, true, weight):
        mse += w*np.square((t-p))
    return mse/len(pred)

def wis_mse(pred, true, weights):
    mse = 0
    denom = 0
    for w in weights:
        denom += w
    for p, t, w in zip(pred, true, weights):
        mse += (w*np.square(t-p))/(denom)
    return mse

def unweighted_mse(pred, true):
    mse = 0
    for p, t in zip(pred, true):
        mse += np.square(t-p)
    return mse/len(pred)