import tqdm
from src.utils import *
from src.environment import HeartStepsEnvironment
from src.policy import HeartStepsBandit
import pickle

np.random.seed(42)
num_users_per_group = 20 # Number of users per user group type
A = 2 # Number of actions, here they are [0, 1]
T = 100 # Horizon in days
S = 20 # Number of simulations
D = 5 # Dimension of the context
M = 1 # Number of outcomes. Here it is just step count
all_users = []
sig = 1
for j, group in enumerate(['active', 'inactive']):
    for i in range(num_users_per_group):
        if group == 'active': sig=sig; y0=89 + np.random.randn()*sig # So that we can get a spectrum of users
        elif group == 'inactive': sig=sig; y0=70 + np.random.randn()*sig
        user = HeartStepsEnvironment(T=T, user_type=group, sig=sig, A=A, y0=y0)
        all_users.append(user)
'''
Run the experiment with the constructed bandits
'''
P = len(all_users)

policies = {
    'TS_y':HeartStepsBandit(M=M, D=D),
    'TS_r': HeartStepsBandit(M=M, D=D),
    'Random': HeartStepsBandit(M=M, D=D),
    'Separate_TS_R': [HeartStepsBandit(M=M, D=D) for _ in range(P)],
    'A=1': HeartStepsBandit(M=M, D=D),
    'A=0': HeartStepsBandit(M=M, D=D),
    'A=1(y < goal)': HeartStepsBandit(M=M, D=D)
}

inst_regret = {pi: np.zeros((S, P, T)) for pi in policies.keys()}
y = {pi: np.zeros((S, P, T)) for pi in policies.keys()}
r = {pi: np.zeros((S, P, T)) for pi in policies.keys()}

for sim, s in enumerate(tqdm.tqdm(range(S))):
    # reset all policies
    for name, pi in policies.items():
        if name == 'Separate_TS_R':
            [pi_s.reset() for pi_s in pi]
        else:
            pi.reset()

    for env in all_users:
        env.reset(policies)

    for t in range(T):
        for i in range(P):
            user_env = all_users[i]
            w_i = user_env.weights
            noise = np.random.randn() * user_env.sig

            for name, pi in policies.items():
                if name == 'Random':
                    a_it = np.random.choice(A)
                elif name == "TS_y":
                    Phi_train = np.asarray([user_env.phi_sa(a=a, n_notif=user_env.num_notifications[name],
                                           group=user_env.group, prev_y=user_env.previous_steps[name], T=T) for a in range(A)])
                    a_it = pi.choose_ts(Phi_train, w=w_i, choose_step=True, user_env=user_env)
                elif name == "TS_r":
                    Phi_train = np.asarray([user_env.phi_sa(a=a, n_notif=user_env.num_notifications[name],
                                           group=user_env.group, prev_y=user_env.previous_steps[name], T=T) for a in range(A)])
                    a_it = pi.choose_ts(Phi_train, w=w_i, choose_step=False, user_env=user_env)
                elif name == "Separate_TS_R":
                    Phi_train = np.asarray([user_env.phi_sa(a=a, n_notif=user_env.num_notifications[name],
                                           group=user_env.group, prev_y=user_env.previous_steps[name], T=T) for a in range(A)])
                    a_it = pi[i].choose_ts(Phi_train, w=w_i, choose_step=False, user_env=user_env)
                elif name == "A=1":
                    a_it = 1
                elif name == "A=0":
                    a_it = 0
                elif name == "A=1(y < goal)":
                    if user_env.previous_steps[name] < user_env.goal:
                        a_it = 1
                    else:
                        a_it = 0

                possible_outcomes = [user_env.pull(a=a, n_notif=user_env.num_notifications[name], noise=noise,
                                             prev_y=user_env.previous_steps[name], T=T) for a in range(A)]
                potential_utilities = [r for (r, y) in possible_outcomes]
                potential_steps = [y for (r, y) in possible_outcomes]

                r_it = potential_utilities[a_it]
                y_it = potential_steps[a_it]

                phi_it = user_env.phi_sa(a=a_it, n_notif=user_env.num_notifications[name], group=user_env.group,
                                         prev_y=user_env.previous_steps[name], T=T)
                if name == 'Separate_TS_R':
                    pi[i].update_individual(phi_it, [y_it])
                else:
                    pi.update_individual(phi_it, [y_it])

                r_star = np.max(potential_utilities)

                y[name][s, i, t] = y_it
                r[name][s, i, t] = r_it
                inst_regret[name][s, i, t] = (r_star - r_it)
                user_env.previous_steps[name] = y_it
                user_env.num_notifications[name] += a_it

        for name, pi in policies.items():
            if name == 'Separate_TS_R':
                [pi_s.update_time() for pi_s in pi]
            else:
                pi.update_time()

with open ('./experiments/inst_regret_heartsteps.pkl', 'wb') as f:
    pickle.dump(inst_regret, f)
with open ('./experiments/y_heartsteps.pkl', 'wb') as f:
    pickle.dump(y, f)
with open ('./experiments/r_heartsteps.pkl', 'wb') as f:
    pickle.dump(r, f)