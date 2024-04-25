import numpy as np
import tqdm
from src.utils import *
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from src.dataset import GymDataset
from src.policy import GymBandit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-decay', action='store_false', help='Disable beta decay.')
args = parser.parse_args()
beta_decay = args.no_decay

np.random.seed(42)
load_cached_data = False
load_cached_regressor = False

dataset = GymDataset()
train_data, test_data = dataset.split()

if load_cached_data:
    print("Loading cached data from: 'models/gym/train_data.pkl'")
    with open('models/gym/train_data.pkl', 'rb') as f:
        X_train, Y_train, train_weights, train_propensities = pickle.load(f)

    print("Loading cached data from: 'models/gym/test_data.pkl'")
    with open('models/gym/test_data.pkl', 'rb') as f:
        X_test, Y_test, test_weights, test_propensities = pickle.load(f)
else:
    print("Featurizing train data...")
    X_train, Y_train, train_weights, train_propensities = dataset.featurize(train_data)
    print("Saving training data to: 'models/gym/train_data.pkl'")
    with open('models/gym/train_data.pkl', 'wb') as f:
        pickle.dump([X_train, Y_train, train_weights, train_propensities], f)
    
    print("Featurizing test data...")
    X_test, Y_test, test_weights, test_propensities = dataset.featurize(test_data)
    print("Saving test data to: 'models/gym/test_data.pkl'")
    with open('models/gym/test_data.pkl', 'wb') as f:
        pickle.dump([X_test, Y_test, test_weights, test_propensities], f)

if load_cached_regressor:
    print("Loading cached regressor from: 'models/gym/gb_regressor.pkl'")
    with open('models/gym/gb_regressor.pkl', 'rb') as f:
        regr = pickle.load(f)
else:
    print("Training a gradient boosting regressor")
    regr = GradientBoostingRegressor()
    regr.fit(X_train, Y_train)

    print("Saving regression model to: 'models/gym/gb_regressor.pkl'")
    with open('models/gym/gb_regressor.pkl', 'wb') as f:
        pickle.dump(regr, f)

'''
Run the experiment with the constructed bandits
'''

P = 209 # number of participants
P = 100
A = 6 # number of actions
T = 4 # time horizon
M = 1 # number of outcomes
S = 10 # number of simulations
D = 27  # dimensionality

policies = {
    'TS_y': GymBandit(M=M, D=D),
    'TS_r': GymBandit(M=M, D=D),
    'Random': GymBandit(M=M, D=D),
    'A=0': GymBandit(M=M, D=D),
    'Eps_Oracle_R': GymBandit(M=M, D=D),
    'Oracle_Y': GymBandit(M=M, D=D),
    'Eps_Oracle_Greedy': GymBandit(M=M, D=D),
    'Avg_Pref': GymBandit(M=M, D=D),
    'Uniform_Pref': GymBandit(M=M, D=D),
    'Separate_TS_R': [GymBandit(M=M, D=D) for _ in range(P)]
}

inst_regret = {pi: np.zeros((S, P, T)) for pi in policies.keys()}
y = {pi: np.zeros((S, P, T+1)) for pi in policies.keys()} # so that we can store the initial (baseline) visits
r = {pi: np.zeros((S, P, T)) for pi in policies.keys()}
alpha = {pi: np.zeros((S, P, T)) for pi in policies.keys()}

for sim, s in enumerate(tqdm.tqdm(range(S))):
    '''
    Match preferences from the survey to users in the test dataset, and generate bandits
    '''
    all_matched = False
    while not all_matched:
        try:
            matched_environments = dataset.match_preferences(test_data)
            all_matched = True
        except KeyboardInterrupt: # otherwise you cannot Ctrl+C out of the script
            break
        except:
            print("Matching failed, retrying...")

    matched_environments = matched_environments[:P]
    
    # reset all policies
    for name, pi in policies.items():
        if name == 'Separate_TS_R':
            [pi_s.reset() for pi_s in pi]
        else:
            pi.reset()

    # initialize a history for each policy in the environment
    for env in matched_environments:
        env.initialize_history(policies)

    for t in range(T):
        for i in range(P):
            user_env = matched_environments[i]
            noise = 0 

            for name, pi in policies.items():
                if t == 0:
                    y[name][s, i, t] = user_env.previous_visits[name][-1]

                if name == 'A=0':
                    a_it = 0
                elif name == 'Random':
                    a_it = np.random.choice(A)
                elif name == 'TS_y':
                    Phi_train = np.asarray([user_env.phi_sa(a=a, static_covariates=user_env.static_covariates,
                                               previous_visits=user_env.previous_visits[name],
                                               num_treatments=user_env.num_treatments[name])
                                            for a in range(A)])
                    a_it = pi.choose_ts(Phi_train, beta=user_env.beta, alpha=user_env.alpha, choose_outcome=True)
                elif name == 'TS_r':
                    Phi_train = np.asarray([user_env.phi_sa(a=a, static_covariates=user_env.static_covariates,
                                               previous_visits=user_env.previous_visits[name],
                                               num_treatments=user_env.num_treatments[name])
                                            for a in range(A)])
                    a_it = pi.choose_ts(Phi_train, beta=user_env.beta, alpha=user_env.alpha, choose_outcome=False)
                elif name == 'Eps_Oracle_R':
                    possible_outcomes = [user_env.pull(a=a, 
                                           regressor=regr,
                                           previous_visits=user_env.previous_visits[name],
                                           num_treatments=user_env.num_treatments[name] , 
                                           noise=noise)
                                        for a in range(A)]
                    a_it = np.argmax([r for (r, y) in possible_outcomes])
                    if np.random.uniform() < 0.1:
                        a_it = np.random.choice(A)
                elif name == 'Eps_Oracle_Greedy':
                    Phi_train = np.asarray([user_env.phi_sa(a=a, static_covariates=user_env.static_covariates,
                                               previous_visits=user_env.previous_visits[name],
                                               num_treatments=user_env.num_treatments[name])
                                            for a in range(A)])
                    a_it = pi.choose_greedy(Phi_train, beta=user_env.beta, alpha=user_env.alpha, choose_outcome=False)
                    if np.random.uniform() < 0.1:
                        a_it = np.random.choice(A)
                elif name == 'Oracle_Y':
                    possible_outcomes = [user_env.pull(a=a, 
                                           regressor=regr,
                                           previous_visits=user_env.previous_visits[name],
                                           num_treatments=user_env.num_treatments[name] , 
                                           noise=noise)
                                        for a in range(A)]
                    a_it = np.argmax([y for (r, y) in possible_outcomes])
                elif name == 'Avg_Pref':
                    Phi_train = np.asarray([user_env.phi_sa(a=a, static_covariates=user_env.static_covariates,
                                               previous_visits=user_env.previous_visits[name],
                                               num_treatments=user_env.num_treatments[name])
                                            for a in range(A)])
                    if beta_decay:
                        decayed_avg_beta = dataset.average_beta - t * (dataset.average_beta / T)
                    else:
                        decayed_avg_beta = dataset.average_beta
                    
                    a_it = pi.choose_ts(Phi_train, beta=decayed_avg_beta, alpha=dataset.average_alpha, choose_outcome=False)
                elif name == 'Uniform_Pref':
                    Phi_train = np.asarray([user_env.phi_sa(a=a, static_covariates=user_env.static_covariates,
                                               previous_visits=user_env.previous_visits[name],
                                               num_treatments=user_env.num_treatments[name])
                                            for a in range(A)])
                    if beta_decay:
                        uniform_beta = 0.5 - t * (0.5 / T)
                    else:
                        uniform_beta = 0.5

                    uniform_alpha = np.ones(A)
                    uniform_alpha[0] = 0
                    uniform_alpha = uniform_alpha / np.sum(uniform_alpha)

                    a_it = pi.choose_ts(Phi_train, beta=uniform_beta, alpha=uniform_alpha, choose_outcome=False)
                elif name == 'Separate_TS_R':
                    Phi_train = np.asarray([user_env.phi_sa(a=a, static_covariates=user_env.static_covariates,
                                               previous_visits=user_env.previous_visits[name],
                                               num_treatments=user_env.num_treatments[name])
                                            for a in range(A)])
                    a_it = pi[i].choose_ts(Phi_train, beta=user_env.beta, alpha=user_env.alpha, choose_outcome=False)

                possible_outcomes = [user_env.pull(a=a, 
                                           regressor=regr,
                                           previous_visits=user_env.previous_visits[name],
                                           num_treatments=user_env.num_treatments[name], 
                                           noise=noise)
                                        for a in range(A)]
                
                potential_utilities = [r for (r, y) in possible_outcomes]
                potential_visits = [y for (r, y) in possible_outcomes]
                
                y_it = potential_visits[a_it]
                r_it = potential_utilities[a_it]

                phi_it = user_env.phi_sa(a=a_it, static_covariates=user_env.static_covariates,
                                             previous_visits=user_env.previous_visits[name],
                                             num_treatments=user_env.num_treatments[name])
                if name == 'Separate_TS_R':
                    pi[i].update_individual(phi_it, [y_it])
                else:
                    pi.update_individual(phi_it, [y_it])

                r_star = np.max(potential_utilities)

                y[name][s, i, t+1] = y_it
                r[name][s, i, t] = r_it
                alpha[name][s, i, t] = user_env.alpha[a_it]
                inst_regret[name][s, i, t] = (r_star - r_it)
                user_env.previous_visits[name].append(y_it)
                user_env.num_treatments[name] += int(a_it > 0)

            # linear beta decay
            if beta_decay:
                user_env.beta -= user_env.original_beta / T

        for name, pi in policies.items():
            if name == 'Separate_TS_R':
                [pi_s.update_time() for pi_s in pi]
            else:
                pi.update_time()

if beta_decay:
    with open ('./experiments/inst_regret_24h_linear_decay.pkl', 'wb') as f:
        pickle.dump(inst_regret, f)
    with open ('./experiments/y_24h_linear_decay.pkl', 'wb') as f:
        pickle.dump(y, f)
    with open ('./experiments/r_24h_linear_decay.pkl', 'wb') as f:
        pickle.dump(r, f)
    with open ('./experiments/alpha_24h_linear_decay.pkl', 'wb') as f:
        pickle.dump(alpha, f)
else:
    with open ('./experiments/inst_regret_24h_no_decay.pkl', 'wb') as f:
        pickle.dump(inst_regret, f)
    with open ('./experiments/y_24h_no_decay.pkl', 'wb') as f:
        pickle.dump(y, f)
    with open ('./experiments/r_24h_no_decay.pkl', 'wb') as f:
        pickle.dump(r, f)
    with open ('./experiments/alpha_24h_no_decay.pkl', 'wb') as f:
        pickle.dump(alpha, f)
