import numpy as np
from itertools import groupby

def streaks(lst):
    if len(lst) == 0 or sum(lst) == 0:
        return 0, 0
    else:
        # Longest streak
        g = groupby(lst, key=lambda x: x > 0)
        longest_streak = len(max([list(s) for v, s in g if v > 0], key=len))

        # Num streaks
        num_streaks = 0
        for key, group in groupby(lst, lambda x: x > 0):
            if key:
                num_streaks += 1

        return longest_streak, num_streaks

class HeartStepsEnvironment:
    def __init__(self, T, user_type, sig, A, y0):
        '''
        :param T: number of timesteps
        :param sig: noise
        :param user_type: ['active', 'inactive']
        '''
        self.group = user_type
        self.A = A
        self.user_type = user_type
        self.sig = sig
        self.T = T
        self.y0 = y0
        # [1, y_{t-1}, active_te, inactive_te, num_notifications]
        self.Theta = [-0.04] + [0.9999] + [0.3, 0.15] + [0]

        if user_type == 'active':
            self.weights = [0.6, 0.4]
            self.group = 0
            self.goal = np.sqrt(10000)
            self.u1_intercept = 0.42
        elif user_type == 'inactive':
            self.weights = [0.4, 0.6]
            self.group = 1
            self.goal = np.sqrt(5600)
            self.u1_intercept = 0.3
        else:
            raise Exception("Not a valid user type, only valid types are active and inactive")
        self.sig = sig
        self.num_notifications = {}
        self.previous_steps = {}

    def pull(self, a, n_notif, prev_y, noise, T):
        p_sa = self.phi_sa(a=a, group=self.group, n_notif=n_notif + a, prev_y=prev_y, T=T)
        y_it = np.dot(p_sa, self.Theta) + noise
        r = 0
        r += self.weights[0] * self.u1(y_it, p_sa)
        r += self.weights[1] * self.u2(y_it, p_sa)
        return r, y_it

    def reset(self, policies):
        for pi in policies:
            self.num_notifications[pi] = 0
            self.previous_steps[pi] = self.y0
    
    def u1(self, y, phi_it):
        if y < self.goal:
            return 0.005 * y
        else:
            return 0.001 * y + self.u1_intercept

    def u2(self, y, phi_it):
        num_notifications = phi_it[-1] # Last element of phi_it is the previous steps?
        return -(num_notifications ** 2) * 0.00003

    def phi_sa(self, a, group, n_notif, prev_y, T):
        if group == 0:
            inv_sigmoid = lambda x: np.exp(-x) / (1 + np.exp(-x))
            map_t = lambda x: (x - T / .95) * (5 / T)
        else:
            inv_sigmoid = lambda x: np.exp(-x) / (1 + np.exp(-x))
            map_t = lambda x: (x - T / 0.65) * (2 / T)
            
        s = inv_sigmoid(map_t(n_notif + a))
        group_te = [0, 0]
        group_te[group] = s * a
        phi = [1, prev_y] + group_te + [n_notif + a]
        return np.asarray(phi)


class GymEnvironment:
    def __init__(self, age, gender, customer_state, is_new_member, alpha, beta, user_history, user_df, treatment_administered):

        # Initialize the state
        top_states = ['CA', 'TX', 'CO', 'WA', 'OR', 'FL', 'NY', 'HI', 'NJ']
        if customer_state not in top_states:
            self.customer_state = np.zeros(10).tolist()
            self.customer_state[-1] = 1
        else:
            self.customer_state = np.eye(10)[top_states.index(customer_state)].tolist()

        self.alpha = alpha
        self.original_beta = beta
        self.beta = beta
        self.age = age
        self.gender = gender
        self.is_new_member = is_new_member
        self.user_history = user_history
        self.user_df = user_df
        self.treatment_administered = treatment_administered

        self.week = 1

        self.static_covariates = [self.age] + self.customer_state + [self.gender, self.is_new_member]
        self.num_treatments = {}
        self.previous_visits = {}

    def phi_sa(self, static_covariates, previous_visits, num_treatments, a):
        if len(previous_visits) == 0:
            avg_visits = 0
            min_visits = 0
            max_visits = 0
            num_visits_last_week = 0
        else:
            avg_visits = np.mean(previous_visits)
            min_visits = np.mean(previous_visits)
            max_visits = np.mean(previous_visits)
            num_visits_last_week = previous_visits[-1]
        num_visits_last_last_week = 0 if len(previous_visits) <= 1 else previous_visits[-2]
        longest_streak, num_streaks = streaks(previous_visits)
        receiving_treatment = int(a > 0)
        total_weeks_receiving_treatment = num_treatments + int(a > 0)

        K = 6  # There are six possible treatments
        treatment_feats = np.zeros((K, K - 1))
        treatment_feats[1:, :] = np.eye(K - 1)

        phi_sa = static_covariates + [avg_visits, min_visits, max_visits, num_visits_last_week, num_visits_last_last_week] + \
                    [longest_streak, num_streaks, receiving_treatment, total_weeks_receiving_treatment] \
                    + treatment_feats[a].tolist()
        return np.asarray(phi_sa)

    def pull(self, a, regressor, previous_visits, num_treatments, noise, custom_alpha=None, custom_beta=None):
        if a == self.treatment_administered: # Get the number of gym visits from the dataset
            y_it = self.user_df[self.user_df['week'] == self.week]['visits'].item()
        else: # Use the model to predict the gym visits
            phi_sa = self.phi_sa(self.static_covariates, previous_visits, num_treatments, a)
            y_it = regressor.predict(np.expand_dims(phi_sa, axis=0)).item()

            prob = min(1, max(y_it - int(y_it), 0))
            y_it = int(y_it) + np.random.choice([0, 1], p=[1-prob, prob]) 

        scale_factor = 7.59
        if custom_alpha is not None and custom_beta is not None:
            r_it = scale_factor * custom_beta * custom_alpha[a] + (1-custom_beta) * y_it
        else:
            r_it = scale_factor * self.beta * self.alpha[a] + (1-self.beta) * y_it 
        return r_it, y_it

    def initialize_history(self, policies):
        for pi in policies:
            self.num_treatments[pi] = 0
            self.previous_visits[pi] = self.user_history.copy()
        
        self.week = 1