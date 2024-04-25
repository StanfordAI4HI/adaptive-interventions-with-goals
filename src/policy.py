import numpy as np

class MultiUserLinearBandit:
    def __init__(self, M, D, lam=1):
        '''
        :param M: number of outcomes
        :param D: number of dimensions in a sample
        :param lam: prior (usually set to 1)
        '''
        self.M = M
        self.D = D
        self.lam = lam

        self.Sigma = np.eye(D) * lam
        self.Sigma_inv = np.eye(D) * (1. / lam)
        self.Omega = np.zeros((D, D))
        self.Beta = np.zeros((M, D))
        self.theta = np.zeros((M, D))
        # self.utilities = [u1, u2]

    def reset(self):
        '''
        Reset the bandit
        '''
        self.Sigma = np.eye(self.D) * self.lam
        self.Sigma_inv = np.eye(self.D) * (1. / self.lam)
        self.Omega = np.zeros((self.D, self.D))
        self.Beta = np.zeros((self.M, self.D))
        self.theta = np.zeros((self.M, self.D))

    def update_individual(self, phi, y):
        '''
        Call this function after observing every user.
        :param phi: phi \in R^d, the representation of the sample
        :param y: y \in R^k, the outcome
        '''
        M = self.M
        for m in range(M):
            self.Beta[m] += (phi * y[m])
        self.Omega += np.outer(phi, phi)

    def update_time(self):
        '''
        Call this function after every timestep (in the horizon)
        '''
        self.Sigma += self.Omega
        self.Omega *= 0
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.theta = (self.Sigma_inv @ self.Beta.T).T

    def choose_mle(self, phi, w, group, choose_step):
        '''
        Choose maximum likelihood estimate
        :param phi:
        :param w:
        :param group:
        :param choose_step:
        :return:
        '''
        A = phi.shape[0]
        scores = []
        for a in range(A):
            # print("Temp Phi: ", temp_phi_a)
            y_it = (phi[a] @ self.theta.squeeze()).item() # Still chooosing noise here.
            if choose_step:
                scores.append(y_it)
            else:
                score = 0 # Only select by utilities.
                for k in range(len(self.utilities)):
                    score += w[k] * self.utilities[k](y_it, phi[a], group) # Use the actual phi[a] here, so we can keep track of notifications
                scores.append(score)
        scores = np.array(scores)
        return np.random.choice(np.flatnonzero(scores == scores.max()))

class HeartStepsBandit(MultiUserLinearBandit):
    def __init__(self, M, D, lam=1):
        super().__init__(M, D, lam)

    def choose_ts(self, phi, w, choose_step, user_env):
        '''
        Choose action based on Thompson Sampling
        :param phi:
        :param w:
        :param choose_step:
        :param group:
        :param print_scores:
        :return:
        '''
        A = phi.shape[0]
        scores = []
        theta_tilde = np.random.multivariate_normal(self.theta.squeeze(), (1. / self.lam) * self.Sigma_inv)
        for a in range(A):
            y_it = phi[a] @ theta_tilde
            if choose_step:
                scores.append(y_it) # Choosing action based on what maximizes y_it
            else:
                score = 0
                score += w[0] * user_env.u1(y_it, phi[a])
                score += w[1] * user_env.u2(y_it, phi[a])
                scores.append(score) # Choosing action based on what maximizes weighted sum of utilies
        scores = np.array(scores)
        return np.random.choice(np.flatnonzero(scores == scores.max()))

class GymBandit(MultiUserLinearBandit):
    def __init__(self, M, D, lam=1):
        super().__init__(M, D, lam)
    
    def choose_ts(self, phi, beta, alpha, choose_outcome):
        # Choose outcome is true if we only care about gym visits, not reward
        A = phi.shape[0]
        scores = []
        theta_tilde = np.random.multivariate_normal(self.theta.squeeze(), (1. / self.lam) * self.Sigma_inv)

        for a in range(A):
            y_it = phi[a] @ theta_tilde
            if choose_outcome:
                scores.append(y_it) # Chooses an action that maximizes gym visits
            else:
                score = 7.5 * beta * alpha[a] + (1-beta) * y_it
                scores.append(score)
        scores = np.array(scores)
        return np.random.choice(np.flatnonzero(scores == scores.max()))
    
    def choose_greedy(self, phi, beta, alpha, choose_outcome):
        A = phi.shape[0]
        scores = []
        for a in range(A):
            y_it = phi[a] @ self.theta.squeeze()
            if choose_outcome:
                scores.append(y_it)
            else:
                score = 7.59 * beta * alpha[a] + (1-beta) * y_it
                scores.append(score)
        scores = np.array(scores)
        return np.random.choice(np.flatnonzero(scores == scores.max()))
        