import numpy as np
import pandas as pd
from collections import Counter

import scipy.special
from src.utils import *

from src.environment import *
import tqdm

class GymDataset:
    def __init__(self, collapse_treatments=True):
        self.collapse_treatments = collapse_treatments
        self.load(self.collapse_treatments)
        self.average_beta = 0.5557
        self.average_alpha = [0.0326018 , 0.49851833, 0.12269805, 0.09385808, 0.11203568, 0.14028805]
        self.beta_distribution = [0.5, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.5, 0.3, 0.5, 0.1, 0.5,
            0.5, 0.5, 0.5, 0.9, 0.5, 0.3, 0.3, 0.5, 0.5, 0.9, 0.7, 0.9, 0.5,
            0.9, 0.5, 0.7, 0.3, 0.5, 0.3, 0.7, 0.3, 0.7, 0.9, 0.9, 0.5, 0.5,
            0.3, 0.3, 0.3, 0.5, 0.5, 0.3, 0.5, 0.1, 0.9, 0.5, 0.5, 0.9, 0.9,
            0.3, 0.5, 0.7, 0.3, 0.7, 0.7, 0.9, 0.5, 0.5, 0.3, 0.9, 0.5, 0.3,
            0.5, 0.7, 0.3, 0.9, 0.3, 0.9, 0.7, 0.3, 0.5, 0.3, 0.5, 0.5, 0.7,
            0.7, 0.5, 0.9, 0.7, 0.7, 0.3, 0.7, 0.9, 0.7, 0.5, 0.3, 0.5, 0.5,
            0.5, 0.9, 0.7, 0.7, 0.7, 0.3, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3,
            0.3, 0.9, 0.5, 0.7, 0.5, 0.9, 0.7, 0.5, 0.5, 0.3, 0.5, 0.7, 0.5,
            0.5, 0.7, 0.5, 0.5, 0.5, 0.9, 0.1, 0.5, 0.5, 0.1, 0.9, 0.7, 0.9,
            0.5, 0.9, 0.5, 0.5, 0.9, 0.5, 0.7, 0.9, 0.5, 0.9, 0.7, 0.5, 0.9,
            0.5, 0.5, 0.3, 0.9, 0.9, 0.5, 0.7, 0.9, 0.7, 0.3, 0.5, 0.5, 0.7,
            0.3, 0.9, 0.3, 0.3, 0.5, 0.3, 0.7, 0.9, 0.5, 0.5, 0.5, 0.7, 0.5,
            0.9, 0.5, 0.7, 0.5, 0.7, 0.1, 0.7, 0.3, 0.7, 0.5, 0.3, 0.7, 0.3,
            0.5, 0.3, 0.5, 0.9, 0.5, 0.9, 0.3, 0.7]

    def load(self, collapse_treatments=True):
        data = pd.read_csv("models/gym/pptdata.csv")
        data = data.drop('Unnamed: 0', axis='columns')
        self.data = data

        state_counts = Counter()
        for p_id, d in data.groupby('participant_id'):
            state = d['customer_state'].unique()[0]
            if np.any(d['customer_state'].isnull()):
                continue
            state_counts[state] += 1
        top_states = [s[0] for s in state_counts.most_common(9)]
        state_idx = {}  # State string --> If it's in top 10, the order, otherwise 9
        for state in state_counts.keys():
            if state in top_states:
                state_idx[state] = top_states.index(state)
            else:
                state_idx[state] = 9
        
        self.state_idx = state_idx

        if collapse_treatments:
            treatments = [
                "Placebo Control",  # baseline
                "Bonus for Returning after Missed Workouts b",  # financial
                "Exercise Social Norms Shared (High and Increasing)",  # affirm values
                "Planning Fallacy Described and Planning Revision Encouraged",  # planning
                "Exercise Commitment Contract Explained",  # advice
                "Fitness Questionnaire with Decision Support & Cognitive Reappraisal Prompt"  # reflection
            ]
        else:
            treatments = [
                "Bonus for Returning after Missed Workouts b",
                "Higher Incentives a",
                "Exercise Social Norms Shared (High and Increasing)",
                "Free Audiobook Provided",
                "Bonus for Returning after Missed Workouts a",
                "Planning Fallacy Described and Planning Revision Encouraged",
                "Choice of Gain- or Loss-Framed Micro-Incentives",
                "Exercise Commitment Contract Explained",
                "Free Audiobook Provided, Temptation Bundling Explained",
                "Following Workout Plan Encouraged",
                "Fitness Questionnaire with Decision Support & Cognitive Reappraisal Prompt",
                "Values Affirmation",
                "Asked Questions about Workouts",
                "Rigidity Rewarded a",
                "Defaulted into 3 Weekly Workouts",
                "Exercise Advice Solicited",
                "Exercise Fun Facts Shared",
                "Fitness Questionnaire",
                "Planning Revision Encouraged",
                "Exercise Social Norms Shared (Low)",
                "Exercise Encouraged with Typed Pledge",
                "Gain-Framed Micro-Incentives",
                "Higher Incentives b",
                "Rigidity Rewarded e",
                "Exercise Encouraged with Signed Pledge",
                "Values Affirmation Followed by Diagnosis as Gritty",
                "Bonus for Consistent Exercise Schedule",
                "Rigidity Rewarded c",
                "Loss-Framed Micro-Incentives",
                "Fitness Questionnaire with Cognitive Reappraisal Prompt",
                "Exercise Encouraged",
                "Planning Workouts Encouraged",
                "Gym Routine Encouraged",
                "Reflecting on Workouts Encouraged",
                "Planning Workouts Rewarded",
                "Effective Workouts Encouraged",
                "Planning Benefits Explained",
                "Reflecting on Workouts Rewarded",
                "Fun Workouts Encouraged",
                "Mon-Fri Consistency Rewarded, Sat-Sun Consistency Rewarded",
                "Exercise Encouraged with E-Signed Pledge",
                "Bonus for Variable Exercise Schedule",
                "Exercise Commitment Contract Explained Post-Intervention",
                "Rewarded for Responding to Questions about Workouts",
                "Defaulted into 1 Weekly Workout",
                "Exercise Social Norms Shared (Low but Increasing)",
                "Rigidity Rewarded d",
                "Exercise Commitment Contract Encouraged",
                "Fitness Questionnaire with Decision Support",
                "Rigidity Rewarded b",
                "Exercise Advice Solicited, Shared with Others",
                "Exercise Social Norms Shared (High)",
                "Placebo Control",
                "Planning, Reminders & Micro-Incentives to Exercise"
            ]

        treatment_idx = {} # maps alphanumeric treatment id to index in [0, K-1]
        for (t_id, t_name), _ in data.groupby(['treatment', 'exp_condition']):
            if t_name in treatments:
                treatment_idx[t_id] = treatments.index(t_name)

        self.treatment_idx = treatment_idx

        cohorts = ['1253', '1250', '1250wave2', '1235wave2', '1252wave2', '1238wave3',
               '1246', '1236', '1240', '1238wave2', '1235wave4', '1255wave2',
               '1234', '1235', '1239', '1245', '1235wave3', '1252', '1238',
               '1241', '1247', '1255', '1244', '1241wave2', '1237', '1242',
               'pure randomization']
        propensities_dict = {}
        for i, c in enumerate(cohorts):
            if c not in propensities_dict:
                propensities_dict[c] = []
            for j in range(54):
                if i == 0:
                    if j not in [29, 53]:
                        propensities_dict[c].append(1.69)
                    elif j == 29:
                        propensities_dict[c].append(6.78)
                    elif j == 53:
                        propensities_dict[c].append(5.08)
                elif i == 1:
                    if j not in [29, 53]:
                        propensities_dict[c].append(1.07)
                    elif j == 29:
                        propensities_dict[c].append(23.21)
                    elif j == 53:
                        propensities_dict[c].append(21.07)
                elif i == 2:
                    if j not in [1, 13, 22, 29, 50, 53]:
                        propensities_dict[c].append(1.02)
                    elif j in [1, 13, 22, 50]:
                        propensities_dict[c].append(10)
                    elif j == 29:
                        propensities_dict[c].append(4.07)
                    elif j == 53:
                        propensities_dict[c].append(7.04)
                elif i == 3:
                    if j not in [1, 13, 22, 29, 50, 53]:
                        propensities_dict[c].append(0.65)
                    elif j in [1, 13, 22, 50]:
                        propensities_dict[c].append(15)
                    elif j == 29:
                        propensities_dict[c].append(2.59)
                    elif j == 53:
                        propensities_dict[c].append(6.30)
                elif i == 4:
                    if j not in [16, 29, 51, 53]:
                        propensities_dict[c].append(0.64)
                    elif j == 16:
                        propensities_dict[c].append(20)
                    elif j == 29:
                        propensities_dict[c].append(21.91)
                    elif j == 51:
                        propensities_dict[c].append(20)
                    elif j == 53:
                        propensities_dict[c].append(6.27)
                elif i == 5:
                    if j not in [3, 8, 29, 53]:
                        propensities_dict[c].append(0.64)
                    elif j in [3, 8]:
                        propensities_dict[c].append(30.00)
                    elif j == 29:
                        propensities_dict[c].append(2.5)
                    elif j == 53:
                        propensities_dict[c].append(6.25)
                elif i == 6:
                    if j not in [0, 4, 27, 29, 47, 53]:
                        propensities_dict[c].append(0.65)
                    elif j in [0, 4, 27, 47]:
                        propensities_dict[c].append(15)
                    elif j == 29:
                        propensities_dict[c].append(2.59)
                    elif j == 53:
                        propensities_dict[c].append(6.30)
                elif i == 7:
                    if j not in [29, 32, 35, 53]:
                        propensities_dict[c].append(0.62)
                    elif j == 29:
                        propensities_dict[c].append(2.5)
                    elif j in [32, 35]:
                        propensities_dict[c].append(30.00)
                    elif j == 53:
                        propensities_dict[c].append(6.25)
                elif i == 8:
                    if j not in [11, 25, 29, 53]:
                        propensities_dict[c].append(0.62)
                    elif j in [11, 25]:
                        propensities_dict[c].append(30)
                    elif j == 29:
                        propensities_dict[c].append(2.5)
                    elif j == 53:
                        propensities_dict[c].append(6.25)
                elif i == 9:
                    if j not in [12, 29, 44, 53]:
                        propensities_dict[c].append(0.73)
                    elif j == 12:
                        propensities_dict[c].append(20)
                    elif j == 29:
                        propensities_dict[c].append(2.91)
                    elif j == 44:
                        propensities_dict[c].append(20)
                    elif j == 53:
                        propensities_dict[c].append(20.73)
                elif i == 10:
                    if j not in [7, 29, 43, 48, 53]:
                        propensities_dict[c].append(0.65)
                    elif j == 53:
                        propensities_dict[c].append(6.3)
                    elif j == 48:
                        propensities_dict[c].append(15)
                    elif j == 43:
                        propensities_dict[c].append(15)
                    elif j == 29:
                        propensities_dict[c].append(16.94)
                    elif j == 7:
                        propensities_dict[c].append(15)
                elif i == 11:
                    if j not in [2, 15, 19, 29, 46, 52, 53]:
                        propensities_dict[c].append(0.66)
                    elif j in [52, 46, 19, 15, 2]:
                        propensities_dict[c].append(12)
                    elif j == 29:
                        propensities_dict[c].append(2.64)
                    elif j == 53:
                        propensities_dict[c].append(6.32)
                elif i == 12:
                    if j not in [20, 24, 29, 31, 41, 53]:
                        propensities_dict[c].append(0.65)
                    elif j in [41, 31, 20, 24]:
                        propensities_dict[c].append(15)
                    elif j == 29:
                        propensities_dict[c].append(2.59)
                    elif j == 53:
                        propensities_dict[c].append(6.3)
                elif i == 13:
                    if j not in [10, 17, 29, 30, 49, 53]:
                        propensities_dict[c].append(0.65)
                    elif j in [10, 17, 30, 49]:
                        propensities_dict[c].append(15)
                    elif j == 29:
                        propensities_dict[c].append(2.59)
                    elif j == 53:
                        propensities_dict[c].append(6.3)
                elif i == 14:
                    if j not in [29, 36, 39, 53]:
                        propensities_dict[c].append(0.62)
                    elif j == 29:
                        propensities_dict[c].append(2.5)
                    elif j == 36:
                        propensities_dict[c].append(30)
                    elif j == 39:
                        propensities_dict[c].append(30)
                    elif j == 53:
                        propensities_dict[c].append(6.25)
                elif i == 15:
                    if j not in [9, 29, 37, 53]:
                        propensities_dict[c].append(0.62)
                    elif j == 9:
                        propensities_dict[c].append(30)
                    elif j == 29:
                        propensities_dict[c].append(2.5)
                    elif j == 37:
                        propensities_dict[c].append(30)
                    elif j == 53:
                        propensities_dict[c].append(6.25)
                elif i == 16:
                    if j not in [16, 29, 51, 53]:
                        propensities_dict[c].append(0.64)
                    elif j == 16:
                        propensities_dict[c].append(20)
                    elif j == 29:
                        propensities_dict[c].append(21.91)
                    elif j == 51:
                        propensities_dict[c].append(20)
                    elif j == 53:
                        propensities_dict[c].append(6.27)
                elif i == 17:
                    if j not in [5, 18, 29, 53]:
                        propensities_dict[c].append(0.62)
                    elif j in [5, 18]:
                        propensities_dict[c].append(30)
                    elif j == 29:
                        propensities_dict[c].append(2.5)
                    elif j == 53:
                        propensities_dict[c].append(6.25)
                elif i == 18:
                    if j not in [26, 29, 33, 42, 53]:
                        propensities_dict[c].append(0.64)
                    elif j in [26, 33, 42]:
                        propensities_dict[c].append(20)
                    elif j == 29:
                        propensities_dict[c].append(2.55)
                    elif j == 53:
                        propensities_dict[c].append(6.27)
                elif i == 19:
                    if j not in [6, 21, 28, 29, 53]: propensities_dict[c].append(0.64)
                    if j in [6, 21, 28]:
                        propensities_dict[c].append(20)
                    elif j == 29:
                        propensities_dict[c].append(2.55)
                    elif j == 53:
                        propensities_dict[c].append(6.27)
                elif i == 20:
                    if j not in [1, 13, 22, 29, 50, 53]:
                        propensities_dict[c].append(0.65)
                    elif j in [1, 13, 22, 50]:
                        propensities_dict[c].append(15)
                    elif j == 29:
                        propensities_dict[c].append(2.59)
                    elif j == 53:
                        propensities_dict[c].append(6.3)
                elif i == 21:
                    if j not in [0, 4, 27, 29, 47, 53]:
                        propensities_dict[c].append(0.65)
                    elif j in [0, 4, 27, 47]:
                        propensities_dict[c].append(15)
                    elif j == 29:
                        propensities_dict[c].append(2.59)
                    elif j == 53:
                        propensities_dict[c].append(6.3)
                elif i == 22:
                    if j not in [12, 29, 44, 53]:
                        propensities_dict[c].append(0.73)
                    elif j in [12, 44]:
                        propensities_dict[c].append(20)
                    elif j == 29:
                        propensities_dict[c].append(2.91)
                    elif j == 53:
                        propensities_dict[c].append(20.73)
                elif i == 23:
                    if j not in [1, 13, 22, 29, 50, 53]:
                        propensities_dict[c].append(1.02)
                    elif j in [1, 13, 22, 50]:
                        propensities_dict[c].append(10)
                    elif j == 29:
                        propensities_dict[c].append(4.07)
                    elif j == 53:
                        propensities_dict[c].append(7.04)
                elif i == 24:
                    if j not in [3, 8, 29, 53]:
                        propensities_dict[c].append(0.98)
                    elif j in [3, 8]:
                        propensities_dict[c].append(20)
                    elif j == 29:
                        propensities_dict[c].append(3.93)
                    elif j == 53:
                        propensities_dict[c].append(6.96)
                elif i == 25:
                    if j not in [0, 4, 27, 29, 47, 53]:
                        propensities_dict[c].append(1.02)
                    elif j in [0, 4, 27, 47]:
                        propensities_dict[c].append(10)
                    elif j == 29:
                        propensities_dict[c].append(4.07)
                    elif j == 53:
                        propensities_dict[c].append(7.04)
                elif i == 26:
                    if j not in [29, 32, 35, 53]:
                        propensities_dict[c].append(.98)
                    elif j == 29:
                        propensities_dict[c].append(3.93)
                    elif j == 32:
                        propensities_dict[c].append(20)
                    elif j == 35:
                        propensities_dict[c].append(20)
                    elif j == 53:
                        propensities_dict[c].append(6.96)

        self.propensities_dict = propensities_dict

    def split(self):
        train_cohorts = ['1253', '1250', '1250wave2', '1235wave2', '1252wave2', '1238wave3',
                     '1246', '1236', '1240', '1238wave2', '1235wave4', '1255wave2',
                     '1234']
        test_cohorts = ['1235', '1239', '1245', '1235wave3', '1252', '1238',
                        '1241', '1247', '1255', '1244', '1241wave2', '1237', '1242',
                        'pure randomization']
        
        train_df = self.data[self.data['cohort'].isin(train_cohorts)]
        test_df = self.data[self.data['cohort'].isin(test_cohorts)]
        return train_df, test_df
    
    def featurize(self, df):
        X = []
        Y = []
        propensities = []
        weights = []
        for p_id, p_data in tqdm.tqdm(df.groupby('participant_id')):
            treatment = p_data['treatment'].unique().item()
            if treatment not in self.treatment_idx:
                continue
            t_idx = self.treatment_idx[treatment]

            history_length = len(p_data[p_data['week'] < 0])
            if history_length < 5:
                continue

            covariates = []
            skip = False
            cohort = p_data['cohort'].unique().item()
            for cov in ['age', 'customer_state', 'gender', 'new_member']:
                d_c = p_data[cov]
                if np.any(d_c.isnull()):
                    skip = True
                    break
                val = d_c.unique()[0]
                if cov == 'customer_state':
                    covariates += list(np.eye(10)[self.state_idx[val]])
                elif cov == 'gender':
                    female = int(val == 'F')
                    covariates.append(female)
                else:
                    covariates.append(val)
            if skip:
                continue

            # Gather history
            for k, row in p_data.iterrows():
                curr_week = row['week']
                full_history = p_data[p_data['week'] < curr_week]

                if full_history.shape[0] == 0:  # Don't include users with 0 history
                    continue
                else:
                    avg_visits = np.mean(full_history['visits'])
                    min_visits = np.min(full_history['visits'])
                    max_visits = np.max(full_history['visits'])

                if curr_week == 1:  # There is no 0 week!
                    try:
                        num_visits_last_week = p_data[p_data['week'] == -1]['visits'].item()
                    except:
                        num_visits_last_week = 0
                    try:
                        num_visits_lastlast_week = p_data[p_data['week'] == -2]['visits'].item()
                    except:
                        num_visits_lastlast_week = 0
                elif curr_week == 2:
                    num_visits_last_week = p_data[p_data['week'] == 1]['visits'].item()
                    try:
                        num_visits_lastlast_week = p_data[p_data['week'] == -1]['visits'].item()
                    except:
                        num_visits_lastlast_week = 0
                elif full_history.shape[0] == 0:
                    num_visits_last_week = 0
                    num_visits_lastlast_week = 0
                else:
                    num_visits_last_week = full_history[full_history['week'] == curr_week - 1]['visits'].item()
                    if full_history.shape[0] == 1:
                        num_visits_lastlast_week = 0
                    else:
                        num_visits_lastlast_week = full_history[full_history['week'] == curr_week - 2]['visits'].item()

                num_visits_this_week = row['visits']

                if full_history.shape[0] == 0:
                    longest_streak, num_streaks = 0, 0
                else:
                    prev_visits = full_history['visits'].tolist()
                    longest_streak, num_streaks = streaks(prev_visits)
                receiving_treatment = 0
                total_weeks_treatment = 0
                if curr_week > 0 and curr_week <= 4:
                    receiving_treatment = 1
                    total_weeks_treatment = curr_week
                elif curr_week > 4:
                    total_weeks_treatment = 4
                
                K = 6 if self.collapse_treatments else 54
                treatment_feats = np.zeros((K, K - 1))
                treatment_feats[1:, :] = np.eye(K - 1)

                features = covariates + [avg_visits, min_visits, max_visits, num_visits_last_week,
                                        num_visits_lastlast_week] + \
                            [longest_streak, num_streaks, receiving_treatment, total_weeks_treatment] \
                            + treatment_feats[t_idx].tolist()
                
                X.append(features)
                Y.append(num_visits_this_week)
                propensities.append(self.propensities_dict[cohort][t_idx])
                weights.append(row['Lc'] / (row['Ngc'] * row['T_i']))

        X = np.asarray(X)
        Y = np.asarray(Y)
        propensities = np.asarray(propensities)
        weights = np.asarray(weights)

        return X, Y, weights, propensities

    def match_preferences(self, df):
        survey_data = pd.read_csv("models/gym/full_survey_redacted.csv")
        survey_data = survey_data.sample(frac=1) # Shuffle the data
        alpha_scale = {'Not at all increase': 0, 'Slightly increase': 1, 'Somewhat increase': 2, 'Moderately increase': 3,
                'Extremely increase': 4}
        beta_scale = {'Strongly disagree': 4, 'Disagree': 3, 'Neither agree nor disagree': 2, 'Agree': 1,
                    'Strongly agree': 0}
        
        print("Matching {} users".format(len(survey_data)))

        all_environments = []
        matched_participants = []

        for index, row in survey_data.iterrows():
            age = int(row['How old are you?'])
            gender = int(row['What is your gender?'] == "Female")  # 1 if female, 0 otherwise
            gender_str = 'F' if gender == 1 else 'M'

            is_new_member = int(row['Do you go to the gym?'] == "No")  # 1 if doesn't go, 0 if goes to the gym
            if not is_new_member:
                baseline_activity = int(row['How many days, on average, do you go to the gym in a week?'])
            else:
                baseline_activity = 0
            
            if not is_new_member:  # If they already go to the gym, get the first set of preferences
                pref_baseline = 0
                pref_finance = alpha_scale[row['Financial incentives a']]
                pref_affirm_values = alpha_scale[row['Messages that affirm your values a']]
                pref_planning = alpha_scale[row['Notifications to plan workouts a']]
                pref_reflection = alpha_scale[row['Notifications to reflect on your number of gym visits per week a']]
                pref_advice = alpha_scale[row['Advice to increase your number of gym visits a']]
                raw_beta = row['If an artificial intelligence (AI) system were to make recommendations for increasing my gym attendance, I would listen to the AI’s advice.']
            else:  # If they don't go to the gym, get the second set of preferences
                pref_baseline = 0
                pref_finance = alpha_scale[row['Financial incentives b']]
                pref_affirm_values = alpha_scale[row['Messages that affirm your values b']]
                pref_planning = alpha_scale[row['Notifications to plan workouts b']]
                pref_reflection = alpha_scale[
                    row['Notifications to reflect on your levels of physical activity per week b']]
                pref_advice = alpha_scale[row['Advice to increase your level of physical activity b']]
                raw_beta = row['If an artificial intelligence (AI) system were to make recommendations for increasing my levels of physical activity, I would listen to the AI’s advice.']

            alpha = [pref_baseline, pref_finance, pref_affirm_values, pref_planning, pref_reflection,
                    pref_advice]  # This is the order of actions
            alpha = scipy.special.softmax(alpha)
            if str(raw_beta) == 'nan':
                beta = np.random.choice(self.beta_distribution)  # If they don't have beta, sample it from the existing distribution
            else:
                beta = beta_scale[raw_beta]
                beta = 0.1 + 0.8 * beta / 4
        
            subset = df[(df['new_member'] == is_new_member) & (df['gender'] == gender_str) & (df['treatment'].isin(self.treatment_idx.keys()))]

            participant_candidates = []
            for p_id, p_data in subset.groupby('participant_id'):
                
                # If they are > 5 years older or younger, skip them
                if abs(p_data['age'].unique()[0] - age) > 5:
                    continue

                # Gather history
                pre_study_history = p_data[p_data['week'] <= 0]  # This is baseline data
                if pre_study_history.shape[0] == 0:  # Don't include users with 0 history
                    continue
                else:
                    baseline_candidate = np.mean(pre_study_history['visits'])
                
                if abs(baseline_candidate - baseline_activity) <= 0.5:
                    participant_candidates.append((p_id, pre_study_history['visits'].tolist()))

            participant_candidates = [p for p in participant_candidates if p[0] not in matched_participants]
            # Select one the participants randomly
            if len(participant_candidates) == 0:
                print('No matched users ' + str(index))
                raise Exception("No matches for participant {}".format(index))
            else:
                idx = np.random.choice(len(participant_candidates))
                matched_participant, user_history = participant_candidates[idx]
                matched_df = subset[subset['participant_id'] == matched_participant]

                matched_participants.append(matched_participant)

                treatment = matched_df['treatment'].unique().item()
                treatment_idx = self.treatment_idx[treatment]

                # Add bandits that all have the same initial state to the all-bandits list
                env = GymEnvironment(age=age, gender=gender, is_new_member=is_new_member, alpha=alpha, beta=beta,
                                customer_state=matched_df['customer_state'].unique().item(), user_history=user_history,
                                user_df=matched_df, treatment_administered=treatment_idx)
                all_environments.append(env)

        return all_environments
