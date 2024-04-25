import os
import pickle
import numpy as np

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['lines.linewidth'] = 3
import matplotlib.pyplot as plt
from matplotlib import font_manager

# search for fonts in system library to find CMU Bright
# this only works on macOS and you will need to install CMU Bright
home_dir = os.path.expanduser('~')
font_dirs = [os.path.join(home_dir, 'Library/Fonts/')]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams.update({'font.size': 24, 'font.family': 'CMU Bright', 'font.weight': 'bold'})

colors = {
    'TS_r': 'C0', # blue
    'TS_y': 'C1', # orange
    'Random': 'C7', # gray 
    'Separate_TS_R': 'C2', # green 
    'Avg_Pref': 'C4', # purple
    'Uniform_Pref': 'C3', # red
    'A=1': 'C5', # brown
    'A=0': 'C9', # cyan
    'A=1(y < goal)': 'C6' # pink
}

linestyles = {
    'TS_r': 'solid',
    'TS_y': 'dashed',
    'Random': 'dotted',
    'Separate_TS_R': 'dashdot',
    'Avg_Pref': (5, (10, 3)),
    'Uniform_Pref': (10, (3, 1, 1, 1)),
    'A=1': 'solid',
    'A=0': 'dashed',
    'A=1(y < goal)': 'dashed'
}

############################################
# Heartsteps                               #
############################################
print("Heartsteps")

with open('./experiments/inst_regret_heartsteps.pkl', 'rb') as f:
    inst_regret_heartsteps = pickle.load(f)
with open('./experiments/y_heartsteps.pkl', 'rb') as f:
    y_hearsteps = pickle.load(f)
with open('./experiments/r_heartsteps.pkl', 'rb') as f:
    r_heartsteps = pickle.load(f)

num_users_per_group = 20 
A = 2 
T = 100 
S = 20 
D = 5 
M = 1

# Step Count vs Utility

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
group = 0

for pi, data in y_hearsteps.items():
    if pi == 'A=1(y < goal)':
        label = 'A=1{y < goal}'
    elif pi == 'A=1':
        label = pi
    else:
        continue

    color = colors[pi]
    linestyle = linestyles[pi]
    
    y_active = data[:, :num_users_per_group, :].reshape(-1, T)
    y_active = np.square(y_active)
    mean = np.mean(y_active, axis=0)
    std = np.std(y_active, axis=0) / np.sqrt(y_active.shape[0])
    ax[0].plot(np.arange(T), mean, label=label, color=color, linestyle=linestyle)
    ax[0].fill_between(np.arange(T), mean-std, mean+std, alpha=0.2, color=color)


for pi, data in r_heartsteps.items():
    if pi == 'A=1(y < goal)':
        label = 'A=1{y < goal}'
    elif pi == 'A=1':
        label = pi
    else:
        continue
    
    color = colors[pi]
    linestyle = linestyles[pi]
    
    y_active = data[:, :num_users_per_group, :].reshape(-1, T)
    mean = np.mean(y_active, axis=0)
    std = np.std(y_active, axis=0) / np.sqrt(y_active.shape[0])
    ax[1].plot(np.arange(T), mean, label=label, color=color, linestyle=linestyle)
    ax[1].fill_between(np.arange(T), mean-std, mean+std, alpha=0.2, color=color)

ax[0].axhline(10000, color='k', linestyle=':', label='Step Count Goal')
ax[0].set_xlim([0, T])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_xlabel("Time (Days)", fontweight='bold')
ax[0].set_title("Step Count", pad=10, fontweight='bold')
ax[0].legend(bbox_to_anchor=(2.6, -0.2), fancybox=True, ncol=4)
ax[0].grid(alpha=0.3)

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_title("Reward (Weighted Utility)", pad=10, fontweight='bold')
ax[1].set_xlabel("Time (Days)", fontweight='bold')
ax[1].grid(alpha=0.3)
ax[1].set_xlim([0, T])

plt.subplots_adjust(wspace=0.3)
plt.savefig('plots/heartsteps-steps-vs-utility.pdf', bbox_inches="tight")

# Cumulative Regret

fig = plt.figure(figsize=(6, 6))
ax = plt.gca()

for pi in ['TS_r', 'TS_y', 'Separate_TS_R', 'Random']:
    data = inst_regret_heartsteps[pi]
    (S, P, T) = data.shape 
    
    if pi == 'TS_r': 
        label = "TS(r)"
    # Random policy
    elif pi == 'Random': 
        label = "Random"
    elif pi == 'TS_y':
        label = "TS(y)"
    elif pi == 'Separate_TS_R':
        label = "TS(r), no data sharing"
    else:
        continue

    color = colors[pi]
    linestyle = linestyles[pi]

    agg_data = np.sum(data, axis=1) # (S, T) 
    cumulative_regret = np.cumsum(agg_data, axis=1) # (S, T)
    mean_cumulative_regret = np.mean(cumulative_regret, axis=0).tolist() # T,
    err_cumulative_regret = (np.std(cumulative_regret, axis=0) / np.sqrt(len(cumulative_regret))).tolist()

    # Add 0 to both the mean CR and err CR graphs
    mean_cr = np.asarray([0] + mean_cumulative_regret)
    err_cr = np.asarray([0] + err_cumulative_regret)

    ax.plot(mean_cr, label=label, color=color, linestyle=linestyle)
    ax.fill_between([i for i in range(T+1)], mean_cr-err_cr, mean_cr+err_cr, alpha=0.2, color=color)
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel("Time (weeks)")

ax.set_ylim([-0.00, 2]) 
ax.set_xlim([0, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

ax.legend(bbox_to_anchor=(2.05, 1.05))
ax.tick_params(axis='both', which='major')

ax.set_xlabel("Time (Days)", fontweight='bold')
ax.set_title("Cumulative Regret", pad=20, fontweight='bold')
plt.savefig("plots/heartsteps-cum-regret.pdf", dpi=300, bbox_inches='tight')

# Step Count Dynamics

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

for pi, data in y_hearsteps.items():
    if pi == 'A=1' or pi == 'A=0' or pi == 'Random':
        label = pi
        color = colors[pi]
        linestyle = linestyles[pi]
    else:
        continue
    
    y_active = data[:, :num_users_per_group, :].reshape(-1, T)
    y_inactive = data[:, num_users_per_group:, :].reshape(-1, T)

    y_active = np.square(y_active)
    mean_active = np.mean(y_active, axis=0)
    std_active = np.std(y_active, axis=0) / np.sqrt(y_active.shape[0])
    ax[0].plot(np.arange(T), mean_active, label=label, color=color, linestyle=linestyle)
    ax[0].fill_between(np.arange(T), mean_active-std_active, mean_active+std_active, alpha=0.2, color=color)
    
    y_inactive = np.square(y_inactive)
    mean_inactive = np.mean(y_inactive, axis=0)
    std_inactive = np.std(y_inactive, axis=0) / np.sqrt(y_inactive.shape[0])
    ax[1].plot(np.arange(T), mean_inactive, label=label, color=color, linestyle=linestyle)
    ax[1].fill_between(np.arange(T), mean_inactive-std_inactive, mean_inactive+std_inactive, alpha=0.2, color=color)

ax[0].set_ylabel('Step Count')
ax[0].axhline(10000, color='k', linestyle=(5, (10, 3)), label='Step Count Goal')
ax[0].set_xlim([0, T])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_xlabel("Time (Days)")
ax[0].set_title("Active")
ax[0].legend(bbox_to_anchor=(2.1, -0.2), fancybox=True, ncol=4)
ax[0].grid(alpha=0.3)

# ax[1].set_ylabel('Step Count')
ax[1].axhline(5600, color='k', linestyle=(5, (10, 3)))
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_title("Inactive")
ax[1].set_xlabel("Time (Days)")
ax[1].grid(alpha=0.3)
ax[1].set_xlim([0, T])

plt.subplots_adjust(wspace=0.3)
plt.savefig('plots/heartsteps-active-inactive.pdf', bbox_inches="tight")

############################################
# 24H Fitness                              #
############################################
print("24H Fitness")

with open('./experiments/inst_regret_24h_linear_decay.pkl', 'rb') as f:
    inst_regret_24h_linear_decay = pickle.load(f)
with open('./experiments/y_24h_linear_decay.pkl', 'rb') as f:
    y_24h_linear_decay = pickle.load(f)
with open('./experiments/r_24h_linear_decay.pkl', 'rb') as f:
    r_24h_linear_decay = pickle.load(f)
with open ('./experiments/alpha_24h_linear_decay.pkl', 'rb') as f:
    alpha_24h_linear_decay = pickle.load(f)

with open('./experiments/inst_regret_24h_no_decay.pkl', 'rb') as f:
    inst_regret_24h_no_decay = pickle.load(f)
with open('./experiments/y_24h_no_decay.pkl', 'rb') as f:
    y_24h_no_decay = pickle.load(f)
with open('./experiments/r_24h_no_decay.pkl', 'rb') as f:
    r_24h_no_decay = pickle.load(f)
with open ('./experiments/alpha_24h_no_decay.pkl', 'rb') as f:
    alpha_24h_no_decay = pickle.load(f)

# Visits vs Preferences

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for pi in ['TS_r', 'TS_y', 'Random']:
    data = y_24h_no_decay[pi]
    if pi == 'TS_r':
        label = 'TS(r)'
    elif pi == 'TS_y':
        label = 'TS(y)'
    else:
        label = pi
    
    color = colors[pi]
    linestyle = linestyles[pi]
    
    S, P, T_ = data.shape
    data = data.reshape((S*P, T_))
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) / np.sqrt(S*P)
    ax[0].plot(np.arange(T_), mean, label=label, color=color, linestyle=linestyle)
    ax[0].fill_between(np.arange(T_), mean-std, mean+std, alpha=0.2, color=color)


for pi in ['TS_r', 'TS_y', 'Random']:
    data = alpha_24h_no_decay[pi]
    if pi == 'TS_r':
        label = 'TS(r)'
    elif pi == 'TS_y':
        label = 'TS(y)'
    else:
        label = pi

    color = colors[pi]
    linestyle = linestyles[pi]
    
    S, P, T = data.shape
    data = data.reshape((S*P, T))
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) / np.sqrt(S*P)
    ax[1].plot(np.arange(1, T+1), mean, label=label, color=color, linestyle=linestyle)
    ax[1].fill_between(np.arange(1, T+1), mean-std, mean+std, alpha=0.2, color=color)

ax[0].set_xlim([0, T])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_xlabel("Time (Weeks)", fontweight='bold')
ax[0].set_ylabel("Weekly Number of Gym Visits")
ax[0].set_title("Gym Visits", pad=10, fontweight='bold')
ax[0].legend(bbox_to_anchor=(2.5, -0.2), fancybox=True, ncol=3)
ax[0].grid(alpha=0.3)

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_title(r"Preference Alignment", pad=10, fontweight='bold')
ax[1].set_xlabel("Time (Weeks)", fontweight='bold')
ax[1].set_ylabel("Average Preference Value")
ax[1].grid(alpha=0.3)
ax[1].set_xlim([1, T])

plt.subplots_adjust(wspace=0.5)
plt.savefig('plots/24h-visits-vs-utility.pdf', bbox_inches="tight")

# Cumulative Regret

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for pi in ['TS_r', 'TS_y', 'Separate_TS_R', 'Avg_Pref', 'Uniform_Pref', 'Random']:
    data_decay = inst_regret_24h_linear_decay[pi]
    data_no_decay = inst_regret_24h_no_decay[pi]
    (S, P, T) = data_decay.shape 
    
    if pi == 'TS_r': 
        label = "TS(r)"
    elif pi == 'Random': 
        label = "Random"
    elif pi == 'TS_y':
        label = "TS(y)"
    elif pi == 'Separate_TS_R':
        label = "TS(r), no data sharing"
    elif pi == 'Avg_Pref':
        label = "TS(r), average preferences"
    elif pi == 'Uniform_Pref':
        label = "TS(r), uniform preferences"
    else:
        continue
    
    color = colors[pi]
    linestyle = linestyles[pi]
    
    # subset_data = data.reshape(S, P, T) # S, P, T
    for i, data in enumerate([data_no_decay, data_decay]):
        agg_data = np.sum(data, axis=1) # (S, T) 
        cumulative_regret = np.cumsum(agg_data, axis=1) # (S, T)
        mean_cumulative_regret = np.mean(cumulative_regret, axis=0) # T,
        err_cumulative_regret = np.std(cumulative_regret, axis=0) / np.sqrt(len(cumulative_regret))

        # Add 0 to both the mean CR and err CR graphs
        mean_cr = np.asarray([0] + mean_cumulative_regret.tolist())
        err_cr = np.asarray([0] + err_cumulative_regret.tolist())

        ax[i].plot(mean_cr, label=label, color=color, linestyle=linestyle)
        ax[i].fill_between([i for i in range(T+1)], mean_cr-err_cr, mean_cr+err_cr, alpha=0.2, color=color)


for i in range(2):
    ax[i].set_xlim([0, T])
    ax[i].set_ylim([0, 1000])
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].grid(alpha=0.3)
    ax[i].set_xlabel('Time (Weeks)', fontweight='bold')
    
ax[0].set_ylabel('Cumulative Regret', fontweight='bold')
ax[0].set_title("No Beta Decay", fontweight='bold', pad=10)
ax[1].set_title("Beta Decay", fontweight='bold', pad=10)

plt.subplots_adjust(wspace=0.3)
ax[0].legend(bbox_to_anchor=(1.1, -0.7), loc='lower center', ncol=2)

plt.savefig("plots/24h-cum-regret.pdf", dpi=300, bbox_inches='tight')

# Calibration Plot

with open('models/gym/train_data.pkl', 'rb') as f:
    X_train, Y_train, train_weights, train_propensities = pickle.load(f)

with open('models/gym/test_data.pkl', 'rb') as f:
    X_test, Y_test, test_weights, test_propensities = pickle.load(f)

with open('models/gym/gb_regressor.pkl', 'rb') as f:
    regr = pickle.load(f)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

fig = plt.figure(figsize=(12, 8))
ax = plt.gca()

Y_pred = regr.predict(X_test)
eps = 0.3
y_jitter = eps * (2 * np.random.random(size=len(Y_test)) - 1)
plt.scatter(Y_pred, Y_test + y_jitter, alpha=0.002, s=100)

plt.plot(range(8), range(8), color='k', label='Exact Calibration', alpha=0.4)

# Polynomial trend line (higher degrees are still quite linear)
degree = 1 # Degree of the polynomial
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(Y_pred.reshape(-1, 1), Y_test)
x_range = np.linspace(min(Y_pred), max(Y_pred), 100)
y_poly = polyreg.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_poly, color='red', linestyle='dashed', alpha=0.75, label='Linear Trend')

for i in range(8):
    idx = Y_test == i
    mean = np.mean(Y_pred[idx])
    if i == 0:
        plt.scatter(mean, i, color='k', marker='x', s=100, alpha=0.75, label='Conditional Mean')
    else:
        plt.scatter(mean, i, color='k', marker='x', s=100, alpha=0.75)

# ax.set_ylim([-0.00, 2]) 
ax.set_xlim([-0.2, 7.1])
ax.set_ylim([-0.5, 7.5])
ax.set_yticks(range(8))
ax.set_xticks(range(8))
ax.set_xlabel("Predicted Gym Visits")
ax.set_ylabel("True Gym Visits")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

plt.legend();

plt.savefig('plots/24h-calibration.png', bbox_inches="tight")

############################################
# Survey Data                              #
############################################
print("Survey Data")

# Alpha preferences

import pandas as pd
import plot_likert
data = pd.read_csv('models/gym/full_survey_redacted.csv')

data['Financial incentives'] = data['Financial incentives a'].fillna('') + \
                                data['Financial incentives b'].fillna('')
data['Messages that affirm your values'] = data['Messages that affirm your values a'].fillna('') + \
                                data['Messages that affirm your values b'].fillna('')
data['Notifications to plan workouts'] = data['Notifications to plan workouts a'].fillna('') + \
                                data['Notifications to plan workouts b'].fillna('')
data['Notifications to reflect'] = data['Notifications to reflect on your number of gym visits per week a'].fillna('') + \
                                data['Notifications to reflect on your levels of physical activity per week b'].fillna('')
data['Advice to increase'] = data['Advice to increase your number of gym visits a'].fillna('') + \
                            data['Advice to increase your level of physical activity b'].fillna('')

import textwrap

all_preferences = data[['Financial incentives', 'Messages that affirm your values',
                         'Notifications to plan workouts', 'Notifications to reflect', 'Advice to increase']]

plot_likert.plot_likert(all_preferences, 
                        ['Not at all increase', 'Slightly increase', 'Somewhat increase', 'Moderately increase', 'Extremely increase'], 
                        bar_labels=True,
                        figsize=(12, 6.5));
ax = plt.gca()
q = "For each of the following, indicate the degree to which you think the following reminders or incentives would motivate you to increase your [gym attendance/levels of physical activity]."
plt.title(textwrap.fill(q, width=70),
         fontweight='bold', pad=20, fontsize=30)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', which=u'both',length=0, pad=10)

labels = [item.get_text() for item in ax.get_yticklabels()]
ax.set_yticklabels([textwrap.fill(label, width=20) for label in labels], fontsize=30)
ax.set_xlabel("Number of Responses", fontweight='bold')
ax.legend(loc='center right', bbox_to_anchor=(1.15, 0.5),fancybox=True, ncol=1)

plt.savefig('plots/alpha_preferences.pdf', bbox_inches="tight")

# Beta preferences

beta_gym = data['If an artificial intelligence (AI) system were to make recommendations for increasing my gym attendance, I would listen to the AI’s advice.']
beta_non_gym = data['If an artificial intelligence (AI) system were to make recommendations for increasing my levels of physical activity, I would listen to the AI’s advice.']

beta = beta_gym.fillna('') + beta_non_gym.fillna('')
beta = beta[beta != '']

plot_likert.plot_likert(beta, 
                        plot_likert.scales.agree, 
                        bar_labels=True,
                        figsize=(12, 3))
                        
ax = plt.gca()
q = "If an artificial intelligence (AI) system were to make recommendations for increasing my [gym attendance/levels of physical activity], I would listen to the AI's advice."
plt.title(textwrap.fill(q, width=70),
         fontweight='bold', pad=20, fontsize=32)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', which=u'both',length=0, pad=10)
ax.set_yticklabels([""])
ax.set_xlabel("Number of Responses", fontweight='bold')
ax.legend(bbox_to_anchor=(1., 1), ncol=1)

plt.savefig('plots/beta_preferences.pdf', bbox_inches="tight")