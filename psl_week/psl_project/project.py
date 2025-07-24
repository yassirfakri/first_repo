import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb as debugger
import seaborn as sns
from random import choices

pdb = debugger.set_trace


data_path = './data/group_2_responses.csv'
reward_data_path = './data/group_2_reward.csv'

data = pd.read_csv(data_path, header=None)
rew_data = pd.read_csv(reward_data_path, header=None)

# 80 subjects + 200 trials => df.shape = (80, 200)


def scatter_data(subject: int, start: int = 0, end: int = 200, group: int = 2):
    df = pd.read_csv(f'./data/group_{group}_responses.csv', header=None)
    rew_df = pd.read_csv(f'./data/group_{group}_reward.csv', header=None)
    plt.clf()
    plt.scatter(np.arange(start, end),
                df.iloc[subject-1, start: end], color='red')
    plt.plot(np.arange(start, end),
             rew_df.iloc[subject-1, start: end], color='blue',)
    plt.legend(['Choice', 'Reward'])
    plt.title(f'Plot for subject {subject} in group {group}')
    plt.xlabel('trial number')
    plt.show()


def calculate_mean_value(group: int = 2) -> float:
    rew_df = pd.read_csv(f'./data/group_{group}_reward.csv', header=None)
    return np.mean(rew_df)


def calculate_number_of_switches(group: int = 2) -> list[int]:
    rew_df = pd.read_csv(f'./data/group_{group}_responses.csv', header=None)
    return [sum(np.abs(np.diff(rew_df))[i]) for i in range(rew_df.shape[0])]


def plot_reward_distribution(group: int = 2) -> None:
    rew_df = pd.read_csv(f'./data/group_{group}_reward.csv', header=None)
    mean_rewards_per_subject = [np.mean(rew_df.iloc[i, :])
                                for i in range(rew_df.shape[0])]
    sns.kdeplot(mean_rewards_per_subject, bw_adjust=0.5, color='blue')
    plt.show()
    plt.clf()


def scatter_list(L: list) -> None:
    plt.scatter(range(len(L)), L)
    plt.show()
    plt.clf()


def plot_hist(data: list) -> None:
    plt.hist(data, bins=len(data))
    plt.show()
    plt.clf()


def sample_normalWithBoundaries(loc, scale, n, bound):
    list = []
    for _ in range(n):
        while True:
            x = np.random.normal(loc, scale, size=1)
            if np.abs(x) < bound:
                break
        list.append(x[0])
    return (list)


def TwoArmed_Switch(N_trial, mean, err):
    First_arm = np.hstack([np.array([x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)]), np.array([-x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)]),
                          np.array([x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)]), np.array([-x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)])])
    Second_arm = np.hstack([np.array([-x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)]), np.array([x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)]),
                           np.array([-x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)]), np.array([x for x in sample_normalWithBoundaries(mean, err, int(N_trial/4), 1)])])
    Res = np.array([First_arm, Second_arm])
    return (Res)


def simulate_wsls_model(rewards: list[list]) -> list[int]:
    choices = [0]
    rewards = [(a, b) for a, b in zip(rewards[0], rewards[1])]
    for i, e in enumerate(rewards):
        if e[choices[i-1]] < 0:
            choices.append(1-choices[i-1])
        else:
            choices.append(choices[i-1])
    return choices[:-1]


def WS_LS(Bandit, epsilon):
    n_trial = np.shape(Bandit)[1]
    trial_number = 0
    First_element = np.random.randint(0, 2, 1)[0]
    response = [First_element]
    reward = [Bandit[response[-1], trial_number]]
    for i in range(1, n_trial):
        if np.random.uniform(low=0.0, high=1.0, size=1)[0] > epsilon:
            if reward[-1] > 0:
                response.append(response[-1])
            else:
                response.append(np.abs(response[-1]-1))
        else:
            response.append(np.random.randint(0, 2, 1)[0])
        reward.append(Bandit[response[-1], i])
    return (response, reward)


def RW_softmax(Bandit, alpha, beta):
    # Initialize
    n_trial = np.shape(Bandit)[1]
    trial_number = 0
    First_element = np.random.randint(0, 2, 1)[0]
    response = [First_element]
    reward = [Bandit[response[-1], trial_number]]
    if First_element == 0:
        Q0 = [0 + alpha*(reward[-1] - 0)]
        Q1 = [0]
    elif First_element == 1:
        Q1 = [0 + alpha*(reward[-1] - 0)]
        Q0 = [0]
    p0 = [np.exp(beta*Q0[-1])/(np.exp(beta*Q0[-1]) + np.exp(beta*Q1[-1]))]

    # Run for each trial
    for i in range(1, n_trial):
        if np.random.uniform(low=0.0, high=1.0, size=1)[0] < p0[-1]:
            response.append(0)
        else:
            response.append(1)
        # Reward and update the Q and p values
        reward.append(Bandit[response[-1], i])
        if response[-1] == 0:
            Q0.append(Q0[-1] + alpha * (reward[-1] - Q0[-1]))
            Q1.append(Q1[-1])
        elif response[-1] == 1:
            Q1.append(Q1[-1] + alpha * (reward[-1] - Q1[-1]))
            Q0.append(Q0[-1])
        p0.append(np.exp(beta * Q0[-1]) /
                  (np.exp(beta * Q0[-1]) + np.exp(beta * Q1[-1])))
    return (response, reward, Q0, Q1, p0)


def calculate_log_likelihood_wsls(reward: list[float], response: list[int], epsilon: float) -> float:
    LL = -np.log(0.5)
    for i in range(1, len(response)):
        if ((reward[i - 1] > 0) & (response[i - 1] == response[i])) | ((reward[i - 1] < 0) & (response[i - 1] != response[i])):
            LL += -np.log((1 - epsilon / 2))
        else:
            LL += -np.log((epsilon / 2))
    return (LL)


def estimate_likelihood_RW_softmax(params, response, reward):
    alpha, beta = params
    LL = -np.log(0.5)
    Q0 = 0
    Q1 = 0
    for i in range(1, np.shape(response)[0]):
        if response[i-1] == 0:
            Q0 = Q0 + alpha * (reward[i-1] - Q0)
        elif response[i-1] == 1:
            Q1 = Q1 + alpha * (reward[i-1] - Q1)
        p0 = np.exp(beta * Q0) / ((np.exp(beta * Q0) + np.exp(beta * Q1)))
        if (response[i] == 0):
            LL += -np.log(p0)
        else:
            LL += -np.log(1-p0)
    return (LL)


if __name__ == '__main__':
    # scatter_data(subject=8)

    mean_rewards_per_subject = [np.mean(rew_data.iloc[i, :])
                                for i in range(rew_data.shape[0])]

    mean_rewards_per_trial = [np.mean(rew_data.iloc[:, i])
                              for i in range(rew_data.shape[1])]

    # mean rewards per subject for each group
    for group in range(2, 3):
        rew_df = pd.read_csv(f'./data/group_{group}_reward.csv', header=None)
        mean_rewards_per_subject = [np.mean(rew_data.iloc[i, :])
                                    for i in range(rew_data.shape[0])]

        mean_rewards_per_trial = [np.mean(rew_data.iloc[:, i])
                                  for i in range(rew_data.shape[1])]

        # plt.plot(np.arange(len(mean_rewards_per_subject)),
        #          mean_rewards_per_subject, color='green')
        # plt.plot(np.arange(80), [0]*80, color='red')
        # plt.xlabel('Subjects')
        # plt.title("Mean reward per subject")
        # plt.show()

        # plt.plot(np.arange(len(mean_rewards_per_trial)),
        #          mean_rewards_per_trial, color='blue')
        # plt.plot(np.arange(200), [0]*200, color='red')
        # plt.xlabel('Trials')
        # plt.title("Mean reward per trial")
        # plt.show()

    mean_value_map = {i: calculate_mean_value(i) for i in range(1, 6)}

    num_switches = calculate_number_of_switches()
    mean_num_switches = np.mean(num_switches)

    # Plot of number of switches per subject
    # plt.scatter(range(len(num_switches)), num_switches)
    # plt.xlabel('Subjects')
    # plt.title("Number of switches per subject")
    # plt.plot(range(len(num_switches)), [
    #          mean_num_switches]*len(num_switches), color='red')
    # plt.show()

    # var_res = sum((e - mean_num_switches) **
    #               2 for e in num_switches) / len(num_switches)
    # print(var_res)

    # Win-stay-lose-shift model without noise
    # mean_rewards_per_subject = []
    # total_rewards = []

    # for i in range(800):
    #     rewards = TwoArmed_Switch(200, 0.2, 0.4)
    #     wsls_choices = simulate_wsls_model(rewards)
    #     subject_rewards = [rewards[wsls_choices[i]][i] for i in range(200)]
    #     mean_reward = np.mean(subject_rewards)
    #     mean_rewards_per_subject.append(mean_reward)
    #     total_rewards.append(subject_rewards)

    total_rewards = []
    for subject in range(10):
        rewards = rew_data.iloc[subject, :]
        total_reward = np.mean(rewards)
        total_rewards.append(total_reward)

    plt.scatter(range(1, 11), total_rewards)
    plt.plot(range(1, 11), [0]*10, color='red')
    plt.show()

    n_subjects = 10
    min_epsilons = []
    for subject in range(n_subjects):
        log_ps = []
        epsilons = np.linspace(0, 1, 50)
        for eps in epsilons:
            log_p = calculate_log_likelihood_wsls(
                rew_data.iloc[subject, :], data.iloc[subject, :], eps)
            log_ps.append(log_p)

        plt.plot(epsilons, log_ps)
        min_p_index = log_ps.index(min(log_ps))
        min_epsilons.append(epsilons[min_p_index])
    pdb()
    plt.legend([f'subject {i+1}' for i in range(n_subjects)])
    plt.xlabel('epsilon')
    plt.ylabel('log likelihood')
    plt.title('Log-likelihood noisy WSLS model')
    plt.show()
    pdb()

    log_ps = []
    alphas = np.linspace(0, 1, 50)
    beta = 30
    n_subjects = 10
    for subject in range(n_subjects):
        for alpha in alphas:
            log_p = estimate_likelihood_RW_softmax(params=(alpha, beta),
                                                   reward=rew_data.iloc[subject, :],
                                                   response=data.iloc[subject, :])
            log_ps.append(log_p)
        plt.plot(alphas, epsilons)
    plt.legend([f'subject {i+1}' for i in range(n_subjects)])
    plt.xlabel('alpha')
    plt.ylabel('-log likelihood')
    plt.title('-Log likelihood RW model')
    plt.show()

    # Noisy Win-stay-lose-shift model
    # mean_rewards_per_subject = []
    # total_rewards = []

    # for i in range(800):
    #     rewards = TwoArmed_Switch(200, 0.2, 0.4)
    #     wsls_result = WS_LS(rewards, 0.8)
    #     subject_rewards = wsls_result[1]
    #     mean_reward = np.mean(subject_rewards)
    #     mean_rewards_per_subject.append(mean_reward)
    #     total_rewards.append(subject_rewards)

    # rewards_df = pd.DataFrame(total_rewards)

    # Rescorla Wagner model
    # mean_rewards_per_subject = []
    # total_rewards = []

    # for i in range(800):
    #     rewards = TwoArmed_Switch(200, 0.2, 0.4)
    #     rw_results = RW_softmax(rewards, alpha=1, beta=40)
    #     subject_rewards = rw_results[1]
    #     mean_reward = np.mean(subject_rewards)
    #     mean_rewards_per_subject.append(mean_reward)
    #     total_rewards.append(subject_rewards)

    # rewards_df = pd.DataFrame(total_rewards)

    plt.plot(np.arange(200), [np.mean(rewards_df.iloc[:, i])
             for i in range(200)])
    plt.plot(np.arange(200), [0]*200, color='red')
    plt.xlabel('Trials')
    plt.title('Mean reward per trial (RW model)')
    plt.show()

    pdb()

    # mean_values = [calculate_mean_value(i) for i in range(1, 6)]
    # plt.bar([f'group {i}' for i in range(1, 6)],
    #         mean_values, width=0.4, color='green')
    # plt.title('Mean reward per group')
    # plt.show()
