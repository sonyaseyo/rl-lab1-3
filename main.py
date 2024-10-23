import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='ansi')

#transitions = env.unwrapped.P

gamma = 0.75
num_states = env.observation_space.n
num_actions = env.action_space.n

value_table = np.zeros(num_states)

# рівноймовірна стратегія
policy = np.ones([num_states, num_actions]) / num_actions

def policy_evaluation(policy, env, gamma, episodes=1000):
    value_table = np.zeros(num_states)
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = np.random.choice(num_actions, p=policy[state])
            next_state, reward, done, truncated, _ = env.step(action)
            G = reward + gamma * value_table[next_state]
            value_table[state] += G
            state = next_state
    return value_table / episodes


v_pi = policy_evaluation(policy, env, gamma)

print("ф-ція ціни стану для рівноймовірної стратегії:")
print(v_pi.reshape(4, 4))

def compute_action_value_function(policy, env, gamma, episodes=1000):
    action_value_table = np.zeros((num_states, num_actions))
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = np.random.choice(num_actions, p=policy[state])
            next_state, reward, done, truncated, _ = env.step(action)
            G = reward + gamma * np.max(action_value_table[next_state])
            action_value_table[state, action] += G
            state = next_state
    return action_value_table / episodes

q_pi = compute_action_value_function(policy, env, gamma)

print("ф-ція ціни дії-стану для рівноймовірної стратегії:")
print(q_pi)


def equiprobable(state, env):
    return np.random.choice(env.action_space.n)


state = env.reset()[0]
action = equiprobable(state, env)
print(f"випадково обрана дія: {action}")


def get_episode(env):
    episode = []
    state = env.reset()[0]
    done = False
    while not done:
        action = equiprobable(state, env)
        next_state, reward, done, truncated, _ = env.step(action)
        episode.append((state, action, reward, next_state, done, truncated))
        state = next_state
    return episode


episode = get_episode(env)
print(f"епізод (state, action, reward, next_state, done, truncated): ")
for step in episode:
    print(step)


def run_episodes(env, num_episodes=100):
    rewards = []
    lengths = []
    for _ in range(num_episodes):
        episode = get_episode(env)
        rewards.append(sum([step[2] for step in episode]))  # Сумарна винагорода
        lengths.append(len(episode))  # Тривалість епізоду
    return rewards, lengths


rewards, lengths = run_episodes(env)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title("Сумарна винагорода")
plt.subplot(1, 2, 2)
plt.plot(lengths)
plt.title("Тривалість епізодів")
plt.show()


def value_iteration(env, gamma=0.75, theta=1e-10, max_iterations=1000):
    value_table = np.zeros(num_states)
    for i in range(max_iterations):
        delta = 0
        for state in range(num_states):
            q_values = []
            for action in range(num_actions):
                q_value = 0
                for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                    q_value += prob * (reward + gamma * value_table[next_state])
                q_values.append(q_value)
            max_value = max(q_values)
            delta = max(delta, np.abs(max_value - value_table[state]))
            value_table[state] = max_value
        if delta < theta:
            break
    return value_table

# оптимальна ф-ція ціни стану
v_star = value_iteration(env)
print("оптимальна ф-ція ціни стану:")
print(v_star.reshape(4, 4))


def extract_policy(value_table, gamma=0.75):
    policy = np.zeros([num_states, num_actions])
    for state in range(num_states):
        q_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                q_values[action] += prob * (reward + gamma * value_table[next_state])
        best_action = np.argmax(q_values)
        policy[state] = np.eye(num_actions)[best_action]
    return policy

# оптимальна стратегія
optimal_policy = extract_policy(v_star)

print("оптимальна стратегія:")
print(optimal_policy)


print("ф-ція ціни стану для оптимальної стратегії:")
print(v_star.reshape(4, 4))

print("Початкова ф-ція ціни стану (для рівноймовірної стратегії):")
print(v_pi.reshape(4, 4))


def get_episode_with_policy(env, policy):
    episode = []
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(policy[state])  # Вибір дії відповідно до оптимальної стратегії
        next_state, reward, done, truncated, _ = env.step(action)
        episode.append((state, action, reward, next_state, done, truncated))
        state = next_state
    return episode


# 15 епізодів зі стратегією π*
rewards_opt, lengths_opt = run_episodes(env, num_episodes=15)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_opt)
plt.title("сумарна винагорода (оптимальна стратегія)")
plt.subplot(1, 2, 2)
plt.plot(lengths_opt)
plt.title("тривалість епізодів (оптимальна стратегія)")
plt.show()

# порівняння результатів з попередніми
print("порівняння середньої винагороди для оптимальної стратегії та рівноймовірної стратегії:")
print(f"середня винагорода (оптимальна стратегія): {np.mean(rewards_opt)}")
print(f"середня винагорода (рівноймовірна стратегія): {np.mean(rewards)}")


def compute_optimal_action_value_function(env, value_table, gamma):
    action_value_table = np.zeros((num_states, num_actions))
    for state in range(num_states):
        for action in range(num_actions):
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                action_value_table[state][action] += prob * (reward + gamma * value_table[next_state])
    return action_value_table


# ф-ція ціни дії-стану для оптимальної стратегії
q_star = compute_optimal_action_value_function(env, v_star, gamma)


print("Оптимальна функція ціни дії-стану:")
print(q_star)


def eps_greedy_policy(q_values, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values))  # випадкова дія
    else:
        return np.argmax(q_values)  # найкраща дія


def expected_sarsa(env, gamma=0.75, epsilon=0.1, alpha=0.1, episodes=1000):
    q_table = np.zeros((num_states, num_actions))
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = eps_greedy_policy(q_table[state], epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            expected_q = np.mean([q_table[next_state, a] for a in range(num_actions)])
            q_table[state, action] += alpha * (reward + gamma * expected_q - q_table[state, action])
            state = next_state
    return q_table


q_sarsa = expected_sarsa(env, gamma=0.75, epsilon=0.1)


print("ф-ція ціни дії-стану за методом Expected SARSA:")
print(q_sarsa)


print("порівняння ф-цій ціни дії-стану:")
print(f"еxpected SARSA: \n{q_sarsa}")
print(f"оптимальна стратегія: \n{q_star}")
print(f"рівноймовірна стратегія: \n{q_pi}")


def create_policy_from_q(q_table, epsilon=0.1):
    policy = np.zeros([num_states, num_actions])
    for state in range(num_states):
        action = eps_greedy_policy(q_table[state], epsilon)
        policy[state] = np.eye(num_actions)[action]
    return policy


# нова стратегія π2 на основі q*
policy_pi2 = create_policy_from_q(q_sarsa)

print("стратегія π2 на основі q* (Expected SARSA):")
print(policy_pi2)


# 100 епізодів зі стратегією π2
rewards_pi2, lengths_pi2 = run_episodes(env, num_episodes=100)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_pi2)
plt.title("сумарна винагорода (стратегія π2)")
plt.subplot(1, 2, 2)
plt.plot(lengths_pi2)
plt.title("тривалість епізодів (стратегія π2)")
plt.show()


print(f"середня винагорода (стратегія π2): {np.mean(rewards_pi2)}")
print(f"середня винагорода (оптимальна стратегія): {np.mean(rewards_opt)}")
print(f"середня винагорода (рівноймовірна стратегія): {np.mean(rewards)}")








