import gymnasium as gym
import numpy as np
import json
import os
from PIL import Image
from collections import defaultdict
import random

# ---- Global limits ----
MAX_STEPS = 10           # max number of transitions when collecting trajectories
MAX_FRAMES = MAX_STEPS   # max number of frames we actually save per episode
TRUNCATED_STEPS = 10     # max steps when we explicitly generate truncated trajectories

# Number of episodes (trajectories) per group
EPISODES_PER_TRAIN_GROUP = 12
EPISODES_PER_TEST_GROUP = 12


# =========================
#   Map generation
# =========================
def generate_random_map(size, p=0.8, seed=None, start=(0, 0), goal=None):
    if seed is not None:
        np.random.seed(seed)
    if goal is None:
        goal = (size - 1, size - 1)  # default: bottom-right

    if size >= 6:
        p = 0.85
    elif size >= 5:
        p = 0.8

    valid = False
    attempts = 0
    while not valid and attempts < 100:
        attempts += 1
        desc = []
        for i in range(size):
            row = []
            for j in range(size):
                if (i, j) == start:
                    row.append('S')
                elif (i, j) == goal:
                    row.append('G')
                else:
                    row.append('F' if np.random.rand() < p else 'H')
            desc.append("".join(row))
        valid = is_valid_map(desc, start=start, goal=goal)

    if not valid:
        print("Warning: fall back to safe map")
        desc = generate_safe_map(size, start=start, goal=goal)

    return desc

def generate_safe_map(size, start=(0, 0), goal=None):
    if goal is None:
        goal = (size - 1, size - 1)
    desc = []
    for i in range(size):
        row = []
        for j in range(size):
            if (i, j) == start:
                row.append('S')
            elif (i, j) == goal:
                row.append('G')
            elif (i + j) % 4 == 0 and (i, j) not in (start, goal):
                row.append('H')
            else:
                row.append('F')
        desc.append("".join(row))
    return desc


def is_valid_map(desc, start=(0, 0), goal=None):
    size = len(desc)
    if goal is None:
        goal = (size - 1, size - 1)

    visited = set()
    queue = [start]

    while queue:
        i, j = queue.pop(0)
        if (i, j) in visited:
            continue
        visited.add((i, j))

        if (i, j) == goal:
            return True

        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < size and 0 <= nj < size:
                if desc[ni][nj] in ['F', 'G'] and (ni, nj) not in visited:
                    queue.append((ni, nj))
    return False



# =========================
#   Helper: numpy -> Python
# =========================
def convert_to_python_types(obj):
    """
    Convert numpy types to native Python types so they can be JSON-serialized.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


# =========================
#   Q-learning agent
# =========================
class QLearningAgent:
    """
    Improved Q-Learning agent for FrozenLake.
    """

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.99, epsilon=0.3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999
        self.q_table = defaultdict(lambda: np.random.uniform(-1, 0, n_actions))

    def get_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """
        Standard Q-learning update.
        """
        current_q = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """
        Exponential epsilon decay.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env, episodes=1000, max_steps=15):
    """
    Train a Q-learning agent on the given environment.
    Uses reward shaping based on distance to the goal.
    """
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.4
    )

    print(f"Training agent for {episodes} episodes...")
    success_count = 0
    recent_success = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Reward shaping
            if reward == 0 and done and terminated:  # fell into a hole
                reward = -1
            elif not done:
                current_row = next_state // env.unwrapped.ncol
                current_col = next_state % env.unwrapped.ncol
                goal_row = (env.unwrapped.ncol * env.unwrapped.nrow - 1) // env.unwrapped.ncol
                goal_col = (env.unwrapped.ncol * env.unwrapped.nrow - 1) % env.unwrapped.ncol
                distance = abs(current_row - goal_row) + abs(current_col - goal_col)
                reward = -0.01 * distance

            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if done:
                if reward > 0:
                    success_count += 1
                    recent_success.append(1)
                else:
                    recent_success.append(0)

                if len(recent_success) > 100:
                    recent_success.pop(0)
                break

        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:
            success_rate = success_count / (episode + 1) * 100
            recent_rate = sum(recent_success) / len(recent_success) * 100 if recent_success else 0
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Overall: {success_rate:.2f}%, "
                  f"Recent 100: {recent_rate:.2f}%, "
                  f"Epsilon: {agent.epsilon:.4f}")

    final_rate = success_count / episodes * 100
    print(f"Training completed! Final success rate: {final_rate:.2f}%")

    # Optional extra training if success rate is too low
    if final_rate < 5:
        print("Low success rate detected, continuing training...")
        extra_episodes = 20000
        for episode in range(extra_episodes):
            state, _ = env.reset()

            for step in range(max_steps):
                action = agent.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if reward == 0 and done and terminated:
                    reward = -1
                elif not done:
                    current_row = next_state // env.unwrapped.ncol
                    current_col = next_state % env.unwrapped.ncol
                    goal_row = (env.unwrapped.ncol * env.unwrapped.nrow - 1) // env.unwrapped.ncol
                    goal_col = (env.unwrapped.ncol * env.unwrapped.nrow - 1) % env.unwrapped.ncol
                    distance = abs(current_row - goal_row) + abs(current_col - goal_col)
                    reward = -0.01 * distance

                agent.update(state, action, reward, next_state, done)
                state = next_state

                if done:
                    if reward > 0:
                        success_count += 1
                    break

            agent.decay_epsilon()

            if (episode + 1) % 5000 == 0:
                current_total = episode + episodes + 1
                current_success_rate = success_count / current_total * 100
                print(f"Extra training - Episode {episode + 1}/{extra_episodes}, "
                      f"Total Success Rate: {current_success_rate:.2f}%")

        final_rate = success_count / (episodes + extra_episodes) * 100
        print(f"Extended training completed! Final success rate: {final_rate:.2f}%")

    return agent


# =========================
#   Rendering helper
# =========================
def render_frozen_lake_to_image(env):
    """
    Render FrozenLake env to a PIL image.
    """
    img_array = env.render()
    img = Image.fromarray(img_array)
    return img


# =========================
#   Trajectory collection
# =========================
def collect_trajectory(env, agent, use_random=False, max_steps=100):
    """
    Collect a single trajectory.

    Returns (state_history, action_history, reward_history, exec_state)
    exec_state in {"success", "terminated", "truncated", "unfinished"}.
    """
    obs, _ = env.reset()
    state_history = [int(obs)]
    action_history = []
    reward_history = []

    for step in range(max_steps):
        if use_random:
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs, training=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        state_history.append(int(next_obs))
        action_history.append(int(action))
        reward_history.append(float(reward))

        obs = next_obs

        if terminated or truncated:
            if terminated and reward > 0:
                exec_state = "success"
            elif terminated and reward == 0:
                exec_state = "terminated"
            else:
                exec_state = "truncated"

            return state_history, action_history, reward_history, exec_state

    return state_history, action_history, reward_history, "unfinished"


def collect_truncated_trajectory(env, max_steps=15):
    """
    Generate a 'truncated' trajectory:
    - Random policy.
    - If it terminates early (hole/goal), discard and resample.
    - If it survives max_steps, label as 'truncated'.
    """
    while True:
        obs, _ = env.reset()
        state_history = [int(obs)]
        action_history = []
        reward_history = []

        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            state_history.append(int(next_obs))
            action_history.append(int(action))
            reward_history.append(float(reward))

            if terminated:
                break

            obs = next_obs

        else:
            # Did not terminate within max_steps
            return state_history, action_history, reward_history, "truncated"
        # If terminated early, try again


def augment_failed_trajectory(env, agent):
    """
    Optional augmentation for failed trajectories (currently unused in the main pipeline).
    """
    state_history, action_history, reward_history, exec_state = collect_trajectory(
        env, agent, use_random=True
    )

    if exec_state == "terminated" and random.random() < 0.5:
        additional_actions = [int(env.action_space.sample()) for _ in range(random.randint(1, 3))]
        return state_history, action_history, reward_history, exec_state, additional_actions

    return state_history, action_history, reward_history, exec_state, []


def filter_and_enhance_trajectories(env, agent, num_samples,
                                    success_rate=0.3,
                                    truncated_ratio=0.5,
                                    max_steps=MAX_STEPS,
                                    truncated_steps=TRUNCATED_STEPS):
    """
    Collect and balance trajectories such that:
    - success trajectories ~= num_samples * success_rate
    - remaining are split between 'terminated' and 'truncated'
      according to truncated_ratio.
    """
    trajectories = []

    target_success = int(num_samples * success_rate)
    remaining = num_samples - target_success
    target_truncated = int(remaining * truncated_ratio)
    target_terminated = remaining - target_truncated

    print(f"Target counts: success={target_success}, "
          f"terminated={target_terminated}, truncated={target_truncated}")

    # 1) Collect 'success' trajectories using the trained agent
    print(f"Collecting {target_success} successful trajectories...")
    success_collected = 0
    attempts = 0
    max_attempts = target_success * 200 if target_success > 0 else 0

    while success_collected < target_success and attempts < max_attempts:
        attempts += 1
        states, actions, rewards, exec_state = collect_trajectory(
            env, agent, use_random=False, max_steps=max_steps
        )

        if exec_state == "success":
            trajectories.append({
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "exec_state": exec_state,
                "additional_actions": [],
                "label": "success",
            })
            success_collected += 1

    if success_collected < target_success:
        print(f"Warning: only collected {success_collected}/{target_success} successes.")
        deficit = target_success - success_collected
        target_terminated += deficit // 2
        target_truncated += deficit - deficit // 2

    # 2) Collect 'terminated' trajectories using a random policy
    print(f"Collecting {target_terminated} terminated trajectories...")
    terminated_collected = 0
    attempts = 0
    max_attempts = target_terminated * 300 if target_terminated > 0 else 0

    while terminated_collected < target_terminated and attempts < max_attempts:
        attempts += 1
        states, actions, rewards, exec_state = collect_trajectory(
            env, agent, use_random=True, max_steps=max_steps
        )

        if exec_state == "terminated":
            trajectories.append({
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "exec_state": exec_state,
                "additional_actions": [],
                "label": "failure",
            })
            terminated_collected += 1

    if terminated_collected < target_terminated:
        print(f"Warning: only collected {terminated_collected}/{target_terminated} terminated.")

    # 3) Collect 'truncated' trajectories using a special random generator
    print(f"Collecting {target_truncated} truncated trajectories...")
    truncated_collected = 0
    attempts = 0
    max_attempts = target_truncated * 300 if target_truncated > 0 else 0

    while truncated_collected < target_truncated and attempts < max_attempts:
        attempts += 1
        result = collect_truncated_trajectory(env, max_steps=truncated_steps)
        if result is None:
            continue
        states, actions, rewards, exec_state = result
        trajectories.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "exec_state": exec_state,
            "additional_actions": [],
            "label": "truncated",
        })
        truncated_collected += 1

    if truncated_collected < target_truncated:
        print(f"Warning: only collected {truncated_collected}/{target_truncated} truncated.")

    print(f"Collected total {len(trajectories)}/{num_samples} trajectories.")
    return trajectories[:num_samples]


# =========================
#   Saving trajectories
# =========================
def save_trajectory_data(env, trajectories, level_split_dir, json_data,
                         group_id=0, max_frames=None):
    """
    Save trajectories for one group (one map) into:
    - image folders: level_split_dir/<group_id>/<data_id>/<step>.png
    - JSON structure:

      {
        "0": {
          "actions": [... list of list ...],
          "states": [... list of list ...],
          "rewards": [... list of float, one per episode ...],
          "data_id": [... list of int ...],
          "env_desc": "FHFHFGFSF",
          "exec_states": [... list of str ...],
          "additional_actions": [... list of list ...]
        },
        ...
      }

    json_data is a dict shared across groups for this split (train/test).
    """
    # Decode environment layout
    env_desc_decoded = [[cell.decode('utf-8') if isinstance(cell, bytes) else cell
                         for cell in row] for row in env.unwrapped.desc]
    env_desc_str = "".join(["".join(row) for row in env_desc_decoded])

    nrow = len(env_desc_decoded)
    ncol = len(env_desc_decoded[0])

    def is_terminal_state(s):
        r = s // ncol
        c = s % ncol
        ch = env_desc_decoded[r][c]
        return ch in ('H', 'G')

    # To avoid duplicate episodes (same states + actions + exec_state) inside this group
    seen_signatures = set()

    grouped_actions = []
    grouped_states = []
    grouped_rewards = []          # one scalar per episode
    grouped_exec_states = []
    grouped_additional_actions = []
    grouped_data_ids = []

    # Episodes inside this group will be indexed 0, 1, 2, ...
    next_data_id = 0

    # Directory for this group under the given split (train/test)
    group_dir = os.path.join(level_split_dir, str(group_id))
    os.makedirs(group_dir, exist_ok=True)

    for traj in trajectories:
        orig_states = traj["states"]
        orig_actions = traj["actions"]
        orig_rewards = traj["rewards"]
        exec_state = traj["exec_state"]
        add_actions = traj["additional_actions"]

        if len(orig_states) == 0:
            continue

        # 1) Cut at the first terminal tile (H/G)
        term_states = [orig_states[0]]
        term_actions = []
        term_rewards = []
        terminated_cut = False

        for i in range(len(orig_actions)):
            next_state = orig_states[i + 1]
            term_states.append(next_state)
            term_actions.append(orig_actions[i])
            term_rewards.append(orig_rewards[i])

            if is_terminal_state(next_state):
                terminated_cut = True
                break

        if not terminated_cut and len(orig_states) > 1 and len(term_states) == 1:
            # Single-state trajectory edge case
            term_states = orig_states[:]
            term_actions = orig_actions[:]
            term_rewards = orig_rewards[:]

        if len(term_states) == 0:
            term_states = [orig_states[0]]
            term_actions = []
            term_rewards = []

        # 2) Remove "standing still" steps (no state change)
        dedup_states = [term_states[0]]
        dedup_actions = []
        dedup_rewards = []

        for i in range(len(term_actions)):
            prev_state = dedup_states[-1]
            next_state = term_states[i + 1]
            if next_state != prev_state:
                dedup_states.append(next_state)
                dedup_actions.append(term_actions[i])
                dedup_rewards.append(term_rewards[i])

        if len(dedup_states) == 0:
            dedup_states = [term_states[0]]
            dedup_actions = []
            dedup_rewards = []

        # 3) Optional frame cap
        if max_frames is not None:
            n_frames = min(max_frames, len(dedup_states))
            states = dedup_states[:n_frames]
            actions = dedup_actions[:max(0, n_frames - 1)]
            # Step-level rewards not used in JSON any more; truncate just for safety
            _ = dedup_rewards[:max(0, n_frames - 1)]
        else:
            states = dedup_states
            actions = dedup_actions

        if len(states) == 0:
            continue

        # 4) De-duplicate episodes within this group
        signature = (tuple(states), tuple(actions), exec_state)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        # 5) Episode-level reward (1.0 for success, otherwise 0.0)
        ep_reward = 1.0 if exec_state == "success" else 0.0

        # 6) Save images to group_dir/<data_id>/<step>.png
        episode_dir = os.path.join(group_dir, str(next_data_id))
        os.makedirs(episode_dir, exist_ok=True)

        for step_idx, state in enumerate(states):
            env.reset()
            env.unwrapped.s = state

            if step_idx == 0:
                env.unwrapped.lastaction = None
            else:
                env.unwrapped.lastaction = actions[step_idx - 1]

            img = render_frozen_lake_to_image(env)
            img_path = os.path.join(episode_dir, f"{step_idx}.png")
            img.save(img_path)

        # 7) Append to group lists
        grouped_states.append(states)
        grouped_actions.append(actions)
        grouped_rewards.append(ep_reward)
        grouped_exec_states.append(exec_state)
        grouped_additional_actions.append(add_actions)
        grouped_data_ids.append(next_data_id)

        next_data_id += 1

    group_key = str(group_id)
    json_data[group_key] = convert_to_python_types({
        "actions": grouped_actions,
        "states": grouped_states,
        "rewards": grouped_rewards,
        "data_id": grouped_data_ids,
        "env_desc": env_desc_str,
        "exec_states": grouped_exec_states,
        "additional_actions": grouped_additional_actions,
    })

    print(f"  Total saved (unique) trajectories in group {group_key}: {len(grouped_states)}")


# =========================
#   Main dataset generator
# =========================
def create_env_for_size(map_size, group_seed):
    """
    Create a FrozenLake-v1 environment for a given map size and group.

    For size 4 and 8, use the built-in maps.
    For other sizes, generate a custom map.
    """
    if map_size == 4:
        env = gym.make('FrozenLake-v1',
                       map_name="4x4",
                       is_slippery=True,
                       render_mode='rgb_array')
        desc = None
    elif map_size == 8:
        env = gym.make('FrozenLake-v1',
                       map_name="8x8",
                       is_slippery=True,
                       render_mode='rgb_array')
        desc = None
    else:
        # desc = generate_random_map(map_size, seed=group_seed)
        start = (0, 2)
        goal = (2, 0)
        desc = generate_random_map(map_size, seed=group_seed, start=start, goal=goal)
        env = gym.make('FrozenLake-v1',
                       desc=desc,
                       is_slippery=True,
                       render_mode='rgb_array')
    return env, desc


def generate_dataset_for_size(map_size, num_train_groups, num_test_groups,
                              base_output_dir, data_subdir):
    """
    Generate dataset for a given map size (level).

    num_train_groups: how many training groups (maps)
    num_test_groups: how many test groups (maps)
    """
    print(f"\n{'=' * 60}")
    print(f"Generating dataset for {map_size}x{map_size} map")
    print(f"{'=' * 60}")

    level_name = f"level{map_size}"
    level_dir = os.path.join(base_output_dir, level_name)
    os.makedirs(level_dir, exist_ok=True)

    # Directories for this level
    train_dir = os.path.join(level_dir, "train")
    test_dir = os.path.join(level_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Training episodes per group
    training_episodes = {
        3: 2000,
        4: 3000,
        5: 5000,
        6: 70000,
    }

    # ---- Training groups ----
    train_json_data = {}
    total_train_trajs = 0

    print("\n--- Generating Training Groups ---")
    for group_id in range(num_train_groups):
        print(f"\n[Train] Group {group_id}")

        # Create env and train agent for this group
        group_seed = 42 + group_id
        train_env, _ = create_env_for_size(map_size, group_seed)
        train_env.reset(seed=group_seed)

        agent = train_agent(
            train_env,
            episodes=training_episodes.get(map_size, 3000),
            max_steps=MAX_STEPS
        )

        # Collect trajectories for this group
        train_trajectories = filter_and_enhance_trajectories(
            train_env,
            agent,
            num_samples=EPISODES_PER_TRAIN_GROUP,
            success_rate=0.3
        )
        total_train_trajs += len(train_trajectories)

        # Save images + JSON for this group
        save_trajectory_data(
            train_env,
            train_trajectories,
            train_dir,
            train_json_data,
            group_id=group_id,
            max_frames=MAX_FRAMES,
        )

        train_env.close()

    train_json_path = os.path.join(train_dir, "train_data.json")
    with open(train_json_path, 'w') as f:
        json.dump(train_json_data, f, indent=2)
    print(f"\n✓ Training groups completed: {total_train_trajs} episodes total")

    # ---- Test groups ----
    test_json_data = {}
    total_test_trajs = 0

    print("\n--- Generating Test Groups ---")
    for group_id in range(num_test_groups):
        print(f"\n[Test] Group {group_id}")

        group_seed = 9999 + group_id
        test_env, _ = create_env_for_size(map_size, group_seed)
        test_env.reset(seed=group_seed)

        # Train a fresh agent for test groups as well
        agent = train_agent(
            test_env,
            episodes=training_episodes.get(map_size, 3000),
            max_steps=MAX_STEPS
        )

        test_trajectories = filter_and_enhance_trajectories(
            test_env,
            agent,
            num_samples=EPISODES_PER_TEST_GROUP,
            success_rate=0.3
        )
        total_test_trajs += len(test_trajectories)

        save_trajectory_data(
            test_env,
            test_trajectories,
            test_dir,
            test_json_data,
            group_id=group_id,
            max_frames=MAX_FRAMES,
        )

        test_env.close()

    test_json_path = os.path.join(test_dir, "test_data.json")
    with open(test_json_path, 'w') as f:
        json.dump(test_json_data, f, indent=2)
    print(f"\n✓ Test groups completed: {total_test_trajs} episodes total")

    return total_train_trajs, total_test_trajs


# =========================
#   Entry point
# =========================
if __name__ == "__main__":
    base_output_dir = "frozenlake"
    data_subdir = "data_samples"  # currently unused but kept for compatibility

    # dataset_config: number of groups (maps) per level
    dataset_config = {
        3: {"train": 50, "test": 10},  
        4: {"train": 50, "test": 10},  
        5: {"train": 50, "test": 10},  
        6: {"train": 50, "test": 10},  
    }

    random.seed(42)
    np.random.seed(42)

    total_train = 0
    total_test = 0
    results = {}

    for map_size, config in dataset_config.items():
        try:
            train_count, test_count = generate_dataset_for_size(
                map_size=map_size,
                num_train_groups=config["train"],
                num_test_groups=config["test"],
                base_output_dir=base_output_dir,
                data_subdir=data_subdir
            )
            total_train += train_count
            total_test += test_count
            results[map_size] = {"train": train_count, "test": test_count}
        except Exception as e:
            print(f"Error generating dataset for {map_size}x{map_size}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETED")
    print("=" * 60)
    print("\nResults by map size (number of trajectories):")
    for map_size, counts in results.items():
        print(f"  {map_size}x{map_size}: Train={counts['train']}, Test={counts['test']}")
    print(f"\nTotal Training Trajectories: {total_train}")
    print(f"Total Test Trajectories: {total_test}")
    print(f"Total Trajectories: {total_train + total_test}")
    print(f"\nDataset structure:")
    print(f"  Images: {base_output_dir}/level*/(train|test)/<group_id>/<data_id>/<step>.png")
    print(f"  Metadata: {base_output_dir}/level*/(train|test)/*_data.json")
    print("=" * 60)
