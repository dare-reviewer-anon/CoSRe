# FrozenLake Dataset Generation (for COSRE Experiments)

This folder contains the code we use to **generate the FrozenLake trajectory datasets** used in our experiments.
The script procedurally generates maps (for sizes 3×3, 4×4, 5×5, 6×6), trains a **Q-learning agent** on each map, and then collects **balanced trajectories** with labels:

* `success` (reach goal),
* `failure` (fall into a hole),
* `truncated` (survive for a fixed number of steps without termination).

Images are rendered for every state, and all metadata is stored in JSON.

---

## 1. Environment Setup

### 1.1. Python version

We tested with:

* Python **3.10**
  (Any recent 3.9–3.11 should work.)

### 1.2. Create a virtual environment (recommended)

Using `conda`:

```bash
conda create -n frozenlake python=3.10 -y
conda activate frozenlake
```

Or using `venv`:

```bash
python -m venv frozenlake_env
source frozenlake_env/bin/activate     # Linux/macOS
# .\frozenlake_env\Scripts\activate    # Windows
```

### 1.3. Install dependencies

Install the required Python packages:

```bash
pip install gymnasium[toy_text] numpy pillow
```

* `gymnasium[toy_text]` – provides the `FrozenLake-v1` environment.
* `numpy` – numeric operations.
* `Pillow` – saving rendered frames as PNG images.

If your environment does not have RGB rendering by default, you may also need:

```bash
pip install matplotlib  # sometimes needed by gymnasium rendering backends
```

---

## 2. File Structure

Assume the dataset generation script is saved as:

```text
generate_frozenlake_dataset.py
```

After running the script, the directory structure will look like:

```text
frozenlake/
  level3/
    train/
      0/              # group_id = 0
        0/            # data_id = 0 (episode 0)
          0.png
          1.png
          ...
        1/
          0.png
          ...
      1/
        ...
      train_data.json
    test/
      0/
        ...
      test_data.json

  level4/
    train/
      ...
    test/
      ...
  level5/
    ...
  level6/
    ...
```

At the end of the script, you will also see a summary printed in the console with the number of trajectories per level and split.

---

## 3. How to Generate the Datasets

### 3.1. Default configuration

In the `__main__` block of `generate_frozenlake_dataset.py` we use:

```python
base_output_dir = "frozenlake"
data_subdir = "data_samples"  # currently unused but kept for compatibility

dataset_config = {
    3: {"train": 10, "test": 5},  # 10 train groups, 5 test groups for 3x3
    4: {"train": 10, "test": 5},  # 4x4
    5: {"train": 10, "test": 5},  # 5x5
    6: {"train": 10, "test": 5},  # 6x6
}
```

* Each **map size** (level) has `train` and `test` **groups**.
* A **group** corresponds to a single FrozenLake map (layout) with multiple episodes/trajectories generated on it.
* The number of trajectories per group is set via:

```python
EPISODES_PER_TRAIN_GROUP = 12
EPISODES_PER_TEST_GROUP = 12
```

Hence, for example, level 3 (3×3) has:

* 10 train groups × 12 episodes each = 120 train trajectories,
* 5 test groups × 12 episodes each = 60 test trajectories.

### 3.2. Global limits

At the top of the script:

```python
MAX_STEPS = 10           # max number of transitions when collecting trajectories
MAX_FRAMES = MAX_STEPS   # max number of frames we actually save per episode
TRUNCATED_STEPS = 10     # max steps when we explicitly generate truncated trajectories
```

* `MAX_STEPS` controls how long the agent is allowed to act when collecting a trajectory.
* `MAX_FRAMES` caps the number of saved frames per episode.
* `TRUNCATED_STEPS` controls how long we run trajectories that are explicitly labeled as `"truncated"`.

### 3.3. Reproducibility (seeds)

In `__main__`, we fix `random` and `numpy` seeds:

```python
random.seed(42)
np.random.seed(42)
```

Additionally, for each **group**, we use deterministic seeds for the environment:

* Train groups: `group_seed = 42 + group_id`
* Test groups:  `group_seed = 9999 + group_id`

This makes both map generation and agent training deterministic across runs (given the same library versions).

### 3.4. Run the script

From the directory containing `generate_frozenlake_dataset.py`, simply run:

```bash
python generate_frozenlake_dataset.py
```

This will:

1. Loop over all map sizes in `dataset_config` (3, 4, 5, 6).
2. For each size:

   * Create train/test directories under `frozenlake/level<size>/`.
   * Generate `train_data.json` and `test_data.json`.
   * Save image frames for each trajectory.

If any error happens for a particular size, the script will print the stack trace and continue with the remaining sizes.

---

## 4. Dataset Generation Logic

### 4.1. Map generation

For each map size, we create an environment using:

```python
def create_env_for_size(map_size, group_seed):
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
        start = (0, 2)
        goal = (2, 0)
        desc = generate_random_map(map_size, seed=group_seed, start=start, goal=goal)
        env = gym.make('FrozenLake-v1',
                       desc=desc,
                       is_slippery=True,
                       render_mode='rgb_array')
    return env, desc
```

* For size **4** and **8**, we use the **built-in** `FrozenLake-v1` maps.
* For sizes **3, 5, 6**, we use a **custom random map** via:

  * `generate_random_map`, with a validity check (`is_valid_map`) to ensure there is a path from start to goal.
  * If we fail to find a valid map within 100 attempts, we fall back to `generate_safe_map` (a structured but still non-trivial layout).

The random map uses:

* `S` – start
* `G` – goal
* `F` – frozen (safe) tile
* `H` – hole

### 4.2. Training the Q-learning agent

For each group (each map), we train a **separate Q-learning agent**:

```python
training_episodes = {
    3: 2000,
    4: 3000,
    5: 5000,
    6: 70000,
}
```

Example call:

```python
agent = train_agent(
    train_env,
    episodes=training_episodes.get(map_size, 3000),
    max_steps=MAX_STEPS
)
```

Key points:

* The agent is **tabular Q-learning**, with:

  * `learning_rate = 0.1`,
  * `discount_factor = 0.95`,
  * `epsilon`-greedy exploration with exponential decay.
* We add **reward shaping**:

  * Negative reward when falling into a hole.
  * Small negative reward proportional to the Manhattan distance to the goal when the episode continues, encouraging progress towards the goal.
* If the final success rate is too low (< 5%), we automatically perform **extra training** (up to 20,000 additional episodes) to obtain a usable policy.

### 4.3. Trajectory collection and balancing

We collect trajectories using a combination of **trained policy** and **random policy**:

```python
filter_and_enhance_trajectories(
    env,
    agent,
    num_samples=EPISODES_PER_TRAIN_GROUP or EPISODES_PER_TEST_GROUP,
    success_rate=0.3,
    truncated_ratio=0.5,
)
```

Target counts:

* `success` trajectories: `num_samples * success_rate`.
* Remaining episodes split between `failure` (`terminated`) and `truncated` according to `truncated_ratio`.

We use:

* `collect_trajectory(...)`:

  * If `use_random=False`: follow the **trained agent** -> typically produces successful trajectories.
  * If `use_random=True`: follow a **random policy** -> typically produces failures.

* `collect_truncated_trajectory(...)`:

  * Purely random policy.
  * If the agent **terminates early**, we discard that episode and resample.
  * If it survives for `TRUNCATED_STEPS` steps without termination, we label as `"truncated"`.

If the script fails to reach the target number of `success` trajectories, it automatically adjusts the target counts for `failure` and `truncated` to maintain the total `num_samples`.

---

## 5. Saving Images and JSON Metadata

### 5.1. Image saving

For each trajectory, we post-process and save:

* First, we **cut** the trajectory at the first **terminal tile** (`H` or `G`).
* Then we **remove “standing still” steps** (consecutive duplicate states).
* Optionally we **truncate** to at most `MAX_FRAMES` frames.

For each remaining state in the trajectory:

```python
env.reset()
env.unwrapped.s = state
env.unwrapped.lastaction = previous_action or None
img = render_frozen_lake_to_image(env)   # env.render() → RGB array → PIL.Image
img.save(f"{episode_dir}/{step_idx}.png")
```

Image path format:

```text
frozenlake/level<map_size>/<train|test>/<group_id>/<data_id>/<step>.png
```

* `group_id` – index of the map instance.
* `data_id` – index of the episode within that group.
* `step` – frame index along the episode (0-based).

### 5.2. JSON schema

For each split (`train` / `test`), we maintain a dictionary `json_data` and save it as:

* `frozenlake/level<map_size>/train/train_data.json`
* `frozenlake/level<map_size>/test/test_data.json`

The schema is:

```jsonc
{
  "0": {
    "actions":            [...],  // list of episodes, each a list of action IDs
    "states":             [...],  // list of episodes, each a list of state IDs (int)
    "rewards":            [...],  // list of episode-level rewards (float, 1.0 success else 0.0)
    "data_id":            [...],  // list of episode indices (int)
    "env_desc":           "<string>", // concatenated map layout, e.g., "SFFFHF..."
    "exec_states":        [...],  // list of strings: "success" | "failure"/"terminated" | "truncated"
    "additional_actions": [...]   // list of lists, currently always [] in main pipeline
  },
  "1": {
    ...
  }
}
```

Notes:

* Top-level keys `"0"`, `"1"`, … correspond to **group IDs** (maps).
* `env_desc` is the flattened map description derived from `env.unwrapped.desc`.
  This allows reconstructing the exact map used for each group.
* `rewards` are **episode-level rewards**:

  * `1.0` if `exec_state == "success"`,
  * `0.0` otherwise.

All numpy types are converted to native Python types via `convert_to_python_types(...)` before serialisation, ensuring standard JSON.

