
# PPO-Based Mini-Behavior Dataset Generation (Ours)

This repository extends the official **Mini-BEHAVIOR** framework
👉 [https://github.com/StanfordVL/mini_behavior](https://github.com/StanfordVL/mini_behavior)
and provides a script to generate **PPO-based rollouts** for the *InstallingAPrinter* task across multiple room sizes. The generator outputs:

* **rendered frames** (`frame_*.png`) for each PPO step
* a **single `data.json` per room size** that stores actions, outcome labels (`A/B/C/D`), rewards, map descriptors, and other metadata.

The script lives at:

```
mini_behavior/mini_behavior/generate_ppo_multilevel_dataset.py
```

---

## 1. Requirements

Install Mini-BEHAVIOR and its dependencies:

```bash
pip install gym-minigrid==1.0.3
pip install setuptools==66.0.0
pip install wheel==0.38.4
pip install gym==0.21.0

# Required for PPO rollouts
pip install stable-baselines3==1.6.2
pip install opencv-python

# Install this repo (extends Stanford Mini-Behavior)
pip install -e .
```

**Note:**
Our PPO-based dataset generator builds directly on the official Stanford Mini-BEHAVIOR implementation.
For full reproducibility, please use the exact versions listed above.

### Pre-trained PPO Models

The complete set of PPO checkpoints (for room sizes 7, 8, 9, and 10) exceeds **100 MB**.
However, **anonymous GitHub repositories strictly enforce a 100 MB storage limit**, which makes it impossible to upload all trained models.

To comply with this constraint, we include **only the smallest checkpoint**:

```
InstallingAPrinter_room7_final.zip
```

This file is sufficient for reviewers to **verify that the dataset-generation script runs correctly end-to-end**.

To reproduce room-7 results, please use the dedicated script:

```
mini_behavior/mini_behavior/generate_ppo_room7.py
```

This script contains identical logic to the full multi-level generator but is restricted to a single room size for verification within the anonymous GitHub size limits.

---

## 2. Script Overview

`generate_ppo_multilevel_dataset.py` defines a `DatasetGenerator` class and a CLI that iterates over:

```python
room_configs = [
    (7,  args.traj_max_steps_7),
    (8,  args.traj_max_steps_8),
    (9,  args.traj_max_steps_9),
    (10, args.traj_max_steps_10),
]
```

For each room size, it:

1. loads the PPO model `InstallingAPrinter_room{size}_final.zip`
2. registers the corresponding environment:

   ```python
   self.env_name = f"MiniGrid-{task}-{room_size}x{room_size}-N2-v0"
   ```
3. generates multiple **procedural maps**
4. runs **K rollouts per map** using the PPO agent
5. **renders** each frame
6. assigns the execution label `A/B/C/D`
7. writes a unified `data.json`

All internal behavior is unchanged from your script.

---

## 3. How to Run

From inside `mini_behavior/mini_behavior/`:

```bash
python generate_ppo_multilevel_dataset.py \
  --task InstallingAPrinter \
  --models_root ../models \
  --output_root datasetppo \
  --max_steps 1000 \
  --num_maps 50 \
  --rollouts_per_map 10 \
  --partial_obs True \
  --traj_max_steps_7 10 \
  --traj_max_steps_8 11 \
  --traj_max_steps_9 12 \
  --traj_max_steps_10 13
```

Output structure:

```
datasetppo/
  room7/
    map_0/rollout_0/frame_0.png ...
    map_1/...
    data.json
  room8/
  room9/
  room10/
```

To verify only room 7 (because only this model is uploaded):

```bash
python generate_ppo_room7.py \
  --models_root ../models \
  --output_root dataset_room7 \
  --num_maps 5 \
  --rollouts_per_map 10
```

---

## 4. What the Script Does

### Environment Creation

```python
env = gym.make(self.env_name)
env.seed(seed) or env.reset(seed=seed)

if not self.partial_obs:
    env = MiniBHFullyObsWrapper(env)

env = ImgObsWrapper(env)
```

### Rendering

Frames are obtained from:

1. MiniGrid’s `grid.render(...)`
2. `env.unwrapped.render(mode="rgb_array")`
3. fallback to a blank image

and saved as:

```
map_{id}/rollout_{id}/frame_{t}.png
```

using **OpenCV** (opencv-python).

### PPO Rollout Logic

* PPO predicts deterministic actions
* stops when success, done, or `traj_max_steps`
* logs:

  * actions, rewards
  * rendered frames
  * rollout success/failure

### Outcome Labels (A/B/C/D)

`exec_state` follows PPO outcome:

* **A**: success (positive reward)
* **B,C,D**: failure categories (using `analyze_failure_reason(...)`)

After generation:

```python
maps_data = self.rebalance_failure_labels(maps_data)
```

B/C/D failures are globally rebalanced; trajectories remain unchanged.

---

## 5. `data.json` Format

Each room directory contains:

```json
{
  "0": {
    "actions": [[...], [...], ...],
    "exec_states": ["A", "C", "B", ...],
    "rewards": [1.0, 0, 0, ...],
    "data_id": [0, 1, 2, ...],
    "env_desc": "....PT#...P...",
    "additional_actions": [[], [], ...],
    "num_rollouts": 10
  },
  "1": { ... }
}
```

`env_desc` is a compact textual grid descriptor over `{P, T, #, .}`.

---

## 6. Dataset Statistics

The script prints:

* A/B/C/D counts
* rollout length distribution
* checks that no rollout exceeds `traj_max_steps_X`
