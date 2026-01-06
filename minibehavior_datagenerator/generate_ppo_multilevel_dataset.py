import gym
from gym_minigrid.wrappers import ImgObsWrapper
from mini_behavior.utils.wrappers import MiniBHFullyObsWrapper
from mini_behavior.register import register
import mini_behavior  # noqa: F401 (needed for env registration side effects)

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import numpy as np
import torch
import torch.nn as nn
import argparse
import json
from collections import defaultdict
from pathlib import Path
import cv2


# ----------------- Feature Extractor (same as training) ----------------- #

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,
                 features_dim: int = 512,
                 normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = observation_space.sample()[None]
            n_flatten = self.cnn(torch.as_tensor(sample_obs).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# ----- Failure classification helpers (ported from your second script) ----- #

ACTION_NAMES = {
    0: "Left",
    1: "Right",
    2: "Forward",
    3: "Toggle",
    4: "Open",
    5: "Close",
    6: "Slice",
    7: "Cook",
    8: "Drop_in",
    9: "Pickup_0",
    10: "Pickup_1",
    11: "Pickup_2",
    12: "Drop_0",
    13: "Drop_1",
    14: "Drop_2",
}


# ---------- helpers to build env_desc, compatible with heuristic script ---------- #

def _is_printer_obj(o):
    t = getattr(o, "type", "")
    return "printer" in str(t).lower()


def _is_table_obj(o):
    if o is None:
        return False
    t = getattr(o, "type", "")
    tl = str(t).lower()
    if any(k in tl for k in ["table", "desk", "counter", "surface"]):
        return True
    # “can_contain” furniture (cabinets, etc.) – treat as table-like
    if getattr(o, "can_contain", None):
        return True
    return False


def _grid_objects_at(grid, x, y):
    """
    Return all non-None objects at (x,y) across dimensions, if available.
    Compatible with Mini-BEHAVIOR / MiniGrid grid implementations.
    """
    # layer-wise (Mini-Behavior)
    if hasattr(grid, "get_all_items"):
        items = grid.get_all_items(x, y)  # [f0,o0,f1,o1,f2,o2]
        return [o for o in items if o is not None]

    if hasattr(grid, "get_all_objs"):
        items = grid.get_all_objs(x, y)
        return [o for o in items if o is not None]

    # simple single object
    o = grid.get(x, y)
    return [o] if o is not None else []


def encode_env_desc(base_env):
    """
    Encode the grid into a short string, similar to the heuristic generator:

    Flatten row by row (y-major), for each cell (x,y):
      'P' = printer present
      'T' = table present
      '#' = wall
      '.' = empty / other
    """
    grid = getattr(base_env, "grid", None)
    if grid is None:
        return ""

    W, H = grid.width, grid.height
    chars = []
    for y in range(H):
        for x in range(W):
            objs_here = _grid_objects_at(grid, x, y)
            if any(_is_printer_obj(o) for o in objs_here):
                chars.append("P")
            elif any(_is_table_obj(o) for o in objs_here):
                chars.append("T")
            else:
                cell = grid.get(x, y)
                if cell is not None and "wall" in str(getattr(cell, "type", "")).lower():
                    chars.append("#")
                else:
                    chars.append(".")
    return "".join(chars)



def analyze_failure_reason(info, action, action_name, total_reward, step_count, max_steps):
    """
    Decide B/C/D using the same heuristic logic as your visualization script.

    Returns one of {"B", "C", "D"}.
    """
    # 1) If env explicitly gives a failure_reason code, trust it
    if isinstance(info, dict) and "failure_reason" in info:
        r = info["failure_reason"]
        if r in ("B", "C", "D"):
            return r

    info = info or {}

    # 2) Info flags → D (missing key object / invalid action)
    if ("missing_object" in info
        or "invalid_action" in info
        or "missing_key_object" in info):
        return "D"

    # 3) Placement-related errors → B
    if ("invalid_placement" in info
        or "wrong_location" in info
        or "invalid_drop" in info):
        return "B"

    # 4) Pickup-related errors → C
    if ("invalid_pickup" in info
        or "no_object_to_pick" in info
        or "pickup_failed" in info):
        return "C"

    # 5) Heuristic from last action type
    if action_name.startswith("Drop") or action_name == "Drop_in":
        return "B"   # put printer in wrong place

    if action_name.startswith("Pickup"):
        return "C"   # pickup failed or never got printer properly

    # 6) Timeout without success
    if step_count >= max_steps and total_reward < 1.0:
        # interpret as "missing or unreachable key object"
        return "D"

    # 7) Default fallback: B
    return "B"


# ----------------- Dataset Generator ----------------- #

class DatasetGenerator:
    def __init__(
        self,
        model_path,
        task="InstallingAPrinter",
        room_size=10,
        max_steps=1000,          # env max_steps
        num_maps=10,
        rollouts_per_map=10,
        partial_obs=True,
        traj_max_steps=None,     # cap on stored trajectory length
    ):
        self.task = task
        self.room_size = room_size
        self.env_max_steps = max_steps
        self.num_maps = num_maps
        self.rollouts_per_map = rollouts_per_map
        self.model_path = model_path
        self.partial_obs = partial_obs
        self.traj_max_steps = (
            traj_max_steps if traj_max_steps is not None else min(200, max_steps)
        )

        # Register environment (MiniBehavior)
        self.env_name = f"MiniGrid-{task}-{room_size}x{room_size}-N2-v0"
        kwargs = {"room_size": room_size, "max_steps": self.env_max_steps}
        
        try:
            register(
                id=self.env_name,
                entry_point=f"mini_behavior.envs:{task}Env",
                kwargs=kwargs,
            )
        except AssertionError:
            # Env already registered in this process; safe to ignore
            print(f"[info] Env '{self.env_name}' already registered, skipping re-registration.")

        # Load PPO model
        print(f"Loading model: {model_path}")
        try:
            self.model = PPO.load(model_path)
            self.model_obs_shape = tuple(self.model.observation_space.shape)
            print("✓ Successfully loaded model using original configuration")
        except Exception as e:
            print(f"✗ Failed to load with original configuration: {e}")
            print("Trying to load using custom feature extractor...")
            policy_kwargs = dict(
                features_extractor_class=MinigridFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            self.model = PPO.load(
                model_path, custom_objects={"policy_kwargs": policy_kwargs}
            )
            self.model_obs_shape = tuple(self.model.observation_space.shape)
            print("✓ Successfully loaded model using custom feature extractor")

        self.rng = np.random.RandomState()

        # Optional: inspect actions enum for sanity
        tmp_env = self.create_env()
        base_env = tmp_env.unwrapped
        self.actions_enum = getattr(base_env, "actions", None)
        if self.actions_enum is not None:
            print("=== actions_enum mapping ===")
            for name, value in self.actions_enum.__dict__.items():
                if not name.startswith("_") and isinstance(value, int):
                    print(f"  {name:<15} -> {value}")
            print("================================")
        tmp_env.close()

    def create_env(self):
        """Create environment with same wrappers as training."""
        env = gym.make(self.env_name)

        # Seed for diversity
        seed = self.rng.randint(0, 2**31 - 1)
        try:
            env.seed(seed)
        except Exception:
            try:
                env.reset(seed=seed)
            except Exception:
                pass

        if not self.partial_obs:
            env = MiniBHFullyObsWrapper(env)

        env = ImgObsWrapper(env)  # returns HWC; we will transpose for PPO
        return env

    def get_render_image(self, env):
        """
        Get a rendered RGB image of the *full* environment, without
        the partial-observation window / FOV overlay.
        """
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env

        # 1) Preferred path: render directly from the grid, no overlays
        grid = getattr(base_env, "grid", None)
        if grid is not None and hasattr(grid, "render"):
            try:
                img = grid.render(
                    tile_size=32,
                    agent_pos=getattr(base_env, "agent_pos", None),
                    agent_dir=getattr(base_env, "agent_dir", None),
                    highlight_mask=None,   # <- no FOV highlighting
                )
                # grid.render already returns H x W x 3 uint8 RGB
                return img
            except Exception as e:
                print(f"[get_render_image] grid.render failed, fallback to env.render: {e}")

        # 2) Fallback: tell the env not to draw highlight / extra windows
        try:
            # Some MiniGrid variants accept highlight=...
            img = base_env.render(
                mode="rgb_array",
                highlight=False,   # <- avoid FOV highlight
            )
            return img
        except TypeError:
            # If highlight is not a valid kwarg in this env
            try:
                img = base_env.render(mode="rgb_array")
                return img
            except Exception as e:
                print(f"[get_render_image] base_env.render failed: {e}")

        # 3) Final fallback: blank canvas with the right geometry
        h = getattr(base_env, "height", 10)
        w = getattr(base_env, "width", 10)
        return np.zeros((h * 32, w * 32, 3), dtype=np.uint8)

    def analyze_failure(self, base_env, trajectory):
        """
        (Kept for reference; not used anymore for exec_state.)
        Classify failure into B/C/D via carrying_printer heuristic.
        """
        has_printer = False
        has_table = False

        # Prefer base_env.objs (how Mini-BEHAVIOR tracks objects)
        objs_dict = getattr(base_env, "objs", {})
        if isinstance(objs_dict, dict):
            for key, objs in objs_dict.items():
                for obj in objs:
                    obj_type = getattr(obj, "type", str(type(obj)))
                    lt = obj_type.lower()
                    if "printer" in lt:
                        has_printer = True
                    if "table" in lt:
                        has_table = True

        # Fallback: also scan grid in case objs_dict is missing
        if not has_printer or not has_table:
            grid = getattr(base_env, "grid", None)
            if grid is not None:
                for x in range(grid.width):
                    for y in range(grid.height):
                        obj = grid.get(x, y)
                        if obj is None:
                            continue
                        obj_type = getattr(obj, "type", str(type(obj)))
                        lt = obj_type.lower()
                        if "printer" in lt:
                            has_printer = True
                        if "table" in lt:
                            has_table = True

        if not has_printer or not has_table:
            return "D"

        ever_carried_printer = any(
            t.get("carrying_printer", False)
            or t.get("carrying_printer_after", False)
            for t in trajectory
        )

        if not ever_carried_printer:
            return "C"

        return "B"

    def rebalance_failure_labels(self, maps_data):
        """
        Post-process exec_states so that among failures (B,C,D),
        the counts of B, C, D are as balanced as possible globally.

        NOTE: this changes labels but does NOT change the underlying trajectories.
        """
        # collect all failure indices
        failure_indices = []  # list of (map_key, rollout_idx)
        for map_key, m in maps_data.items():
            for i, s in enumerate(m["exec_states"]):
                if s in ("B", "C", "D"):
                    failure_indices.append((map_key, i))

        F = len(failure_indices)
        if F < 3:
            # nothing useful to rebalance
            return maps_data

        # target counts per label
        base = F // 3
        remainder = F % 3
        labels = ["B", "C", "D"]
        target = {lab: base for lab in labels}
        for i in range(remainder):
            target[labels[i]] += 1

        # shuffle failure episodes
        self.rng.shuffle(failure_indices)

        # assign new labels to match the target counts
        idx = 0
        for lab in labels:
            for _ in range(target[lab]):
                map_key, rollout_idx = failure_indices[idx]
                maps_data[map_key]["exec_states"][rollout_idx] = lab
                idx += 1

        return maps_data

    def _to_policy_obs(self, obs):
        """
        Convert env obs to what the PPO policy expects.

        - If model expects HWC and obs is HWC -> return as is.
        - If model expects CHW and obs is HWC -> transpose.
        - If shapes already match -> return as is.
        """
        obs = np.asarray(obs)
        if obs.ndim != 3:
            return obs  # e.g. vector obs

        h, w, c_or_ch = obs.shape
        s = self.model_obs_shape

        # If shapes already match, just return
        if s == obs.shape:
            return obs

        if len(s) == 3:
            # Model expects CHW
            if s[0] in (1, 3, 32) and s[1] == h and s[2] == w:
                # Our env gives HWC -> convert to CHW
                return np.transpose(obs, (2, 0, 1))
            # Model expects HWC
            if s[2] in (1, 3, 32) and s[0] == h and s[1] == w:
                # Our env already HWC -> no change
                return obs

        # Fallback: do not change
        return obs

    def rollout_with_ppo(self):
        env = self.create_env()
        obs = env.reset()
        obs_policy = self._to_policy_obs(obs)

        base_env = env.unwrapped

        actions = []
        trajectory = []
        carrying_printer = False
        done = False
        step_idx = 0
        success = False

        total_reward = 0.0
        last_info = {}
        last_action = None

        while not done and step_idx < self.traj_max_steps:
            img = self.get_render_image(env)
            state_info = {
                "step": int(step_idx),
                "action": None,
                "agent_pos": [int(pos) for pos in base_env.agent_pos],
                "agent_dir": int(base_env.agent_dir),
                "carrying_printer": carrying_printer,
                "carrying_printer_after": carrying_printer,
                "image": img,
            }

            # action from PPO
            action, _ = self.model.predict(obs_policy, deterministic=True)
            action = int(action)
            actions.append(action)
            state_info["action"] = action

            obs, reward, done_env, info = env.step(action)
            obs_policy = self._to_policy_obs(obs)

            total_reward += float(reward)
            last_info = info
            last_action = action

            if reward > 0:
                success = True
                done = True
            else:
                done = bool(done_env)

            # carrying_printer flag (still useful for debugging / future)
            if hasattr(base_env, "carrying") and base_env.carrying is not None:
                obj_type = (
                    base_env.carrying.type
                    if hasattr(base_env.carrying, "type")
                    else str(type(base_env.carrying))
                )
                carrying_printer = "printer" in obj_type.lower()
            else:
                carrying_printer = False

            state_info["carrying_printer_after"] = carrying_printer
            trajectory.append(state_info)

            step_idx += 1

        # final frame
        img_final = self.get_render_image(env)
        final_state = {
            "step": int(step_idx),
            "action": -1,
            "agent_pos": [int(pos) for pos in base_env.agent_pos],
            "agent_dir": int(base_env.agent_dir),
            "carrying_printer": carrying_printer,
            "carrying_printer_after": carrying_printer,
            "image": img_final,
        }
        trajectory.append(final_state)

        # ----- New classification: A vs B/C/D with heuristic ----- #
        if success:
            exec_state = "A"
        else:
            if last_action is None:
                # extremely degenerate, but just call it D
                exec_state = "D"
            else:
                action_name = ACTION_NAMES.get(int(last_action), f"Unknown_{last_action}")
                exec_state = analyze_failure_reason(
                    last_info or {},
                    int(last_action),
                    action_name,
                    total_reward,
                    step_idx,
                    self.env_max_steps,
                )

        env.close()
        return actions, exec_state, trajectory, total_reward

    def generate_dataset(self, output_dir="dataset_ppo_maps"):
        """
        Generate dataset in BFS-style structure:

        output_dir/
          map_0/
            rollout_0/frame_0.png, frame_1.png, ...
            rollout_1/...
          map_1/...
          data.json  # { "0": { "actions": [...], "exec_states": [...], ... }, ... }
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        maps_data = {}

        print(
            f"Start generating dataset: {self.num_maps} maps, "
            f"{self.rollouts_per_map} rollouts per map "
            f"({self.num_maps * self.rollouts_per_map} rollouts total)"
        )
        print(
            f"Using {'partial' if self.partial_obs else 'full'} observation, "
            f"traj_max_steps = {self.traj_max_steps}"
        )

        total_rollouts = self.num_maps * self.rollouts_per_map
        rollout_counter = 0

        for map_id in range(self.num_maps):
            print(f"\n=== Generating map {map_id}/{self.num_maps - 1} ===")
            map_dir = output_path / f"map_{map_id}"
            map_dir.mkdir(exist_ok=True)

            # --- compute env_desc once per map from a fresh env --- #
            env0 = self.create_env()
            try:
                env0.reset()
            except TypeError:
                # older gym versions
                _ = env0.reset()
            base_env0 = env0.unwrapped if hasattr(env0, "unwrapped") else env0
            try:
                env_desc = encode_env_desc(base_env0)
            except Exception as e:
                print(f"[warn] encode_env_desc failed for map {map_id}: {e}")
                env_desc = ""
            env0.close()

            map_actions = []
            map_exec_states = []
            map_rewards = []               
            map_additional_actions = []    

            for rollout_id in range(self.rollouts_per_map):
                rollout_counter += 1
                print(
                    f"\r  rollout {rollout_id + 1}/{self.rollouts_per_map} "
                    f"(global {rollout_counter}/{total_rollouts})",
                    end="",
                )

                actions, exec_state, trajectory, total_reward = self.rollout_with_ppo()

                # Save frames for this rollout
                rollout_dir = map_dir / f"rollout_{rollout_id}"
                rollout_dir.mkdir(exist_ok=True)

                for step_idx, state in enumerate(trajectory):
                    if state.get("action", None) == -1:
                        continue
                    img = state["image"]
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)

                    img_path = rollout_dir / f"frame_{step_idx}.png"
                    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                map_actions.append([int(a) for a in actions])
                map_exec_states.append(exec_state)
                map_rewards.append(float(total_reward))   
                map_additional_actions.append([])         

            # per-map data structure (keep num_rollouts, add new fields)
            maps_data[str(map_id)] = {
                "actions": map_actions,
                "exec_states": map_exec_states,
                "rewards": map_rewards,                          
                "data_id": list(range(self.rollouts_per_map)),   
                "env_desc": env_desc,                            
                "additional_actions": map_additional_actions,     
                "num_rollouts": int(self.rollouts_per_map),
            }

        # Balance B/C/D failures (same logic)
        maps_data = self.rebalance_failure_labels(maps_data)

        # Save global data.json
        data_json = convert_numpy_types(maps_data)
        with open(output_path / "data.json", "w") as f:
            json.dump(data_json, f, indent=2)

        self.print_statistics(maps_data)
        self.print_length_statistics(maps_data)

        print(f"\nDataset generation finished! Saved at: {output_path}")
        return maps_data


    def print_statistics(self, maps_data):
        counts = defaultdict(int)
        total_rollouts = 0
        for m in maps_data.values():
            for s in m["exec_states"]:
                counts[s] += 1
                total_rollouts += 1

        print("\n" + "=" * 50)
        print("Dataset statistics (over all rollouts)")
        print("=" * 50)
        print(f"Total rollouts: {total_rollouts}")
        reason_map = {
            "A": "Success",
            "B": "Wrong placement",
            "C": "Never carried printer",
            "D": "Missing key objects",
        }
        for k, v in counts.items():
            print(
                f"  {reason_map.get(k, k)} ({k}): {v} "
                f"({v / max(1, total_rollouts) * 100:.1f}%)"
            )
        print("=" * 50)

    def print_length_statistics(self, maps_data):
        """
        Inspect trajectory lengths (number of actions per rollout) and
        verify they respect traj_max_steps. Also report mean/min/max/std.
        """
        lengths = []

        for m in maps_data.values():
            for action_seq in m["actions"]:
                lengths.append(len(action_seq))

        if not lengths:
            print("\n[Length stats] No rollouts found.")
            return

        lengths = np.asarray(lengths, dtype=np.int32)
        mean_len = float(lengths.mean())
        std_len = float(lengths.std())
        min_len = int(lengths.min())
        max_len = int(lengths.max())

        print("\n" + "=" * 50)
        print("Trajectory length statistics")
        print("=" * 50)
        print(f"Designed traj_max_steps: {self.traj_max_steps}")
        print(f"Number of rollouts:     {len(lengths)}")
        print(f"Min length:             {min_len}")
        print(f"Max length:             {max_len}")
        print(f"Mean length:            {mean_len:.2f}")
        print(f"Std length:             {std_len:.2f}")

        # Hard sanity check: no rollout should exceed traj_max_steps
        if max_len > self.traj_max_steps:
            print("⚠ WARNING: some trajectories exceed traj_max_steps!")
        else:
            print("✓ All trajectories respect traj_max_steps upper bound.")
        print("=" * 50)


# ----------------- CLI ----------------- #

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Shared options
    parser.add_argument(
        "--task",
        default="InstallingAPrinter",
        help="Task name",
    )
    parser.add_argument(
        "--models_root",
        default="../models",
        help="Directory containing InstallingAPrinter_room{7,8,9,10}_final.zip",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Environment max_steps parameter (same for all rooms)",
    )
    parser.add_argument(
        "--num_maps",
        type=int,
        default=100,
        help="Number of different maps (groups) to generate per room",
    )
    parser.add_argument(
        "--rollouts_per_map",
        type=int,
        default=10,
        help="Number of PPO rollouts per map",
    )
    parser.add_argument(
        "--output_root",
        default="datasetppo",
        help="Base output directory; we create room7/, room8/, room9/, room10/ under this",
    )
    parser.add_argument(
        "--partial_obs",
        type=str2bool,
        default=True,
        help="Whether to use partial observation (True/False)",
    )

    # Per-room maximum trajectory lengths (configurable)
    parser.add_argument(
        "--traj_max_steps_7",
        type=int,
        default=10,
        help="Maximum trajectory length for room_size=7",
    )
    parser.add_argument(
        "--traj_max_steps_8",
        type=int,
        default=11,
        help="Maximum trajectory length for room_size=8",
    )
    parser.add_argument(
        "--traj_max_steps_9",
        type=int,
        default=12,
        help="Maximum trajectory length for room_size=9",
    )
    parser.add_argument(
        "--traj_max_steps_10",
        type=int,
        default=13,
        help="Maximum trajectory length for room_size=10",
    )

    args = parser.parse_args()

    room_configs = [
        (7, args.traj_max_steps_7),
        (8, args.traj_max_steps_8),
        (9, args.traj_max_steps_9),
        (10, args.traj_max_steps_10),
    ]

    for room_size, traj_max in room_configs:
        model_path = Path(args.models_root) / f"InstallingAPrinter_room{room_size}_final.zip"
        output_dir = Path(args.output_root) / f"room{room_size}"

        print("\n" + "#" * 70)
        print(f"Generating dataset for room_size={room_size}")
        print(f"Model:      {model_path}")
        print(f"Output dir: {output_dir}")
        print(f"traj_max_steps: {traj_max}")
        print("#" * 70 + "\n")

        generator = DatasetGenerator(
            model_path=str(model_path),
            task=args.task,
            room_size=room_size,
            max_steps=args.max_steps,
            num_maps=args.num_maps,
            rollouts_per_map=args.rollouts_per_map,
            partial_obs=args.partial_obs,
            traj_max_steps=traj_max,
        )
        generator.generate_dataset(output_dir=str(output_dir))

