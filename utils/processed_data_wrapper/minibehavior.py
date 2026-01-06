# coding=utf-8
# MiniBehavior (e.g., InstallingAPrinter) dataset for MVoT (toy version)

import json
import os
import math

import datasets
from PIL import Image

_CITATION = """
@misc{mini_behavior,
    title        = {Mini-behavior: A procedurally generated benchmark for long-horizon decision-making in embodied ai},
    author       = {Jin, Emily and Hu, Jiaheng and Huang, Zhuoyi and Zhang, Ruohan and Wu, Jiajun and Fei-Fei, Li and Mart{\'\i}n-Mart{\'\i}n, Roberto},
    journal={arXiv preprint arXiv:2310.01824},
    year         = {2023},
}
"""

_DESCRIPTION = """
MiniBehavior dataset (e.g., InstallingAPrinter task) formatted for multimodal
trajectory modeling. Each example contains a sequence of grid states, high-level
actions, and rendered frames, along with a trajectory-level outcome label in
{A, B, C, D}.
"""


_LICENSE = "CC BY 4.0"

_DATA_DIR = r"minibehavior"

_URLS = {
    "data_dir": _DATA_DIR,
}

SINGLE_STEP_VISUALIZATION_INSTRUCTION = {
    "minibehavior_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
LONG_HORIZON_VISUALIZATION_INSTRUCTION = {
    "minibehavior_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
REAL_GOAL_INSTRUCTION = {
    "minibehavior_simulation": (
        "Task: MiniBehavior (e.g., InstallingAPrinter).\n"
        "Given the action sequence and visual observations, determine the outcome "
        "category of this trajectory. Return one of A, B, C, or D, corresponding to "
        "the outcome annotation in the dataset.\n"
        "Full Action Sequence: <ACTION_SEQ>\n"
    )
}

# Action ID -> symbolic name.
# Based on your earlier actions_enum mapping:
#   0:left  1:right  2:forward  3:toggle  4:open  5:close
#   6:slice 7:cook  8:drop_in  9:pickup_0 10:pickup_1 11:pickup_2
#   12:drop_0 13:drop_1 14:drop_2
ACTION_DICT = {
    0: "left",
    1: "right",
    2: "forward",
    3: "toggle",
    4: "open",
    5: "close",
    6: "slice",
    7: "cook",
    8: "drop_in",
    9: "pickup_0",
    10: "pickup_1",
    11: "pickup_2",
    12: "drop_0",
    13: "drop_1",
    14: "drop_2",
}


class MiniBehaviorConfig(datasets.BuilderConfig):
    """BuilderConfig for MiniBehavior."""

    def __init__(self, tasks, modes, data_dir, **kwargs):
        """BuilderConfig for MiniBehavior.

        Args:
          tasks: list of task types (e.g., ["simulation"])
          modes: list of training modes
          data_dir: base directory for data_samples
          **kwargs: forwarded to super
        """
        super(MiniBehaviorConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir


class MiniBehavior(datasets.GeneratorBasedBuilder):
    """MiniBehavior dataset."""

    BUILDER_CONFIG_CLASS = MiniBehaviorConfig
    BUILDER_CONFIGS = [
        MiniBehaviorConfig(
            name="processed_minibehavior",
            version=datasets.Version("0.0.0"),
            description=_DESCRIPTION,
            tasks=["simulation"],
            modes=["single_step_visualization", "action_reasoning"],
            data_dir="data_samples",
        )
    ]

    DEFAULT_CONFIG_NAME = "processed_minibehavior"

    def _info(self):
        features = datasets.Features(
            {
                "idx": datasets.Value("int32"),
                "input_text": datasets.Value("string"),
                "input_imgs": datasets.Sequence(datasets.Image()),
                "label_text": datasets.Value("string"),
                "label_imgs": datasets.Sequence(datasets.Image()),
                "label_img_paths": datasets.Sequence(datasets.Value("string")),
                "input_img_paths": datasets.Sequence(datasets.Value("string")),
                "task": datasets.Value("string"),
                "train_task": datasets.Value("string"),
                # sequence of positions along the trajectory;
                # here we store (row, col) pairs recovered from states
                "coords": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"))
                ),
                # grid / room size (e.g., 7 for 7x7)
                "maze_size": datasets.Value("int32"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = _URLS

        tasks = self.config.tasks
        modes = self.config.modes
        data_dir = self.config.data_dir

        # Make the base prefix globally visible inside helper functions
        global _DATA_DIR_PREFIX
        _DATA_DIR_PREFIX = data_dir

        data_dirs = []
        # data_dir (e.g., "data_samples") + "minibehavior"
        data_dirs.append(os.path.join(data_dir, downloaded_files["data_dir"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "train",
                    "modes": modes,
                    "tasks": tasks,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("dev"),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "dev",
                    "modes": modes,
                    "tasks": tasks,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("test"),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "test",
                    "modes": modes,
                    "tasks": tasks,
                },
            ),
        ]

    def _generate_examples(self, data_dirs, split, modes, tasks):
        all_data = []

        for data_dir in data_dirs:
            # We assume JSON files are stored in subdirectories, similar to FrozenLake.
            # If your files are flat (directly under data_dir), you can simplify this glob.
            subdirs = [
                os.path.join(data_dir, d)
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]

            json_files = []
            for sub_dir in subdirs:
                for fname in os.listdir(sub_dir):
                    if fname.endswith(".json") or fname.endswith("data.json"):
                        json_files.append(os.path.join(sub_dir, fname))

            train_data = []
            dev_data = []

            for json_path in json_files:
                with open(json_path, "r") as f:
                    data = json.load(f)

                # data is a dict: {env_id_str: env_dict}
                for env_key, env_dict in data.items():
                    try:
                        env_id = int(env_key)
                    except ValueError:
                        # If env_key is not an int, you can still keep it as string,
                        # but we stick to int for consistency.
                        env_id = env_key

                    flattened = flatten_env_data(env_dict, env_id=env_id)
                    # Toy setup: use same envs for train/dev/test.
                    train_data += flattened
                    dev_data += flattened

            if split == "train":
                all_data += [
                    {**item, "task": "minibehavior_simulation"} for item in train_data
                ]
            else:
                all_data += [
                    {**item, "task": "minibehavior_simulation"} for item in dev_data
                ]

        data_idx = 0
        for data_item in all_data:
            interleaved_data_list = get_interleaved_data(
                data_item,
                mode=modes,
            )

            for item in interleaved_data_list:
                action_list = data_item["action_list"]
                additional_action_list = data_item["additional_actions"]

                full_action_text = "".join(
                    [f"{describe_action_token(a)} " for a in action_list + additional_action_list]
                )

                return_info = {
                    "idx": data_idx,
                    "input_text": item["input_text"].replace(
                        "<ACTION_SEQ>", full_action_text
                    ),
                    "input_imgs": item["input_imgs"],
                    "label_text": item["label_text"],
                    "label_imgs": item["label_imgs"],
                    "label_img_paths": item["label_img_paths"],
                    "input_img_paths": item["input_img_paths"],
                    "task": item["task"],
                    "train_task": item["train_task"],
                    "coords": item["coords"],
                    "maze_size": data_item["grid_num"],
                }

                yield data_idx, return_info
                data_idx += 1




def state_to_coordinates(state_idx, grid_num):
    """
    Convert a raw state index into (row, col) on a grid_num x grid_num grid.

    Many MiniBehavior state encodings include additional information (e.g.,
    inventory, object states). We recover the cell index by taking modulo
    grid_num**2, which preserves the agent's location if the encoding is
    location_index + k * (grid_num**2).
    """
    base = state_idx % (grid_num * grid_num)
    row = base // grid_num
    col = base % grid_num
    return [row, col]


def flatten_env_data(env_dict, env_id):
    """
    Flatten one environment/map dict into a list of trajectory-level dicts.

    NEW env_dict structure (from the PPO generator):
        {
          "actions":            [[...], [...], ...],
          "exec_states":        [...],      # "A"/"B"/"C"/"D"
          "rewards":            [...],      # total reward per rollout
          "data_id":            [...],      # rollout id (used in dir name)
          "env_desc":           "...",      # W*H grid encoding
          "additional_actions": [[...], ...],
          "num_rollouts":       int
        }

    We no longer have per-step "states", so we cannot recover true coordinates.
    We instead create dummy coords with the right length so that the rest of
    the MVoT pipeline (which expects coords) keeps working.
    """
    env_desc = env_dict["env_desc"]
    # env_desc is a flattened W*H string, so grid_num = sqrt(len(env_desc))
    grid_num = int(math.sqrt(len(env_desc)))

    actions_all = env_dict["actions"]                    # List[List[int]]
    exec_states_all = env_dict["exec_states"]            # List[str]
    rewards_all = env_dict.get("rewards", None)          # List[float] or None
    data_ids = env_dict.get("data_id",
                            list(range(len(actions_all))))
    additional_all = env_dict.get(
        "additional_actions",
        [[] for _ in actions_all],
    )

    num_trajs = len(actions_all)
    flattened = []

    for d_i in range(num_trajs):
        actions_i = actions_all[d_i]
        if len(actions_i) == 0:
            # Skip empty trajectories
            continue

        # Primitive actions -> textual tokens "left-1", "pickup_0-1", ...
        action_list = [f"{ACTION_DICT[v]}-1" for v in actions_i]

        # We don't have true states, so we just create dummy coords
        # with length = len(actions) + 1 (initial + steps),
        # matching the length used later in get_interleaved_data.
        path_locs = [[0, 0] for _ in range(len(actions_i) + 1)]

        additional_actions_i = additional_all[d_i]
        additional_action_list = [
            f"{ACTION_DICT[v]}-1" for v in additional_actions_i
        ]

        reward = rewards_all[d_i] if rewards_all is not None else 0.0

        traj_dict = {
            "action_list": action_list,
            "path_locs": path_locs,
            "labels": reward,                    # numeric reward, if needed
            "data_id": data_ids[d_i],           # rollout id
            "env_desc": env_desc,
            "env_id": env_id,                   # this is map_id
            "grid_num": grid_num,               # room size (7,8,9,10)
            "exec_state": exec_states_all[d_i], # "A"/"B"/"C"/"D"
            "additional_actions": additional_action_list,
        }
        flattened.append(traj_dict)

    return flattened

def get_interleaved_data(
    data_item,
    mode=("single_step_visualization", "action_reasoning"),
):
    """
    Build interleaved image/text examples from a single trajectory dict.

    data_item has:
        - action_list:        ["left-1", "forward-1", ...]
        - path_locs:          dummy coords, len = len(actions)+1
        - grid_num:           room size (7, 8, 9, 10)
        - env_id:             map index (0,1,...)
        - data_id:            rollout index within map
        - exec_state:         "A"/"B"/"C"/"D"
        - additional_actions: ["pickup_0-1", ...]
        - task:               "minibehavior_simulation"
    """
    interleaved_data = []

    action_list = data_item["action_list"]
    pos_list = data_item["path_locs"]  # length should be len(actions)+1
    task = data_item["task"]

    # ----------------------------------------------------------------------
    # Image paths: match PPO generator structure
    # data_samples/
    #   minibehavior/
    #     room{grid_num}/
    #       map_{env_id}/
    #         rollout_{data_id}/
    #           frame_0.png, frame_1.png, ...
    # ----------------------------------------------------------------------
    img_dir = os.path.join(
        "minibehavior",
        f"room{data_item['grid_num']}",
        f"map_{data_item['env_id']}",
        f"rollout_{data_item['data_id']}",
    )

    num_actions = len(action_list)

    # We only have frames 0..(num_actions-1) saved.
    # To preserve the old assumption that there is a "final" frame at index L,
    # we reuse the last frame as the terminal one.
    all_images = [
        os.path.join(img_dir, f"frame_{pos_idx}.png")
        for pos_idx in range(num_actions)
    ]
    if num_actions > 0:
        all_images.append(
            os.path.join(img_dir, f"frame_{num_actions - 1}.png")
        )

    # "input_img" token + all actions (same semantics as before).
    all_actions = ["input_img"] + action_list

    # For coords, we already created len(actions)+1 dummy positions in flatten_env_data
    all_pos = pos_list

    if not task.endswith("simulation"):
        raise ValueError("Task not found. Should be 'minibehavior_simulation'.")

    # Prepend base data directory (e.g., data_samples/)
    all_images = [os.path.join(_DATA_DIR_PREFIX, p) for p in all_images]

    try:
        all_pil_images = [
            Image.open(input_img_path).convert("RGB").resize((256, 256))
            for input_img_path in all_images
        ]
    except Exception:
        # If any image is missing or unreadable, skip this trajectory.
        return interleaved_data

    # ------------- single_step_visualization -------------
    if "single_step_visualization" in mode:
        # Batches of images, actions, and coords over time
        image_batches = [all_images[:2]] + [
            all_images[:1] + all_images[i : i + 2]
            for i in range(1, len(all_images) - 1)
        ]
        pil_image_batches = [all_pil_images[:2]] + [
            all_pil_images[:1] + all_pil_images[i : i + 2]
            for i in range(1, len(all_pil_images) - 1)
        ]
        action_batches = [all_actions[:2]] + [
            all_actions[0 : i + 2] for i in range(1, len(all_actions) - 1)
        ]
        pos_batches = [all_pos[:2]] + [
            all_pos[0 : i + 2] for i in range(1, len(all_pos) - 1)
        ]

        for image_batch, pil_image_batch, action_batch, pos_batch in zip(
            image_batches, pil_image_batches, action_batches, pos_batches
        ):
            # Use first + second-last frame as inputs (same as Maze/FrozenLake).
            if len(image_batch) > 2:
                input_image_paths = [image_batch[0], image_batch[-2]]
                input_imgs = pil_image_batch[:1] + pil_image_batch[-2:-1]
            else:
                input_image_paths = [image_batch[0]]
                input_imgs = pil_image_batch[:1]

            label_image_paths = image_batch[-1:]
            label_imgs = pil_image_batch[-1:]

            texts = get_text_from_actions(action_batch)
            input_texts = texts
            output_texts = ["<image>"]

            if len(input_texts) != 2:
                history_text = "".join(input_texts[1:-1]) + "<image>" + input_texts[-1]
            else:
                history_text = input_texts[-1]

            init_state_text = input_texts[0] + "<image>"

            input_text = SINGLE_STEP_VISUALIZATION_INSTRUCTION[task].replace(
                "<INIT_STATE>", init_state_text
            ).replace("<ACTION_HISTORY>", history_text)
            input_text = REAL_GOAL_INSTRUCTION[task] + input_text

            return_info = {
                "task": task,
                "input_text": input_text,
                "label_text": "<image>",
                "input_imgs": input_imgs,
                "input_img_paths": input_image_paths,
                "label_imgs": label_imgs,
                "label_img_paths": label_image_paths,
                "train_task": "single_step_visualization",
                "coords": pos_batch,
            }
            interleaved_data.append(return_info)

    # ------------- action_reasoning -------------
    if "action_reasoning" in mode:
        image_batches = [all_images[:1]] + [
            all_images[:1] + all_images[i : i + 1]
            for i in range(1, len(all_images))
        ]
        pil_image_batches = [all_pil_images[:1]] + [
            all_pil_images[:1] + all_pil_images[i : i + 1]
            for i in range(1, len(all_pil_images))
        ]
        action_batches = [all_actions[0:2]] + [
            all_actions[0 : i + 2] for i in range(1, len(all_actions))
        ]
        pos_batches = [all_pos[0:2]] + [
            all_pos[0 : i + 2] for i in range(1, len(all_pos))
        ]

        last_batch_idx = len(image_batches) - 1

        for batch_idx, (image_batch, pil_image_batch, action_batch, pos_batch) in enumerate(
            zip(image_batches, pil_image_batches, action_batches, pos_batches)
        ):
            if len(image_batch) != 1:
                input_image_paths = [image_batch[0], image_batch[-1]]
                input_imgs = pil_image_batch[:1] + pil_image_batch[-1:]
            else:
                input_image_paths = [image_batch[0]]
                input_imgs = pil_image_batch[:1]

            label_image_paths = []
            label_imgs = []

            texts = get_text_from_actions(action_batch)
            if batch_idx != last_batch_idx:
                input_texts = texts[:-1]
                output_texts = texts[-1:]
            else:
                input_texts = texts
                output_texts = [
                    f"Action sequence finished. The answer is {get_answer(data_item)}."
                ]

            history_text = "".join(input_texts[1:]) + "<image>" if batch_idx != 0 else ""
            init_state_text = input_texts[0] + "<image>"

            input_text = LONG_HORIZON_VISUALIZATION_INSTRUCTION[task].replace(
                "<INIT_STATE>", init_state_text
            ).replace("<ACTION_HISTORY>", history_text)
            input_text = REAL_GOAL_INSTRUCTION[task] + input_text

            return_info = {
                "task": task,
                "input_text": input_text,
                "label_text": " ".join(o_t.strip() for o_t in output_texts),
                "input_imgs": input_imgs,
                "input_img_paths": input_image_paths,
                "label_imgs": label_imgs,
                "label_img_paths": label_image_paths,
                "train_task": "action_reasoning",
                "coords": pos_batch,
            }
            interleaved_data.append(return_info)

    return interleaved_data

def get_text_from_actions(action_list):
    """
    Convert action tokens like 'left-1', 'pickup_0-1', 'toggle-1' into
    short natural-language fragments.
    """
    text_list = []
    for action in action_list:
        if action == "input_img":
            text_list.append("Initial State: ")
        elif action == "start":
            text_list.append("Start Point. ")
        else:
            meta_action = action.split("-")[0]  # e.g. 'pickup_0'
            base = meta_action.split("_")[0]    # e.g. 'pickup'

            if base in ("left", "right", "forward"):
                text_list.append(f"Move {base}. ")
            elif base == "pickup":
                text_list.append("Pick up the object. ")
            elif base == "drop":
                text_list.append("Drop the object. ")
            elif base == "toggle":
                text_list.append("Toggle the device. ")
            else:
                # Fallback
                text_list.append(f"Action {base}. ")
    return text_list


def describe_action_token(action_token):
    """
    For <ACTION_SEQ> description: action_token is something like 'left-1',
    'pickup_0-1', etc. We convert to a concise phrase.
    """
    name = action_token.split("-")[0]  # e.g. 'pickup_0'
    base = name.split("_")[0]

    if base in ("left", "right", "forward"):
        return f"Move {base}."
    elif base == "pickup":
        return "Pick up the object."
    elif base == "drop":
        return "Drop the object."
    elif base == "toggle":
        return "Toggle the device."
    else:
        return f"Action {base}."


def get_answer(data_item):
    """
    For MiniBehavior, exec_state is already one of {'A','B','C','D'}.
    We simply return that letter for the global outcome classification.
    """
    return data_item["exec_state"]
