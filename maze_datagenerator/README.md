# Maze Dataset: Reproducible Generation & Visualization Pipeline

This repository provides a **fully reproducible maze-trajectory dataset generator** used in our paper.
It extends the official MazeDataset library:

 **[https://github.com/understanding-search/maze-dataset](https://github.com/understanding-search/maze-dataset)**

and introduces deterministic, step-wise path-prefix visualization needed for multi-step spatial reasoning.

Our pipeline includes:

1. **An updated plotting module** (`maze_dataset/plotting/plot_maze.py`)
   – extended with step-wise rendering, deterministic arrow-based paths, and fixed candidate-letter placement.
2. **A complete dataset generation script** for producing trajectories, frames, metadata, and multiple-choice candidates.
3. **A reproducible environment configuration** guaranteeing that all images, paths, and JSON outputs are identical across machines.

---

# 1. Environment Setup

We use **Python ≥ 3.10**.

Install dependencies:

```bash
pip install maze-dataset
pip install matplotlib numpy pillow
```

The dataset generator depends on:

* **maze-dataset** – procedural maze construction, path extraction
* **matplotlib** – rendering frames
* **numpy**, **random** – controlled RNG for determinism
* **Pillow** – saving PNG images

---

# 2. Deterministic Reproducibility (Important)

We fix *all* random number generators:

```python
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
```

This ensures:

* identical maze layouts
* identical ground-truth paths
* identical candidate locations
* identical letter assignments
* identical frame rendering

**Every run generates byte-identical results across machines.**

---

# 3. Modifications to `plot_maze.py`

The original MazeDataset visualization utilities do **not** support step‐wise path-prefix frames, deterministic letter placement, or consistent arrow rendering.
Our dataset requires these extensions.

We therefore modified `maze_dataset/plotting/plot_maze.py` with the following:

---

## (1) Arrow-Based Path Rendering

We replaced the default line/heatmap rendering with **directional arrows**:

* arrows show movement directions between steps
* start position drawn as a filled circle
* target position drawn as an X
* consistent scale across maze sizes

This makes step-wise reasoning frames more interpretable.

---

## (2) Deterministic `plot_with_random_letters()`

We introduce:

```python
plot_with_random_letters(...)
```

Features:

* assigns **exactly four deterministic candidate letters** (A, B, C, D)
* consistent candidate placement across all prefix frames of a trajectory
* ability to highlight the *true* target endpoint
* fixed letter mapping per maze → reproducible across runs

This implements the **multiple-choice reasoning setup** that is not present in the original library.

---

## (3) Step-Prefix Rendering

We add support for:

```python
path_prefix_len = t
```

Thus frame t shows:

```
start → step 1 → step 2 → ... → step t
```

This enables fine-grained step-by-step supervision.

---

## (4) Improved Maze Coloring & Rendering Stability

We fixed:

* colorbar inconsistencies
* wall thickness variation
* node-color artifacts when heatmaps are off
* figure layout nondeterminism in matplotlib

The result is consistent and clean visualization for all frames.

---

# 4. Dataset Generation Pipeline

The dataset generator:

1. creates mazes using the MazeDataset library (DFS-based)
2. computes ground-truth paths
3. extracts directional actions (`up`, `down`, `left`, `right`)
4. samples four candidate endpoints
5. assigns deterministic letters (A–D)
6. renders **every prefix of the path**
7. stores metadata + frames

### Key parameters

| Parameter              | Description                        |
| ---------------------- | ---------------------------------- |
| `grid_n`               | Maze side length (3, 4, 5, 6)      |
| `n_mazes`              | Number of mazes per grid size      |
| `path_prefix_len`      | How many steps to render per frame |
| `candidate_locs`       | Four multiple-choice endpoints     |
| `correct_answer_index` | Which candidate is correct         |
| `maze_info`            | Serialized representation of walls |

---

# 5. Frame-by-Frame Output

For each maze **i** and path length **T**:

```
maze/grid{N}/path_i_0.png
maze/grid{N}/path_i_1.png
...
maze/grid{N}/path_i_(T-1).png
```

Each image corresponds to a path prefix.

This design enables models to:

* build spatial understanding step-by-step
* incrementally reason over partially traversed mazes
* choose the correct target out of four candidates

---

# 6. JSON Metadata Format

Metadata is saved as:

```
maze/grid{N}_paths.json
```

Each record:

```json
{
  "path_id": 0,
  "grid_num": 5,
  "path_locs": [[0, 0], [1, 0], [2, 0], ...],
  "file_prefix": "maze/grid5/path_0",
  "action_list": ["down-1", "down-1", "right-1", ...],
  "candidate_locs": [[4, 4], [1, 2], [3, 0], [0, 4]],
  "correct_answer_index": 2,
  "maze_info": "FTFTFFFFTTTF..."
}
```

Where:

* **`path_locs`**: ground-truth coordinates
* **`action_list`**: step-by-step movement directions
* **`candidate_locs`**: multiple-choice endpoints
* **`correct_answer_index`**: label for MCQ task
* **`maze_info`**: full maze structural encoding (DFS walls, serialized row-major)

This metadata allows full reconstruction of the maze and task instance.

---

