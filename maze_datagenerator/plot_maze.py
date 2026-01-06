# orignial maze-dataset:

# maze_dataset\plotting\plot_maze.py:

# ```python
"""provides `MazePlot`, which has many tools for plotting mazes with multiple paths, colored nodes, and more"""

from __future__ import annotations  # for type hinting self as return value

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence, Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float

from maze_dataset import MazeDataset
from maze_dataset.constants import Coord, CoordArray, CoordList
from maze_dataset.maze import (
	LatticeMaze,
	SolvedMaze,
	TargetedLatticeMaze,
)
import random
LARGE_NEGATIVE_NUMBER: float = -1e10


@dataclass(kw_only=True)
class PathFormat:
	"""formatting options for path plot"""

	label: str | None = None
	fmt: str = "o"
	color: str | None = None
	cmap: str | None = None
	line_width: float | None = None
	quiver_kwargs: dict | None = None

	def combine(self, other: PathFormat) -> PathFormat:
		"""combine with other PathFormat object, overwriting attributes with non-None values.

		returns a modified copy of self.
		"""
		output: PathFormat = deepcopy(self)
		for key, value in other.__dict__.items():
			if key == "path":
				err_msg: str = f"Cannot overwrite path attribute! {self = }, {other = }"
				raise ValueError(
					err_msg,
				)
			if value is not None:
				setattr(output, key, value)

		return output


# styled path
@dataclass
class StyledPath(PathFormat):
	"a `StyledPath` is a `PathFormat` with a specific path"

	path: CoordArray


DEFAULT_FORMATS: dict[str, PathFormat] = {
	"true": PathFormat(
		label="true path",
		fmt="--",
		color="red",
		line_width=2.5,
		quiver_kwargs=None,
	),
	"predicted": PathFormat(
		label=None,
		fmt=":",
		color=None,
		line_width=2,
		quiver_kwargs={"width": 0.015},
	),
}


def process_path_input(
	path: CoordList | CoordArray | StyledPath,
	_default_key: str,
	path_fmt: PathFormat | None = None,
	**kwargs,
) -> StyledPath:
	"convert a path, which might be a list or array of coords, into a `StyledPath`"
	styled_path: StyledPath
	if isinstance(path, StyledPath):
		styled_path = path
	elif isinstance(path, np.ndarray):
		styled_path = StyledPath(path=path)
		# add default formatting
		styled_path = styled_path.combine(DEFAULT_FORMATS[_default_key])
	elif isinstance(path, list):
		styled_path = StyledPath(path=np.array(path))
		# add default formatting
		styled_path = styled_path.combine(DEFAULT_FORMATS[_default_key])
	else:
		err_msg: str = (
			f"Expected CoordList, CoordArray or StyledPath, got {type(path)}: {path}"
		)
		raise TypeError(
			err_msg,
		)

	# add formatting from path_fmt
	if path_fmt is not None:
		styled_path = styled_path.combine(path_fmt)

	# add formatting from kwargs
	for key, value in kwargs.items():
		setattr(styled_path, key, value)

	return styled_path


DEFAULT_PREDICTED_PATH_COLORS: list[str] = [
	"tab:orange",
	"tab:olive",
	"sienna",
	"mediumseagreen",
	"tab:purple",
	"slategrey",
]


class MazePlot:
	"""Class for displaying mazes and paths"""

	def __init__(self, maze: LatticeMaze, unit_length: int = 14) -> None:
		"""UNIT_LENGTH: Set ratio between node size and wall thickness in image.

		Wall thickness is fixed to 1px
		A "unit" consists of a single node and the right and lower connection/wall.
		Example: ul = 14 yields 13:1 ratio between node size and wall thickness
		"""
		self.unit_length: int = unit_length
		self.maze: LatticeMaze = maze
		self.true_path: StyledPath | None = None
		self.predicted_paths: list[StyledPath] = []
		self.node_values: Float[np.ndarray, "grid_n"] = None
		self.custom_node_value_flag: bool = False
		self.node_color_map: str = "Blues"
		self.target_token_coord: Coord = None
		self.preceding_tokens_coords: CoordArray = None
		self.colormap_center: float | None = None
		self.cbar_ax = None
		self.marked_coords: list[tuple[Coord, dict]] = list()

		self.marker_kwargs_current: dict = dict(
			marker="s",
			color="green",
			ms=12,
		)
		self.marker_kwargs_next: dict = dict(
			marker="P",
			color="green",
			ms=12,
		)

		if isinstance(maze, SolvedMaze):
			self.add_true_path(maze.solution)
		else:
			if isinstance(maze, TargetedLatticeMaze):
				self.add_true_path(SolvedMaze.from_targeted_lattice_maze(maze).solution)

	@property
	def solved_maze(self) -> SolvedMaze:
		"get the underlying `SolvedMaze` object"
		if self.true_path is None:
			raise ValueError(
				"Cannot return SolvedMaze object without true path. Add true path with add_true_path method.",
			)
		return SolvedMaze.from_lattice_maze(
			lattice_maze=self.maze,
			solution=self.true_path.path,
		)

	def add_true_path(
		self,
		path: CoordList | CoordArray | StyledPath,
		path_fmt: PathFormat | None = None,
		**kwargs,
	) -> MazePlot:
		"add a true path to the maze with optional formatting"
		self.true_path = process_path_input(
			path=path,
			_default_key="true",
			path_fmt=path_fmt,
			**kwargs,
		)

		return self

	def add_predicted_path(
		self,
		path: CoordList | CoordArray | StyledPath,
		path_fmt: PathFormat | None = None,
		**kwargs,
	) -> MazePlot:
		"""Recieve predicted path and formatting preferences from input and save in predicted_path list.

		Default formatting depends on nuber of paths already saved in predicted path list.
		"""
		styled_path: StyledPath = process_path_input(
			path=path,
			_default_key="predicted",
			path_fmt=path_fmt,
			**kwargs,
		)

		# set default label and color if not specified
		if styled_path.label is None:
			styled_path.label = f"predicted path {len(self.predicted_paths) + 1}"

		if styled_path.color is None:
			color_num: int = len(self.predicted_paths) % len(
				DEFAULT_PREDICTED_PATH_COLORS,
			)
			styled_path.color = DEFAULT_PREDICTED_PATH_COLORS[color_num]

		self.predicted_paths.append(styled_path)
		return self

	def add_multiple_paths(
		self,
		path_list: Sequence[CoordList | CoordArray | StyledPath],
	) -> MazePlot:
		"""Function for adding multiple paths to MazePlot at once.

		> DOCS: what are the two ways?
		This can be done in two ways:
		1. Passing a list of
		"""
		for path in path_list:
			self.add_predicted_path(path)
		return self

	def add_node_values(
		self,
		node_values: Float[np.ndarray, "grid_n grid_n"],
		color_map: str = "Blues",
		target_token_coord: Coord | None = None,
		preceeding_tokens_coords: CoordArray = None,
		colormap_center: float | None = None,
		colormap_max: float | None = None,
		hide_colorbar: bool = False,
	) -> MazePlot:
		"""add node values to the maze for visualization as a heatmap

		> DOCS: what are these arguments?

		# Parameters:
		- `node_values : Float[np.ndarray, &quot;grid_n grid_n&quot;]`
		- `color_map : str`
			(defaults to `"Blues"`)
		- `target_token_coord : Coord | None`
			(defaults to `None`)
		- `preceeding_tokens_coords : CoordArray`
			(defaults to `None`)
		- `colormap_center : float | None`
			(defaults to `None`)
		- `colormap_max : float | None`
			(defaults to `None`)
		- `hide_colorbar : bool`
			(defaults to `False`)

		# Returns:
		- `MazePlot`
		"""
		assert node_values.shape == self.maze.grid_shape, (
			"Please pass node values of the same sape as LatticeMaze.grid_shape"
		)
		# assert np.min(node_values) >= 0, "Please pass non-negative node values only."

		self.node_values = node_values
		# Set flag for choosing cmap while plotting maze
		self.custom_node_value_flag = True
		# Retrieve Max node value for plotting, +1e-10 to avoid division by zero
		self.node_color_map = color_map
		self.colormap_center = colormap_center
		self.colormap_max = colormap_max
		self.hide_colorbar = hide_colorbar

		if target_token_coord is not None:
			self.marked_coords.append((target_token_coord, self.marker_kwargs_next))
		if preceeding_tokens_coords is not None:
			for coord in preceeding_tokens_coords:
				self.marked_coords.append((coord, self.marker_kwargs_current))
		return self

	def plot(
		self,
		dpi: int = 100,
		title: str = "",
		fig_ax: tuple | None = None,
		plain: bool = False,
	) -> MazePlot:
		"""Plot the maze and paths."""
		# set up figure
		if fig_ax is None:
			self.fig = plt.figure(dpi=dpi)
			self.ax = self.fig.add_subplot(1, 1, 1)
		else:
			self.fig, self.ax = fig_ax

		# plot maze
		self._plot_maze()

		# Plot labels
		if not plain:
			tick_arr = np.arange(self.maze.grid_shape[0])
			self.ax.set_xticks(self.unit_length * (tick_arr + 0.5), tick_arr)
			self.ax.set_yticks(self.unit_length * (tick_arr + 0.5), tick_arr)
			self.ax.set_xlabel("col")
			self.ax.set_ylabel("row")
			self.ax.set_title(title)
		else:
			self.ax.set_xticks([])
			self.ax.set_yticks([])
			self.ax.set_xlabel("")
			self.ax.set_ylabel("")
			self.ax.axis("off")

		# plot paths
		if self.true_path is not None:
			self._plot_path(self.true_path)
		for path in self.predicted_paths:
			self._plot_path(path)

		# plot markers
		for coord, kwargs in self.marked_coords:
			self._place_marked_coords([coord], **kwargs)

		return self

	def _rowcol_to_coord(self, point: Coord) -> np.ndarray:
		"""Transform Point from MazeTransformer (row, column) notation to matplotlib default (x, y) notation where x is the horizontal axis."""
		point = np.array([point[1], point[0]])
		return self.unit_length * (point + 0.5)

	def mark_coords(self, coords: CoordArray | list[Coord], **kwargs) -> MazePlot:
		"""Mark coordinates on the maze with a marker.

		default marker is a blue "+":
		`dict(marker="+", color="blue")`
		"""
		kwargs = {
			**dict(marker="+", color="blue"),
			**kwargs,
		}
		for coord in coords:
			self.marked_coords.append((coord, kwargs))

		return self

	def _place_marked_coords(
		self,
		coords: CoordArray | list[Coord],
		**kwargs,
	) -> MazePlot:
		coords_tp = np.array([self._rowcol_to_coord(coord) for coord in coords])
		self.ax.plot(coords_tp[:, 0], coords_tp[:, 1], **kwargs)

		return self

	def _plot_maze(self) -> None:  # noqa: C901, PLR0912
		"""Define Colormap and plot maze.

		Colormap: x is -inf: black
		else: use colormap
		"""
		img = self._lattice_maze_to_img()

		# if no node_values have been passed (no colormap)
		if self.custom_node_value_flag is False:
			self.ax.imshow(img, cmap="gray", vmin=-1, vmax=1)

		else:
			assert self.node_values is not None, "Please pass node values."
			assert not np.isnan(self.node_values).any(), (
				"Please pass node values, they cannot be nan."
			)

			vals_min: float = np.nanmin(self.node_values)
			vals_max: float = np.nanmax(self.node_values)
			# if both are negative or both are positive, set max/min to 0
			if vals_max < 0.0:
				vals_max = 0.0
			elif vals_min > 0.0:
				vals_min = 0.0

			# adjust vals_max, in case you need consistent colorbar across multiple plots
			vals_max = self.colormap_max or vals_max

			# create colormap
			cmap = mpl.colormaps[self.node_color_map]
			# TODO: this is a hack, we make the walls black (while still allowing negative values) by setting the nan color to black
			cmap.set_bad(color="black")

			if self.colormap_center is not None:
				if not (vals_min < self.colormap_center < vals_max):
					if vals_min == self.colormap_center:
						vals_min -= 1e-10
					elif vals_max == self.colormap_center:
						vals_max += 1e-10
					else:
						err_msg: str = f"Please pass colormap_center value between {vals_min} and {vals_max}"
						raise ValueError(
							err_msg,
						)

				norm = mpl.colors.TwoSlopeNorm(
					vmin=vals_min,
					vcenter=self.colormap_center,
					vmax=vals_max,
				)
				_plotted = self.ax.imshow(img, cmap=cmap, norm=norm)
			else:
				_plotted = self.ax.imshow(img, cmap=cmap, vmin=vals_min, vmax=vals_max)

			# Add colorbar based on the condition of self.hide_colorbar
			if not self.hide_colorbar:
				ticks = np.linspace(vals_min, vals_max, 5)

				if (vals_min < 0.0 < vals_max) and (0.0 not in ticks):
					ticks = np.insert(ticks, np.searchsorted(ticks, 0.0), 0.0)

				if (
					self.colormap_center is not None
					and self.colormap_center not in ticks
					and vals_min < self.colormap_center < vals_max
				):
					ticks = np.insert(
						ticks,
						np.searchsorted(ticks, self.colormap_center),
						self.colormap_center,
					)

				cbar = plt.colorbar(
					_plotted,
					ticks=ticks,
					ax=self.ax,
					cax=self.cbar_ax,
				)
				self.cbar_ax = cbar.ax

		# make the boundaries of the image thicker (walls look weird without this)
		for axis in ["top", "bottom", "left", "right"]:
			self.ax.spines[axis].set_linewidth(2)

	def _lattice_maze_to_img(
		self,
		connection_val_scale: float = 0.93,
	) -> Bool[np.ndarray, "row col"]:
		"""Build an image to visualise the maze.

		Each "unit" consists of a node and the right and lower adjacent wall/connection. Its area is ul * ul.
		- Nodes have area: (ul-1) * (ul-1) and value 1 by default
			- take node_value if passed via .add_node_values()
		- Walls have area: 1 * (ul-1) and value -1
		- Connections have area: 1 * (ul-1); color and value 0.93 by default
			- take node_value if passed via .add_node_values()

		Axes definition:
		(0,0)     col
		----|----------->
			|
		row |
			|
			v

		Returns a matrix of side length (ul) * n + 1 where n is the number of nodes.
		"""
		# TODO: this is a hack, but if you add 1 always then non-node valued plots have their walls dissapear. if you dont add 1, you get ugly colors between nodes when they are colored
		node_bdry_hack: int
		connection_list_processed: Float[np.ndarray, "dim row col"]
		# Set node and connection values
		if self.node_values is None:
			scaled_node_values = np.ones(self.maze.grid_shape)
			connection_values = scaled_node_values * connection_val_scale
			node_bdry_hack = 0
			# TODO: hack
			# invert connection list
			connection_list_processed = np.logical_not(self.maze.connection_list)
		else:
			# TODO: hack
			scaled_node_values = self.node_values
			# connection_values = scaled_node_values
			connection_values = np.full_like(scaled_node_values, np.nan)
			node_bdry_hack = 1
			connection_list_processed = self.maze.connection_list

		# Create background image (all pixels set to -1, walls everywhere)
		img: Float[np.ndarray, "row col"] = -np.ones(
			(
				self.maze.grid_shape[0] * self.unit_length + 1,
				self.maze.grid_shape[1] * self.unit_length + 1,
			),
			dtype=float,
		)

		# Draw nodes and connections by iterating through lattice
		for row in range(self.maze.grid_shape[0]):
			for col in range(self.maze.grid_shape[1]):
				# Draw node
				img[
					row * self.unit_length + 1 : (row + 1) * self.unit_length
					+ node_bdry_hack,
					col * self.unit_length + 1 : (col + 1) * self.unit_length
					+ node_bdry_hack,
				] = scaled_node_values[row, col]

				# Down connection
				if not connection_list_processed[0, row, col]:
					img[
						(row + 1) * self.unit_length,
						col * self.unit_length + 1 : (col + 1) * self.unit_length,
					] = connection_values[row, col]

				# Right connection
				if not connection_list_processed[1, row, col]:
					img[
						row * self.unit_length + 1 : (row + 1) * self.unit_length,
						(col + 1) * self.unit_length,
					] = connection_values[row, col]

		return img

	# def _plot_path(self, path_format: PathFormat) -> None:
	# 	if len(path_format.path) == 0:
	# 		warnings.warn(f"Empty path, skipping plotting\n{path_format = }")
	# 		return
	# 	p_transformed = np.array(
	# 		[self._rowcol_to_coord(coord) for coord in path_format.path],
	# 	)
	# 	if path_format.quiver_kwargs is not None:
	# 		try:
	# 			x: np.ndarray = p_transformed[:, 0]
	# 			y: np.ndarray = p_transformed[:, 1]
	# 		except Exception as e:
	# 			err_msg: str = f"Error in plotting quiver path:\n{path_format = }\n{p_transformed = }\n{e}"
	# 			raise ValueError(
	# 				err_msg,
	# 			) from e

	# 		# Generate colors from the colormap
	# 		if path_format.cmap is not None:
	# 			n = len(x) - 1  # Number of arrows
	# 			cmap = plt.get_cmap(path_format.cmap)
	# 			colors = [cmap(i / n) for i in range(n)]
	# 		else:
	# 			colors = path_format.color

	# 		self.ax.quiver(
	# 			x[:-1],
	# 			y[:-1],
	# 			x[1:] - x[:-1],
	# 			y[1:] - y[:-1],
	# 			scale_units="xy",
	# 			angles="xy",
	# 			scale=1,
	# 			color=colors,
	# 			**path_format.quiver_kwargs,
	# 		)
	# 	else:
	# 		self.ax.plot(
	# 			p_transformed[:, 0],
	# 			p_transformed[:, 1],
	# 			path_format.fmt,
	# 			lw=path_format.line_width,
	# 			color=path_format.color,
	# 			label=path_format.label,
	# 		)
	# 	# mark endpoints
	# 	self.ax.plot(
	# 		[p_transformed[0][0]],
	# 		[p_transformed[0][1]],
	# 		"o",
	# 		color=path_format.color,
	# 		ms=10,
	# 	)
	# 	self.ax.plot(
	# 		[p_transformed[-1][0]],
	# 		[p_transformed[-1][1]],
	# 		"x",
	# 		color=path_format.color,
	# 		ms=10,
	# 	)


	def _plot_path(self, path_format: PathFormat) -> None:
		if len(path_format.path) == 0:
			warnings.warn(f"Empty path, skipping plotting\n{path_format = }")
			return

		# Convert (row, col) → plotting (x, y)
		p_transformed = np.array(
			[self._rowcol_to_coord(coord) for coord in path_format.path]
		)

		# -------------------------------------------------------------
		# New version: draw arrows for every step
		# -------------------------------------------------------------
		x = p_transformed[:, 0]
		y = p_transformed[:, 1]

		for i in range(len(x) - 1):
			dx = x[i + 1] - x[i]
			dy = y[i + 1] - y[i]

			# Draw arrow (scale_units="xy" makes them consistent)
			self.ax.arrow(
				x[i], y[i],           # starting point
				dx, dy,               # direction
				length_includes_head=True,
				head_width=0.6 * self.unit_length / 6,
				head_length=0.6 * self.unit_length / 5,
				fc=path_format.color,
				ec=path_format.color,
				linewidth=path_format.line_width if path_format.line_width else 2,
				overhang=0.3,
			)

		# -------------------------------------------------------------
		# Also mark start (circle) and end (X)
		# -------------------------------------------------------------
		self.ax.plot(
			[p_transformed[0][0]],
			[p_transformed[0][1]],
			"o",
			color=path_format.color,
			ms=10,
		)
		self.ax.plot(
			[p_transformed[-1][0]],
			[p_transformed[-1][1]],
			"x",
			color=path_format.color,
			ms=10,
			mew=2.5,
		)


	def to_ascii(
		self,
		show_endpoints: bool = True,
		show_solution: bool = True,
	) -> str:
		"wrapper for `self.solved_maze.as_ascii()`, shows the path if we have `self.true_path`"
		if self.true_path:
			return self.solved_maze.as_ascii(
				show_endpoints=show_endpoints,
				show_solution=show_solution,
			)
		else:
			return self.maze.as_ascii(show_endpoints=show_endpoints)


	def plot_with_random_letters(
			self,
			candidate_coords: list[tuple[int, int]],
			dpi: int = 100,
			title: str = "",
			fig_ax: tuple | None = None,
			plain: bool = False,
			show_true_path: bool = True,
			show_start: bool = True,
			show_rowcol_labels: bool = True,
			path_prefix_len: int | None = None,
			coord_letter_map: dict[tuple[int, int], str] | None = None,
	) -> tuple["MazePlot", str]:
		"""
		Plot the maze with four letters A, B, C, D at the given candidate coordinates.

		If coord_letter_map is provided, it is used as-is (no randomness),
		so the same mapping can be reused across steps of the same path.
		"""

		# Validate candidate coords
		if len(candidate_coords) < 4:
			raise ValueError(
				f"Not enough candidate coordinates: need at least 4, got {len(candidate_coords)}"
			)

		# Get true path
		if self.true_path is None or len(self.true_path.path) == 0:
			raise ValueError("No true path available to determine the endpoint")

		full_true_path = self.true_path.path

		true_endpoint = full_true_path[-1]
		true_start = full_true_path[0]

		if isinstance(true_endpoint, np.ndarray):
			true_endpoint = tuple(true_endpoint)
		else:
			true_endpoint = tuple(true_endpoint)

		if isinstance(true_start, np.ndarray):
			true_start = tuple(true_start)
		else:
			true_start = tuple(true_start)

		if true_endpoint not in candidate_coords:
			raise ValueError(
				f"True endpoint {true_endpoint} must be included in candidate coordinates"
			)

		# -------------------------------------------------
		# Set coord_letter_map: either fixed (given) or random
		# -------------------------------------------------
		if coord_letter_map is None:
			# Randomly generate mapping once (old behavior)
			other_candidates = [coord for coord in candidate_coords if coord != true_endpoint]
			if len(other_candidates) < 3:
				raise ValueError(
					"Not enough other candidates to place 3 additional letters: "
					f"need at least 3, got {len(other_candidates)}"
				)

			# True endpoint + 3 others
			selected_other_coords = random.sample(other_candidates, 3)
			selected_coords = [true_endpoint] + selected_other_coords

			letters = ["A", "B", "C", "D"]
			correct_letter = random.choice(letters)

			remaining_letters = [l for l in letters if l != correct_letter]
			random.shuffle(remaining_letters)

			all_selected_letters = [correct_letter] + remaining_letters
			coord_letter_map = dict(zip(selected_coords, all_selected_letters))

			true_target_letter = correct_letter
		else:
			# Use provided fixed mapping; assume it covers the 4 candidate coords
			if true_endpoint not in coord_letter_map:
				raise ValueError(
					f"Provided coord_letter_map does not contain true endpoint {true_endpoint}"
				)
			true_target_letter = coord_letter_map[true_endpoint]

		# -------------------------------------------------
		# Set up fig/ax and draw maze + path + letters
		# -------------------------------------------------
		if fig_ax is None:
			fig = plt.figure(dpi=dpi)
			ax = fig.add_subplot(1, 1, 1)
		else:
			fig, ax = fig_ax

		original_fig = getattr(self, "fig", None)
		original_ax = getattr(self, "ax", None)
		self.fig = fig
		self.ax = ax

		# Draw maze
		self._plot_maze()

		# Axes labels
		if not plain:
			tick_arr = np.arange(self.maze.grid_shape[0])
			if show_rowcol_labels:
				ax.set_xticks(self.unit_length * (tick_arr + 0.5))
				ax.set_xticklabels(tick_arr)
				ax.set_yticks(self.unit_length * (tick_arr + 0.5))
				ax.set_yticklabels(tick_arr)
				ax.set_xlabel("col")
				ax.set_ylabel("row")
			else:
				ax.set_xticks(self.unit_length * (tick_arr + 0.5))
				ax.set_xticklabels([])
				ax.set_yticks(self.unit_length * (tick_arr + 0.5))
				ax.set_yticklabels([])
				ax.set_xlabel("")
				ax.set_ylabel("")
			if title != "":
				ax.set_title(title)
		else:
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xlabel("")
			ax.set_ylabel("")
			ax.axis("off")

		# Plot true path or just start
		if show_true_path and self.true_path is not None:
			if path_prefix_len is None:
				path_to_plot = full_true_path
			else:
				path_to_plot = full_true_path[:path_prefix_len]

			original_path = self.true_path.path
			self.true_path.path = np.array(path_to_plot)
			self._plot_path(self.true_path)
			self.true_path.path = original_path

		elif show_start and self.true_path is not None and len(full_true_path) > 0:
			start_coord_arr = np.array([self._rowcol_to_coord(true_start)])
			self.ax.plot(
				[start_coord_arr[0][0]],
				[start_coord_arr[0][1]],
				"o",
				color=(self.true_path.color or "red"),
				ms=10,
			)

		# Predicted paths and markers
		for path in self.predicted_paths:
			self._plot_path(path)

		for coord, kwargs in self.marked_coords:
			self._place_marked_coords([coord], **kwargs)

		# Draw letters using coord_letter_map
		for coord, letter in coord_letter_map.items():
			plot_coord = self._rowcol_to_coord(coord)
			# ax.text(
			# 	plot_coord[0],
			# 	plot_coord[1],
			# 	letter,
			# 	ha="center",
			# 	va="center",
			# 	fontsize=16,
			# 	weight="bold",
			# 	bbox=dict(
			# 		boxstyle="round,pad=0.3",
			# 		facecolor="white",
			# 		alpha=0.9,
			# 		edgecolor="black",
			# 		linewidth=1.5,
			# 	),
			# )
			ax.text(
				plot_coord[0],
				plot_coord[1],
				letter,
				ha="center",
				va="center",
				fontsize=20,
				weight="bold",
				color="black",        # You can change this if needed
			)


		if original_fig is not None:
			self.fig = original_fig
		if original_ax is not None:
			self.ax = original_ax

		return self, true_target_letter


# from maze_dataset.plotting.plot_maze import MazePlot
# import inspect

# print(hasattr(MazePlot, "plot_with_random_letters"))
# print(inspect.getsource(MazePlot))

# ```

## new generation function

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from pathlib import Path
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

grid_n_s = [(3, 100), (4, 100), (5, 100), (6, 100)]

for grid_n, n_mazes in grid_n_s:

    cfg = MazeDatasetConfig(
        name="test",
        grid_n=grid_n,
        n_mazes=n_mazes,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
        maze_ctor_kwargs=dict(do_forks=False),
    )

    dataset = MazeDataset.from_config(cfg)
    output_list = []

    for i, maze_data in enumerate(dataset.mazes):

        plot = MazePlot(maze_data)

        # Extract true path
        true_path_raw = plot.true_path.path
        path_coords = [(int(p[0]), int(p[1])) for p in true_path_raw]
        start_coord = path_coords[0]
        end_coord = path_coords[-1]

        # Build action list
        action_list = []
        for j in range(1, len(path_coords)):
            r0, c0 = path_coords[j - 1]
            r1, c1 = path_coords[j]
            if r1 > r0: action_list.append("down-1")
            elif r1 < r0: action_list.append("up-1")
            elif c1 > c0: action_list.append("right-1")
            elif c1 < c0: action_list.append("left-1")

        # Candidates: fixed per path
        all_coords = [(r, c) for r in range(grid_n) for c in range(grid_n)]
        other_coords = [c for c in all_coords if c != end_coord and c != start_coord]
        selected_other_coords = random.sample(other_coords, 3)

        candidate_locs = [end_coord] + selected_other_coords
        random.shuffle(candidate_locs)  # shuffle candidate positions

        correct_answer_index = candidate_locs.index(end_coord)

        # ---------- NEW: fixed coord_letter_map per path ----------
        letters = ["A", "B", "C", "D"]
        random.shuffle(letters)  # random per path, but fixed across steps
        coord_letter_map = {
            coord: letter for coord, letter in zip(candidate_locs, letters)
        }
        # -----------------------------------------------------------

        # Serialize maze structure
        conn = maze_data.connection_list
        maze_info = ""
        for r in range(grid_n):
            for c in range(grid_n):
                maze_info += "T" if conn[1, r, c] else "F"
                maze_info += "T" if conn[0, r, c] else "F"

        # Save meta for JSON
        output_list.append({
            "path_id": i,
            "grid_num": grid_n,
            "path_locs": path_coords,
            "file_prefix": f"maze/grid{grid_n}/path_{i}",
            "action_list": action_list,
            "candidate_locs": candidate_locs,
            "correct_answer_index": correct_answer_index,
            "maze_info": maze_info,
            # optionally store letters if you want
            # "coord_letter_map": coord_letter_map,
        })

        # ----------------- Save frames 0…K-1 -----------------

        num_steps = len(path_coords)

        for step_idx in range(num_steps):

            path_prefix_len = step_idx + 1

            step_plot = MazePlot(maze_data)

            step_plot_with_letters, _ = step_plot.plot_with_random_letters(
                candidate_coords=candidate_locs,
                title="",
                show_true_path=True,
                show_rowcol_labels=False,
                show_start=True,
                path_prefix_len=path_prefix_len,
                coord_letter_map=coord_letter_map,  # <-- FIXED MAPPING HERE
            )

            step_fname = f"maze/grid{grid_n}/path_{i}_{step_idx}.png"
            Path(step_fname).parent.mkdir(parents=True, exist_ok=True)

            step_plot_with_letters.fig.savefig(
                step_fname,
                dpi=150,
                bbox_inches="tight"
            )
            plt.close(step_plot_with_letters.fig)
            print(f"Saved: {step_fname}")

    # Save JSON
    json_filename = f"maze/grid{grid_n}_paths.json"
    Path(json_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(json_filename, "w") as f:
        json.dump(output_list, f, indent=2)

    print(f"Saved JSON: {json_filename}")



