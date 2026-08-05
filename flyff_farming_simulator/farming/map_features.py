from __future__ import annotations

from collections import OrderedDict
from enum import IntEnum, unique
from heapq import heappop, heappush
from math import hypot, inf, isfinite, sqrt
from threading import RLock
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]
Cell = tuple[int, int]
DEFAULT_TELEPORT_BUFFER_RADIUS_CELLS = 2.0
DEFAULT_GEODESIC_CACHE_SIZE = 512
DEFAULT_MAXIMUM_GEODESIC_EXPANSIONS = 8_192
DEFAULT_CONTEXT_MAP_RADIUS_CELLS = 50
DEFAULT_CONTEXT_MAP_SIDE = 21

_DIRECTIONS: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (1, -1, sqrt(2.0)),
    (0, -1, 1.0),
    (-1, -1, sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, sqrt(2.0)),
    (0, 1, 1.0),
    (1, 1, sqrt(2.0)),
)


@unique
class DirectPathState(IntEnum):
    """Wall evidence for the straight player-to-actor segment."""

    BLOCKED = -1
    UNKNOWN = 0
    CLEAR = 1


@unique
class MapCellRisk(IntEnum):
    """Semantic map state at one player/layout cell.

    The values are ordered by increasing policy risk.  They are not written
    directly into the observation; ``local_crop`` uses the explicit encodings
    below so outside/unknown can remain neutral.
    """

    OUTSIDE_OR_UNKNOWN = 0
    SAFE = 1
    OBSTACLE_BUFFER = 2
    OBSTACLE = 3
    TELEPORT_BUFFER = 4
    TELEPORT_TRIGGER = 5


LOCAL_MAP_SAFE: float = -1.0
LOCAL_MAP_OBSTACLE_BUFFER: float = -0.25
LOCAL_MAP_OUTSIDE_OR_UNKNOWN: float = 0.0
LOCAL_MAP_OBSTACLE: float = 0.50
LOCAL_MAP_TELEPORT_BUFFER: float = 0.75
LOCAL_MAP_TELEPORT_TRIGGER: float = 1.0


def _risk_encoding(risk: MapCellRisk) -> np.float32:
    return np.float32(
        {
            MapCellRisk.OUTSIDE_OR_UNKNOWN: LOCAL_MAP_OUTSIDE_OR_UNKNOWN,
            MapCellRisk.SAFE: LOCAL_MAP_SAFE,
            MapCellRisk.OBSTACLE_BUFFER: LOCAL_MAP_OBSTACLE_BUFFER,
            MapCellRisk.OBSTACLE: LOCAL_MAP_OBSTACLE,
            MapCellRisk.TELEPORT_BUFFER: LOCAL_MAP_TELEPORT_BUFFER,
            MapCellRisk.TELEPORT_TRIGGER: LOCAL_MAP_TELEPORT_TRIGGER,
        }[risk]
    )


class FarmingMapFeatures:
    """Immutable farming map arrays with cached, bounded feature queries.

    Array coordinates use ``(x, y)`` while NumPy storage uses ``[y, x]``.
    Fine local crops and coarse context crops are emitted in row-major order.
    The context crop spans a much wider radius while aggregating risk inside
    each coarse bin so thin walls and teleport zones are not skipped.
    """

    def __init__(
        self,
        *,
        traversable: NDArray[np.bool_] | np.ndarray,
        forbidden: NDArray[np.bool_] | np.ndarray,
        safe_traversable: NDArray[np.bool_] | np.ndarray | None = None,
        teleport_buffer_radius_cells: float = DEFAULT_TELEPORT_BUFFER_RADIUS_CELLS,
        geodesic_cache_size: int = DEFAULT_GEODESIC_CACHE_SIZE,
        maximum_geodesic_expansions: int = DEFAULT_MAXIMUM_GEODESIC_EXPANSIONS,
    ) -> None:
        traversable_array = _readonly_bool_array(traversable, name="traversable")
        forbidden_array = _readonly_bool_array(forbidden, name="forbidden")
        if traversable_array.shape != forbidden_array.shape:
            raise ValueError(
                "traversable and forbidden arrays must have the same shape"
            )

        if safe_traversable is None:
            safe_array = _readonly_bool_array(
                traversable_array & ~forbidden_array,
                name="safe_traversable",
            )
        else:
            safe_array = _readonly_bool_array(
                safe_traversable,
                name="safe_traversable",
            )
            if safe_array.shape != traversable_array.shape:
                raise ValueError(
                    "safe_traversable and traversable arrays must have the same shape"
                )
            if np.any(safe_array & ~traversable_array):
                raise ValueError(
                    "safe_traversable must be a subset of traversable cells"
                )
            if np.any(safe_array & forbidden_array):
                raise ValueError("safe_traversable cannot include forbidden cells")

        if not isfinite(float(teleport_buffer_radius_cells)):
            raise ValueError("teleport_buffer_radius_cells must be finite")
        if teleport_buffer_radius_cells < 0.0:
            raise ValueError("teleport_buffer_radius_cells cannot be negative")
        if geodesic_cache_size < 1:
            raise ValueError("geodesic_cache_size must be positive")
        if maximum_geodesic_expansions < 1:
            raise ValueError("maximum_geodesic_expansions must be positive")

        self._traversable = traversable_array
        self._forbidden = forbidden_array
        self._safe_traversable = safe_array
        self._has_forbidden = bool(np.any(forbidden_array))
        self._teleport_buffer_radius_cells = float(teleport_buffer_radius_cells)
        self._geodesic_cache_size = int(geodesic_cache_size)
        self._maximum_geodesic_expansions = int(maximum_geodesic_expansions)
        self._forbidden_distance_field: FloatArray | None = None
        self._geodesic_cache: OrderedDict[
            tuple[Cell, Cell, float | None, int],
            float,
        ] = OrderedDict()
        self._geodesic_field_cache: OrderedDict[
            tuple[Cell, float, int], dict[Cell, float]
        ] = OrderedDict()
        self._geodesic_field_cache_size = min(8, self._geodesic_cache_size)
        self._lock = RLock()

    @property
    def traversable(self) -> BoolArray:
        return _readonly_view(self._traversable)

    @property
    def forbidden(self) -> BoolArray:
        return _readonly_view(self._forbidden)

    @property
    def safe_traversable(self) -> BoolArray:
        return _readonly_view(self._safe_traversable)

    @property
    def has_forbidden(self) -> bool:
        return self._has_forbidden

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._traversable.shape[0]), int(self._traversable.shape[1])

    @property
    def teleport_buffer_radius_cells(self) -> float:
        return self._teleport_buffer_radius_cells

    @property
    def geodesic_cache_entries(self) -> int:
        with self._lock:
            return len(self._geodesic_cache)

    def contains(self, cell: Cell) -> bool:
        x, y = int(cell[0]), int(cell[1])
        height, width = self.shape
        return 0 <= x < width and 0 <= y < height

    def is_forbidden(self, cell: Cell | None) -> bool:
        if cell is None or not self.contains(cell):
            return False
        x, y = int(cell[0]), int(cell[1])
        return bool(self._forbidden[y, x])

    def cell_risk(self, cell: Cell | None) -> MapCellRisk:
        """Classify one cell without treating the hand-traced wall as exact.

        Teleport evidence has precedence over ordinary wall evidence because
        it is the only map state that can terminate a policy session.  A black
        (non-traversable) cell remains distinct from its inflated traversable
        buffer so reward and observation semantics can weight them differently.
        """

        if cell is None or not self.contains(cell):
            return MapCellRisk.OUTSIDE_OR_UNKNOWN
        x, y = int(cell[0]), int(cell[1])
        forbidden_distance_field = (
            self._get_forbidden_distance_field() if self._has_forbidden else None
        )
        return self._cell_risk_at(x, y, forbidden_distance_field)

    def _cell_risk_at(
        self,
        x: int,
        y: int,
        forbidden_distance_field: FloatArray | None,
    ) -> MapCellRisk:
        height, width = self.shape
        if x < 0 or y < 0 or x >= width or y >= height:
            return MapCellRisk.OUTSIDE_OR_UNKNOWN
        if bool(self._forbidden[y, x]):
            return MapCellRisk.TELEPORT_TRIGGER
        if forbidden_distance_field is not None:
            distance = float(forbidden_distance_field[y, x])
            if distance <= self._teleport_buffer_radius_cells:
                return MapCellRisk.TELEPORT_BUFFER
        if not bool(self._traversable[y, x]):
            return MapCellRisk.OBSTACLE
        if bool(self._safe_traversable[y, x]):
            return MapCellRisk.SAFE
        return MapCellRisk.OBSTACLE_BUFFER

    def is_obstacle(self, cell: Cell | None) -> bool:
        return self.cell_risk(cell) is MapCellRisk.OBSTACLE

    def is_obstacle_buffer(self, cell: Cell | None) -> bool:
        return self.cell_risk(cell) is MapCellRisk.OBSTACLE_BUFFER

    def cell_state(self, cell: Cell) -> float:
        """Return ``-1`` free, ``0`` outside/unknown, or ``+1`` blocked."""

        if not self.contains(cell):
            return 0.0
        x, y = int(cell[0]), int(cell[1])
        return -1.0 if bool(self._safe_traversable[y, x]) else 1.0

    def normalized_position(self, cell: Cell | None) -> tuple[float, float]:
        if cell is None or not self.contains(cell):
            return 0.0, 0.0
        height, width = self.shape
        x_scale = max(1, width - 1)
        y_scale = max(1, height - 1)
        return (
            float(np.clip(cell[0] / x_scale, 0.0, 1.0) * 2.0 - 1.0),
            float(np.clip(cell[1] / y_scale, 0.0, 1.0) * 2.0 - 1.0),
        )

    def forbidden_distance(self, cell: Cell | None) -> float | None:
        """Return exact Euclidean cell distance to the nearest trigger cell."""

        if cell is None or not self.contains(cell) or not self._has_forbidden:
            return None
        field = self._get_forbidden_distance_field()
        x, y = int(cell[0]), int(cell[1])
        value = float(field[y, x])
        return value if isfinite(value) else None

    def local_crop(self, center: Cell | None, side: int = 11) -> FloatArray:
        """Build the policy crop with distinct wall and teleport encodings.

        ``1.0`` is an exact trigger, ``0.75`` its safety buffer, ``0.50`` a
        hand-traced black obstacle, ``-0.25`` the inflated obstacle buffer,
        ``-1.0`` safe traversable, and ``0.0`` outside/unknown.
        """

        side = int(side)
        if side < 1 or side % 2 == 0:
            raise ValueError("local crop side must be a positive odd number")
        result = np.zeros(side * side, dtype=np.float32)
        if center is None:
            return result

        height, width = self.shape
        forbidden_distance_field = (
            self._get_forbidden_distance_field() if self._has_forbidden else None
        )
        center_x, center_y = int(center[0]), int(center[1])
        radius = side // 2
        offset = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = center_x + dx, center_y + dy
                risk = self._cell_risk_at(x, y, forbidden_distance_field)
                result[offset] = _risk_encoding(risk)
                offset += 1
        return result

    def context_crop(
        self,
        center: Cell | None,
        *,
        radius_cells: int = DEFAULT_CONTEXT_MAP_RADIUS_CELLS,
        side: int = DEFAULT_CONTEXT_MAP_SIDE,
    ) -> FloatArray:
        """Return a coarse risk overview centered on the player.

        The default 21x21 grid covers +/-50 map cells. Each output value is
        the highest-risk state found inside its roughly 5x5 source-cell bin.
        This gives the policy minimap-scale awareness without expanding the
        observation by the 10,201 values required by a raw 101x101 crop.
        """

        radius_cells = int(radius_cells)
        side = int(side)
        if radius_cells < 1:
            raise ValueError("context crop radius must be positive")
        if side < 3 or side % 2 == 0:
            raise ValueError("context crop side must be an odd integer >= 3")
        result = np.zeros(side * side, dtype=np.float32)
        if center is None:
            return result

        offsets = np.arange(-radius_cells, radius_cells + 1, dtype=np.int32)
        groups = tuple(np.array_split(offsets, side))
        forbidden_distance_field = (
            self._get_forbidden_distance_field() if self._has_forbidden else None
        )
        center_x, center_y = int(center[0]), int(center[1])
        height, width = self.shape
        output = 0
        for y_offsets in groups:
            y0 = center_y + int(y_offsets[0])
            y1 = center_y + int(y_offsets[-1]) + 1
            for x_offsets in groups:
                x0 = center_x + int(x_offsets[0])
                x1 = center_x + int(x_offsets[-1]) + 1
                clip_x0 = max(0, x0)
                clip_y0 = max(0, y0)
                clip_x1 = min(width, x1)
                clip_y1 = min(height, y1)
                partly_outside = (
                    x0 < 0 or y0 < 0 or x1 > width or y1 > height
                )
                risk = MapCellRisk.OUTSIDE_OR_UNKNOWN
                if clip_x1 > clip_x0 and clip_y1 > clip_y0:
                    forbidden = self._forbidden[clip_y0:clip_y1, clip_x0:clip_x1]
                    if np.any(forbidden):
                        risk = MapCellRisk.TELEPORT_TRIGGER
                    elif forbidden_distance_field is not None and np.any(
                        forbidden_distance_field[
                            clip_y0:clip_y1, clip_x0:clip_x1
                        ]
                        <= self._teleport_buffer_radius_cells
                    ):
                        risk = MapCellRisk.TELEPORT_BUFFER
                    else:
                        traversable = self._traversable[
                            clip_y0:clip_y1, clip_x0:clip_x1
                        ]
                        safe = self._safe_traversable[
                            clip_y0:clip_y1, clip_x0:clip_x1
                        ]
                        if np.any(~traversable):
                            risk = MapCellRisk.OBSTACLE
                        elif np.any(traversable & ~safe):
                            risk = MapCellRisk.OBSTACLE_BUFFER
                        elif partly_outside:
                            risk = MapCellRisk.OUTSIDE_OR_UNKNOWN
                        elif np.any(safe):
                            risk = MapCellRisk.SAFE
                result[output] = _risk_encoding(risk)
                output += 1
        return result

    def direct_path_state(
        self,
        start: Cell | None,
        end: Cell | None,
    ) -> DirectPathState:
        """Classify a direct segment, ignoring the actor's final cell."""

        if (
            start is None
            or end is None
            or not self.contains(start)
            or not self.contains(end)
        ):
            return DirectPathState.UNKNOWN
        unknown = False
        cells = tuple(bresenham_cells(start, end))
        for cell in cells[1:-1]:
            state = self.cell_state(cell)
            if state > 0.5:
                return DirectPathState.BLOCKED
            if abs(state) < 0.5:
                unknown = True
        return DirectPathState.UNKNOWN if unknown else DirectPathState.CLEAR

    def segment_crosses_forbidden(
        self,
        start: Cell | None,
        end: Cell | None,
    ) -> bool:
        if start is None or end is None:
            return False
        return any(self.is_forbidden(cell) for cell in bresenham_cells(start, end))

    def geodesic_distance(
        self,
        start: Cell,
        end: Cell,
        *,
        maximum_distance_cells: float | None = None,
        maximum_expansions: int | None = None,
    ) -> float:
        """Return a bounded eight-connected safe distance, cached by query."""

        limit = (
            None if maximum_distance_cells is None else float(maximum_distance_cells)
        )
        if limit is not None and (not isfinite(limit) or limit <= 0.0):
            raise ValueError("maximum_distance_cells must be finite and positive")
        expansions = (
            self._maximum_geodesic_expansions
            if maximum_expansions is None
            else int(maximum_expansions)
        )
        if expansions < 1:
            raise ValueError("maximum_expansions must be positive")

        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        # Bounded A* can exhaust its expansion budget in one direction but not
        # the other because tie ordering is directional.  Preserve that fact in
        # the cache key instead of treating every query as symmetric.
        cache_key = (start, end, limit, expansions)
        with self._lock:
            cached = self._geodesic_cache.get(cache_key)
            if cached is not None:
                self._geodesic_cache.move_to_end(cache_key)
                return cached

        result = self._search_geodesic(
            start,
            end,
            maximum_distance_cells=limit,
            maximum_expansions=expansions,
        )
        with self._lock:
            self._geodesic_cache[cache_key] = result
            self._geodesic_cache.move_to_end(cache_key)
            while len(self._geodesic_cache) > self._geodesic_cache_size:
                self._geodesic_cache.popitem(last=False)
        return result

    def geodesic_distances(
        self,
        start: Cell,
        targets: Iterable[Cell],
        *,
        maximum_distance_cells: float | None = None,
        maximum_expansions: int | None = None,
    ) -> tuple[float, ...]:
        return tuple(
            self.geodesic_distance(
                start,
                target,
                maximum_distance_cells=maximum_distance_cells,
                maximum_expansions=maximum_expansions,
            )
            for target in targets
        )

    def bounded_geodesic_field(
        self,
        start: Cell,
        *,
        maximum_distance_cells: float,
        maximum_expansions: int | None = None,
    ) -> dict[Cell, float]:
        """Return all safe geodesic distances reachable within one bounded search.

        Recorded simulation needs distances to many actors from the same player
        cell. Running one Dijkstra expansion is exactly equivalent to issuing
        many bounded point queries, but avoids repeating the same exploration
        dozens of times per frame.
        """

        limit = float(maximum_distance_cells)
        if not isfinite(limit) or limit <= 0.0:
            raise ValueError("maximum_distance_cells must be finite and positive")
        expansions = (
            self._maximum_geodesic_expansions
            if maximum_expansions is None
            else int(maximum_expansions)
        )
        if expansions < 1:
            raise ValueError("maximum_expansions must be positive")
        start = (int(start[0]), int(start[1]))
        cache_key = (start, limit, expansions)
        with self._lock:
            cached = self._geodesic_field_cache.get(cache_key)
            if cached is not None:
                self._geodesic_field_cache.move_to_end(cache_key)
                return cached

        height, width = self._safe_traversable.shape
        sx, sy = start
        if sx < 0 or sy < 0 or sx >= width or sy >= height:
            return {}
        if not self._safe_traversable[sy, sx]:
            return {}

        # Integer cell IDs and local array/shape references avoid millions of
        # tuple allocations and ``contains()`` calls in dense simulator maps.
        # The search order and corner-cut rule remain exactly the same.
        safe = self._safe_traversable
        start_id = sy * width + sx
        queue: list[tuple[float, int]] = [(0.0, start_id)]
        best_ids: dict[int, float] = {start_id: 0.0}
        expanded = 0
        while queue and expanded < expansions:
            distance, cell_id = heappop(queue)
            if distance > best_ids.get(cell_id, inf) + 1.0e-9:
                continue
            if distance > limit:
                continue
            expanded += 1
            y, x = divmod(cell_id, width)
            for dx, dy, movement_cost in _DIRECTIONS:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height or not safe[ny, nx]:
                    continue
                if (
                    dx
                    and dy
                    and (
                        not safe[y, nx]
                        or not safe[ny, x]
                    )
                ):
                    continue
                candidate = distance + movement_cost
                neighbor_id = ny * width + nx
                if candidate > limit or candidate + 1.0e-9 >= best_ids.get(neighbor_id, inf):
                    continue
                best_ids[neighbor_id] = candidate
                heappush(queue, (candidate, neighbor_id))
        best = {
            (cell_id % width, cell_id // width): distance
            for cell_id, distance in best_ids.items()
        }
        with self._lock:
            self._geodesic_field_cache[cache_key] = best
            self._geodesic_field_cache.move_to_end(cache_key)
            while len(self._geodesic_field_cache) > self._geodesic_field_cache_size:
                self._geodesic_field_cache.popitem(last=False)
        return best

    def _get_forbidden_distance_field(self) -> FloatArray:
        with self._lock:
            if self._forbidden_distance_field is None:
                field = _euclidean_distance_to_true(self._forbidden)
                field.setflags(write=False)
                self._forbidden_distance_field = field
            return self._forbidden_distance_field

    def _search_geodesic(
        self,
        start: Cell,
        end: Cell,
        *,
        maximum_distance_cells: float | None,
        maximum_expansions: int,
    ) -> float:
        if not self.contains(start) or not self.contains(end):
            return inf
        sx, sy = start
        ex, ey = end
        if not self._safe_traversable[sy, sx] or not self._safe_traversable[ey, ex]:
            return inf
        if start == end:
            return 0.0
        initial_heuristic = hypot(ex - sx, ey - sy)
        if (
            maximum_distance_cells is not None
            and initial_heuristic > maximum_distance_cells
        ):
            return inf

        queue: list[tuple[float, float, int, int]] = [(initial_heuristic, 0.0, sx, sy)]
        best: dict[Cell, float] = {start: 0.0}
        expanded = 0
        while queue and expanded < maximum_expansions:
            _estimated, distance, x, y = heappop(queue)
            if distance > best.get((x, y), inf) + 1.0e-9:
                continue
            expanded += 1
            if (x, y) == end:
                return float(distance)

            for dx, dy, movement_cost in _DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not self.contains((nx, ny)) or not self._safe_traversable[ny, nx]:
                    continue
                if (
                    dx
                    and dy
                    and (
                        not self._safe_traversable[y, nx]
                        or not self._safe_traversable[ny, x]
                    )
                ):
                    continue
                candidate = distance + movement_cost
                if (
                    maximum_distance_cells is not None
                    and candidate > maximum_distance_cells
                ):
                    continue
                if candidate + 1.0e-9 >= best.get((nx, ny), inf):
                    continue
                heuristic = hypot(ex - nx, ey - ny)
                if (
                    maximum_distance_cells is not None
                    and candidate + heuristic > maximum_distance_cells
                ):
                    continue
                best[(nx, ny)] = candidate
                heappush(
                    queue,
                    (candidate + heuristic, candidate, nx, ny),
                )
        return inf


def bresenham_cells(start: Cell, end: Cell) -> Iterable[Cell]:
    """Yield every integer cell on a segment, including both endpoints."""

    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _readonly_bool_array(value: np.ndarray, *, name: str) -> BoolArray:
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    contiguous = np.ascontiguousarray(array, dtype=np.bool_)
    # An immutable bytes owner prevents callers from walking the ``base`` chain
    # and re-enabling writes on the storage held by this object.
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _readonly_view(array: BoolArray) -> BoolArray:
    """Return a view whose write flag cannot be re-enabled by a caller."""

    result = array.view()
    result.setflags(write=False)
    return result


def _euclidean_distance_to_true(mask: BoolArray) -> FloatArray:
    initial = np.where(mask, 0.0, np.inf)
    horizontal = np.empty(initial.shape, dtype=np.float64)
    for row_index in range(initial.shape[0]):
        horizontal[row_index, :] = _squared_distance_transform_1d(initial[row_index, :])

    squared = np.empty(initial.shape, dtype=np.float64)
    for column_index in range(initial.shape[1]):
        squared[:, column_index] = _squared_distance_transform_1d(
            horizontal[:, column_index]
        )
    return np.ascontiguousarray(np.sqrt(squared), dtype=np.float32)


def _squared_distance_transform_1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    count = int(values.size)
    result = np.full(count, np.inf, dtype=np.float64)
    sites = np.flatnonzero(np.isfinite(values))
    if sites.size == 0:
        return result

    vertices = np.empty(sites.size, dtype=np.int64)
    boundaries = np.empty(sites.size + 1, dtype=np.float64)
    envelope_size = 0
    vertices[0] = int(sites[0])
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for raw_site in sites[1:]:
        site = int(raw_site)
        while True:
            vertex = int(vertices[envelope_size])
            intersection = (
                (float(values[site]) + site * site)
                - (float(values[vertex]) + vertex * vertex)
            ) / (2.0 * site - 2.0 * vertex)
            if intersection > boundaries[envelope_size] or envelope_size == 0:
                break
            envelope_size -= 1
        envelope_size += 1
        vertices[envelope_size] = site
        boundaries[envelope_size] = intersection
        boundaries[envelope_size + 1] = np.inf

    envelope_index = 0
    for position in range(count):
        while boundaries[envelope_index + 1] < position:
            envelope_index += 1
        vertex = int(vertices[envelope_index])
        result[position] = (position - vertex) ** 2 + float(values[vertex])
    return result
