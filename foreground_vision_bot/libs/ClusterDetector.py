from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import hypot

Point = tuple[int, int]


@dataclass(frozen=True)
class MobCluster:
    center: Point
    size: int
    points: tuple[Point, ...]
    average_distance: float


class ClusterDetector:
    """
    Groups nearby mob positions without requiring external ML libraries.

    The algorithm connects points that are within `max_distance` pixels of
    one another and returns each connected group as a cluster.
    """

    def __init__(
        self,
        max_distance: float = 140.0,
        min_cluster_size: int = 1,
    ) -> None:
        if max_distance <= 0:
            raise ValueError("max_distance must be greater than zero")

        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least one")

        self.max_distance = float(max_distance)
        self.min_cluster_size = int(min_cluster_size)

    def detect(
        self,
        points: Iterable[Sequence[int | float]],
    ) -> list[MobCluster]:
        normalized_points = self._normalize_points(points)

        if not normalized_points:
            return []

        unvisited = set(range(len(normalized_points)))
        clusters: list[MobCluster] = []

        while unvisited:
            start_index = unvisited.pop()
            component = [start_index]
            queue = [start_index]

            while queue:
                current_index = queue.pop()
                current_point = normalized_points[current_index]

                neighbors = [
                    other_index
                    for other_index in tuple(unvisited)
                    if self._distance(
                        current_point,
                        normalized_points[other_index],
                    )
                    <= self.max_distance
                ]

                for neighbor_index in neighbors:
                    unvisited.remove(neighbor_index)
                    queue.append(neighbor_index)
                    component.append(neighbor_index)

            component_points = tuple(normalized_points[index] for index in component)

            if len(component_points) >= self.min_cluster_size:
                clusters.append(self._build_cluster(component_points))

        clusters.sort(
            key=lambda cluster: (
                -cluster.size,
                cluster.average_distance,
                cluster.center[0],
                cluster.center[1],
            )
        )

        return clusters

    def largest(
        self,
        points: Iterable[Sequence[int | float]],
    ) -> MobCluster | None:
        clusters = self.detect(points)
        return clusters[0] if clusters else None

    @staticmethod
    def _normalize_points(
        points: Iterable[Sequence[int | float]],
    ) -> list[Point]:
        normalized: list[Point] = []

        for point in points:
            if len(point) != 2:
                raise ValueError(f"Expected a two-value point, got: {point!r}")

            normalized.append((round(point[0]), round(point[1])))

        return normalized

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return hypot(
            first[0] - second[0],
            first[1] - second[1],
        )

    @classmethod
    def _build_cluster(
        cls,
        points: tuple[Point, ...],
    ) -> MobCluster:
        center_x = round(sum(point[0] for point in points) / len(points))
        center_y = round(sum(point[1] for point in points) / len(points))
        center = (center_x, center_y)

        average_distance = sum(cls._distance(point, center) for point in points) / len(
            points
        )

        return MobCluster(
            center=center,
            size=len(points),
            points=tuple(sorted(points)),
            average_distance=average_distance,
        )
