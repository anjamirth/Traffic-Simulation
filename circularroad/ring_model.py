# ring_model.py
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from Cars import Cars

@dataclass
class StepStats:
    t: int
    slow_cars: int
    jam_clusters: int

class RingRoadModel:
    """
    Discrete ring road (L cells) with Nagel–Schreckenberg-style rules.
    Cars live in discrete cells; movement is parallel (synchronous update).
    Human cars randomise with probability p.
    """

    def __init__(self, L: int, num_cars: int, v_max: int, p: float, alpha: float, seed: int = 1):
        self.L = L
        self.num_cars = num_cars
        self.v_max = v_max
        self.p = p
        self.alpha = alpha

        random.seed(seed)

        self.cars: List[Cars] = []
        self.lane: List[Optional[Cars]] = [None] * L

        # evenly spaced initial positions
        spacing = L // num_cars
        positions = [(i * spacing) % L for i in range(num_cars)]

        k_auto = int(round(alpha * num_cars))  # first k_auto are "autonomous" (not human)
        for i, pos in enumerate(positions):
            is_human = (i >= k_auto)
            c = Cars(v_max=v_max, current_edge=(0, 0), human=is_human, lane=0, pos=pos)
            self.cars.append(c)
            self.lane[pos] = c

        self.t = 0

    def _gap_ahead(self, pos: int, lane_snapshot: List[Optional[Cars]]) -> int:
        """How many empty cells ahead until next car (0..L-1)."""
        for d in range(1, self.L):
            j = (pos + d) % self.L
            if lane_snapshot[j] is not None:
                return d - 1
        return self.L - 1

    def jam_clusters(self, threshold: int = 1) -> int:
        jam = [0] * self.L
        for c in self.cars:
            if c.speed <= threshold:
                jam[c.pos] = 1
        if sum(jam) == 0:
            return 0
        clusters = 0
        for i in range(self.L):
            if jam[i] == 1 and jam[i - 1] == 0:
                clusters += 1
        return clusters

    def slow_cars(self, threshold: int = 1) -> int:
        return sum(1 for c in self.cars if c.speed <= threshold)

    def step(self, jam_threshold: int = 1) -> StepStats:
        """Advance simulation by one timestep (parallel update)."""

        # snapshot occupancy at start of step
        lane_old = [None] * self.L
        for c in self.cars:
            lane_old[c.pos] = c

        # Phase 1: update speeds based on lane_old
        for c in self.cars:
            c.accelarate()
            g = self._gap_ahead(c.pos, lane_old)
            c.brake(g)
            if c.isHuman:
                c.randomise(self.p)

        # Phase 2: compute new positions simultaneously
        new_positions = [(c.pos + c.speed) % self.L for c in self.cars]

        # collision check (should never happen in a correct CA)
        if len(new_positions) != len(set(new_positions)):
            raise AssertionError("Collision: two cars ended up in the same cell")

        # apply movement and rebuild lane
        self.lane = [None] * self.L
        for c, new_pos in zip(self.cars, new_positions):
            c.pos = new_pos
            self.lane[c.pos] = c

        stats = StepStats(
            t=self.t,
            slow_cars=self.slow_cars(threshold=jam_threshold),
            jam_clusters=self.jam_clusters(threshold=jam_threshold),
        )
        self.t += 1
        return stats

    def get_positions_speeds(self) -> List[Tuple[int, int, bool]]:
        """Return (pos, speed, isHuman) for each car."""
        return [(c.pos, c.speed, c.isHuman) for c in self.cars]
