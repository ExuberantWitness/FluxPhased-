"""2D battlefield engine: vehicle kinematics, array rotation, artillery.

Models:
- 4 mobile radar vehicles with 2D dynamics (30 km/h max)
- Each vehicle carries a 360-degree rotatable phased array
- 2 artillery pieces with 30s shell flight delay and 350-360m kill radius
- Open terrain (no occlusion, ground clutter only)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from ..config import VehicleConfig, ArtilleryConfig, BattlefieldConfig, CPIConfig


DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


@dataclass
class VehicleState:
    """State of a single radar vehicle."""
    x: float
    y: float
    heading: float          # degrees, 0=East, 90=North
    speed: float            # m/s
    array_rotation: float   # degrees, array boresight relative to vehicle heading
    alive: bool = True

    @property
    def array_bearing(self) -> float:
        """Absolute array bearing in global frame (degrees)."""
        return (self.heading + self.array_rotation) % 360.0

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        heading_rad = self.heading * DEG2RAD
        return (self.speed * np.cos(heading_rad), self.speed * np.sin(heading_rad))


@dataclass
class Shell:
    """An artillery shell in flight."""
    launch_time: float
    impact_time: float
    target_x: float
    target_y: float
    impact_x: float  # with CEP scatter
    impact_y: float
    team: str  # "red" or "blue"

    def time_to_impact(self, current_time: float) -> float:
        return max(0.0, self.impact_time - current_time)


@dataclass
class ArtilleryState:
    """State of an artillery piece."""
    team: str
    x: float
    y: float
    cooldown_remaining: float = 0.0
    shells_in_flight: List[Shell] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.cooldown_remaining <= 0.0


class Battlefield:
    """2D battlefield with vehicle dynamics, array rotation, and artillery."""

    def __init__(
        self,
        vehicle_cfg: VehicleConfig,
        artillery_cfg: ArtilleryConfig,
        battlefield_cfg: BattlefieldConfig,
        cpi_cfg: CPIConfig,
        seed: int = 42,
    ):
        self.vehicle_cfg = vehicle_cfg
        self.artillery_cfg = artillery_cfg
        self.battlefield_cfg = battlefield_cfg
        self.cpi_cfg = cpi_cfg
        self._rng = np.random.default_rng(seed)

        self.dt = cpi_cfg.cpi_duration  # 50ms per step
        self.current_time: float = 0.0

        # Initialize vehicles
        self.vehicles: dict[str, VehicleState] = {}
        self._init_vehicles()

        # Initialize artillery
        self.artillery: dict[str, ArtilleryState] = {}
        self._init_artillery()

        # Map boundaries
        self.map_x_min = battlefield_cfg.boundary_margin
        self.map_x_max = battlefield_cfg.map_size[0] - battlefield_cfg.boundary_margin
        self.map_y_min = battlefield_cfg.boundary_margin
        self.map_y_max = battlefield_cfg.map_size[1] - battlefield_cfg.boundary_margin

    def _init_vehicles(self):
        cfg = self.battlefield_cfg
        for i, (pos, heading) in enumerate(zip(cfg.red_positions[:2], cfg.red_headings)):
            self.vehicles[f"red_radar_{i}"] = VehicleState(
                x=pos[0], y=pos[1], heading=heading,
                speed=0.0, array_rotation=0.0
            )
        for i, (pos, heading) in enumerate(zip(cfg.blue_positions[:2], cfg.blue_headings)):
            self.vehicles[f"blue_radar_{i}"] = VehicleState(
                x=pos[0], y=pos[1], heading=heading,
                speed=0.0, array_rotation=0.0
            )

    def _init_artillery(self):
        cfg = self.battlefield_cfg
        self.artillery["red"] = ArtilleryState(
            team="red",
            x=cfg.red_positions[2][0],
            y=cfg.red_positions[2][1],
        )
        self.artillery["blue"] = ArtilleryState(
            team="blue",
            x=cfg.blue_positions[2][0],
            y=cfg.blue_positions[2][1],
        )

    def get_vehicle(self, agent_id: str) -> VehicleState:
        return self.vehicles[agent_id]

    def get_team_vehicles(self, team: str) -> List[Tuple[str, VehicleState]]:
        prefix = f"{team}_radar_"
        return [(aid, v) for aid, v in self.vehicles.items() if aid.startswith(prefix)]

    def get_enemy_vehicles(self, team: str) -> List[Tuple[str, VehicleState]]:
        enemy = "blue" if team == "red" else "red"
        return self.get_team_vehicles(enemy)

    def update_vehicle(
        self,
        agent_id: str,
        speed: float,
        heading_change: float,
        array_rotation: float,
    ):
        """Apply vehicle control actions for one CPI.

        Args:
            agent_id: Vehicle identifier
            speed: Target speed (m/s), [-max_speed, max_speed]
            heading_change: Heading change rate (deg/s)
            array_rotation: Desired array rotation angle (deg)
        """
        v = self.vehicles[agent_id]
        if not v.alive:
            return

        # Clamp speed
        speed = np.clip(speed, 0.0, self.vehicle_cfg.max_speed)
        # Smooth speed changes (acceleration limited)
        max_dv = self.vehicle_cfg.max_accel * self.dt
        v.speed = np.clip(speed, v.speed - max_dv, v.speed + max_dv)

        # Update heading
        max_dh = self.vehicle_cfg.max_rotation_speed * self.dt
        dh = np.clip(heading_change * self.dt, -max_dh, max_dh)
        v.heading = (v.heading + dh) % 360.0

        # Update position
        v.x += v.speed * np.cos(v.heading * DEG2RAD) * self.dt
        v.y += v.speed * np.sin(v.heading * DEG2RAD) * self.dt

        # Boundary clamp
        v.x = np.clip(v.x, self.map_x_min, self.map_x_max)
        v.y = np.clip(v.y, self.map_y_min, self.map_y_max)

        # Array rotation
        max_array_dr = self.vehicle_cfg.max_rotation_speed * self.dt
        current_az = v.array_rotation
        target_az = array_rotation
        # Shortest rotation path
        diff = ((target_az - current_az + 540) % 360) - 180
        dr = np.clip(diff, -max_array_dr, max_array_dr)
        v.array_rotation = (current_az + dr) % 360.0

    def fire_artillery(self, team: str, target_x: float, target_y: float) -> bool:
        """Fire artillery at target coordinates.

        Returns True if shell was launched, False if on cooldown.
        """
        arty = self.artillery[team]
        if not arty.ready:
            return False

        # Add CEP scatter
        impact_x = target_x + self._rng.normal(0, self.artillery_cfg.cep / 2.0)
        impact_y = target_y + self._rng.normal(0, self.artillery_cfg.cep / 2.0)

        shell = Shell(
            launch_time=self.current_time,
            impact_time=self.current_time + self.artillery_cfg.flight_time,
            target_x=target_x,
            target_y=target_y,
            impact_x=impact_x,
            impact_y=impact_y,
            team=team,
        )
        arty.shells_in_flight.append(shell)
        arty.cooldown_remaining = self.artillery_cfg.cooldown
        return True

    def check_shell_impacts(self) -> List[Tuple[str, str]]:
        """Check for shell impacts at current time.

        Returns list of (killed_agent_id, killer_team).
        """
        kills = []
        for team, arty in self.artillery.items():
            for shell in list(arty.shells_in_flight):
                if self.current_time >= shell.impact_time:
                    # Check kill radius against enemy vehicles
                    enemy_team = "blue" if team == "red" else "red"
                    for agent_id, v in self.get_team_vehicles(enemy_team):
                        if not v.alive:
                            continue
                        dist = np.sqrt(
                            (v.x - shell.impact_x) ** 2 + (v.y - shell.impact_y) ** 2
                        )
                        if dist <= self.artillery_cfg.kill_radius_max:
                            # Deterministic kill within max radius for simplicity
                            # (could add probabilistic kill between min and max)
                            v.alive = False
                            kills.append((agent_id, team))

                    arty.shells_in_flight.remove(shell)

        return kills

    def check_team_alive(self, team: str) -> bool:
        """Check if a team still has ALL radar vehicles alive.

        Per rules: '一方两部雷达任意一部被摧毁，另一方获胜'
        So team is alive only if ALL its radars are alive.
        """
        for _, v in self.get_team_vehicles(team):
            if not v.alive:
                return False
        return True

    def check_game_over(self) -> Optional[str]:
        """Check if game is over. Returns winning team or None."""
        red_alive = self.check_team_alive("red")
        blue_alive = self.check_team_alive("blue")

        if not red_alive and not blue_alive:
            return "draw"
        elif not red_alive:
            return "blue"
        elif not blue_alive:
            return "red"
        return None

    def step_time(self):
        """Advance simulation by one CPI."""
        self.current_time += self.dt

        # Update artillery cooldowns
        for arty in self.artillery.values():
            arty.cooldown_remaining = max(0.0, arty.cooldown_remaining - self.dt)

    def get_distance(self, agent_a: str, agent_b: str) -> float:
        """Distance between two vehicles (m)."""
        va = self.vehicles[agent_a]
        vb = self.vehicles[agent_b]
        return np.sqrt((va.x - vb.x) ** 2 + (va.y - vb.y) ** 2)

    def get_bearing(self, from_agent: str, to_agent: str) -> float:
        """Bearing from one vehicle to another (degrees, global frame)."""
        va = self.vehicles[from_agent]
        vb = self.vehicles[to_agent]
        return np.degrees(np.arctan2(vb.y - va.y, vb.x - va.x))

    def get_relative_angle(self, from_agent: str, to_agent: str) -> float:
        """Angle to another vehicle relative to from_agent's array boresight.

        Returns angle in [-180, 180] degrees.
        """
        va = self.vehicles[from_agent]
        bearing = self.get_bearing(from_agent, to_agent)
        rel = bearing - va.array_bearing
        return ((rel + 180) % 360) - 180

    def get_state_snapshot(self) -> dict:
        """Get full battlefield state for observation building."""
        return {
            "time": self.current_time,
            "vehicles": {
                aid: {
                    "x": v.x,
                    "y": v.y,
                    "heading": v.heading,
                    "speed": v.speed,
                    "array_rotation": v.array_rotation,
                    "array_bearing": v.array_bearing,
                    "alive": v.alive,
                }
                for aid, v in self.vehicles.items()
            },
            "artillery": {
                team: {
                    "ready": a.ready,
                    "cooldown": a.cooldown_remaining,
                    "shells": len(a.shells_in_flight),
                }
                for team, a in self.artillery.items()
            },
        }
