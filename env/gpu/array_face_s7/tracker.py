"""S7 mission tracker — unchanged from S6 (5-tuple missions with bearings)."""
from env.gpu.array_face_s6.tracker import S6MissionTracker

# S7 reuses the S6 tracker verbatim: per-(svc, az) credit, same accounting.
S7MissionTracker = S6MissionTracker

__all__ = ["S7MissionTracker"]
