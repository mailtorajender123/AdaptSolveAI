import numpy as np

class SchedulingSafetySupervisor:
    """Rule/CBF-inspired safety filter for online scheduling.

    It minimally modifies unsafe machine assignments by replacing them with the
    feasible machine with the smallest projected load.
    """
    def __init__(self, fallback='least_loaded'):
        self.fallback = fallback

    def filter_action(self, env, proposed_action: int):
        feasible = env.feasible_actions()
        proposed_action = int(proposed_action)
        if feasible[proposed_action]:
            return proposed_action, {'modified': 0, 'safety_violation': 0}
        if self.fallback == 'least_loaded':
            projected = env.machine_remaining.copy()
            if env.queue:
                projected += env.queue[0].processing_time
            feasible_idx = np.where(feasible)[0]
            chosen = int(feasible_idx[np.argmin(projected[feasible_idx])])
        else:
            chosen = int(np.where(feasible)[0][0])
        return chosen, {'modified': 1, 'safety_violation': 1}

    def margin(self, env):
        if not env.queue:
            return 1.0
        job = env.queue[0]
        slack = job.deadline - env.t
        projected = np.min(env.machine_remaining + job.processing_time)
        return float((slack - projected) / max(1.0, slack))
