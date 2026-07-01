def constraint_satisfaction_rate(violations):
    return 1.0 - (sum(violations) / max(1, len(violations)))
