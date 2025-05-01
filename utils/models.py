import numpy as np

def grad_rate(absenteeism):
    """Predict graduation rate based on absenteeism."""
    return 95 - 0.8 * (absenteeism - 5) ** 1.1

def srs_from_literacy(lit):
    """Compute Success Risk Score based on literacy rate."""
    return 100 - 0.8 * (lit - 70)

def juvenile_risk(programs):
    """Estimate juvenile system exposure based on youth program access."""
    return 40 - 10 * np.log1p(programs)

def college_readiness(internet):
    """Estimate college readiness based on broadband access."""
    return 0.7 * internet - 20
