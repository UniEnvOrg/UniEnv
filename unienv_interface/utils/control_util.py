from fractions import Fraction
from math import gcd
from functools import reduce
from typing import Iterable, Tuple, List, Union

Number = Union[int, float]

def _lcm(a: int, b: int) -> int:
    return abs(a // gcd(a, b) * b) if a and b else 0

def _lcmm(values: Iterable[int]) -> int:
    return reduce(_lcm, values, 1)

def _gcdm(values: Iterable[int]) -> int:
    return reduce(gcd, values)

def find_best_timestep(
    timesteps: Iterable[Number],
    *,
    max_denominator: int = 10_000,
    return_fraction: bool = False,
) -> Tuple[Union[float, Fraction], List[int]]:
    """
    Compute the simulation timestep dt such that every sensor period is an
    integer multiple of dt (i.e., dt is the GCD of the periods). Works with floats.

    Args:
        timesteps: Iterable of sensor periods (seconds, ms, etc.). Must be > 0.
        max_denominator: Max denominator when rational-approximating floats.
                         Increase if your periods are very fine-grained.
        return_fraction:  If True, returns dt as a Fraction; otherwise a float.

    Returns:
        dt: The best simulation timestep (float or Fraction).
        steps_per_sensor: For each input period T_i, the integer k_i = T_i / dt.

    Raises:
        ValueError: If list is empty or contains non-positive values.
    """
    # Validate and convert to Fractions
    periods = list(timesteps)
    if not periods:
        raise ValueError("timesteps must be a non-empty sequence.")
    if any(p <= 0 for p in periods):
        raise ValueError("All timesteps must be positive.")

    fracs = [Fraction(p).limit_denominator(max_denominator) for p in periods]

    # Find common denominator (LCM of all denominators)
    D = _lcmm([f.denominator for f in fracs])

    # Scale each fraction to that denominator → integers
    ints = [f.numerator * (D // f.denominator) for f in fracs]

    # GCD of the integerized periods → integer g; convert back via g/D
    g = _gcdm(ints)
    dt_frac = Fraction(g, D)  # exact rational dt

    # Sanity: dt must be > 0
    if dt_frac <= 0:
        raise RuntimeError("Computed non-positive timestep; check inputs.")

    steps_per_sensor = [int(f // dt_frac) for f in fracs]  # exact integer division

    if return_fraction:
        return dt_frac, steps_per_sensor
    else:
        return float(dt_frac), steps_per_sensor