# Johannes Kepler

Born: December 27, 1571
Died: November 15, 1630
Tags: people, german, astronomer, mathematician
Aliases: Kepler

German astronomer and mathematician. Best known for his three laws of
planetary motion, which mathematized the heliocentric model and provided
the empirical bridge between Tycho Brahe's observational data and
Newton's later gravitational theory.

## The three laws

1. **Law of orbits** — planets move in elliptical orbits with the Sun at one focus
2. **Law of areas** — a line connecting a planet to the Sun sweeps out equal areas in equal times
3. **Law of periods** — the square of the orbital period is proportional to the cube of the semi-major axis

These laws replaced the perfectly-circular orbits assumed by both
geocentrism and earlier Copernican models.

```python
import math
def period_from_axis(a_au: float) -> float:
    """Returns orbital period in years given semi-major axis in AU."""
    return math.sqrt(a_au ** 3)

print(period_from_axis(1.0))   # Earth: 1.0 year
print(period_from_axis(5.2))   # Jupiter: ~11.86 years
```

## Optics

Kepler's *Dioptrice* (1611) described the Keplerian telescope with two
convex lenses. Modern astronomical telescopes use Kepler's design, not
Galileo's.
