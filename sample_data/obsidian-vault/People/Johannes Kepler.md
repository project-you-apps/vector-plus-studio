---
title: Johannes Kepler
born: 1571-12-27
died: 1630-11-15
tags: [people, german, astronomer, mathematician]
aliases: [Kepler]
---

# Johannes Kepler

German astronomer and mathematician. Best known for his three laws of
planetary motion, which mathematized the [[Heliocentrism|heliocentric]]
model and provided the empirical bridge between [[Tycho Brahe]]'s
observational data and Newton's later gravitational theory.

## The three laws

Kepler's laws govern the motion of planets around the Sun:

1. **Law of orbits** — planets move in elliptical orbits with the Sun at one focus
2. **Law of areas** — a line connecting a planet to the Sun sweeps out equal areas in equal times
3. **Law of periods** — the square of a planet's orbital period is proportional to the cube of its semi-major axis

These laws replaced the perfectly-circular orbits assumed by both
[[Geocentrism]] and earlier [[Heliocentrism|Copernican]] models. The
ellipse-not-circle insight came directly from analyzing [[Tycho Brahe]]'s
observations of Mars.

```python
# Kepler's third law in modern form
# T² ∝ a³  (T = orbital period, a = semi-major axis)
import math
def period_from_axis(a_au: float) -> float:
    """Returns orbital period in years given semi-major axis in AU."""
    return math.sqrt(a_au ** 3)

print(period_from_axis(1.0))   # Earth: 1.0 year
print(period_from_axis(5.2))   # Jupiter: ~11.86 years
```

## Optics and the telescope

Kepler also made foundational contributions to optics. His *Dioptrice* (1611)
described the [[Telescope|Keplerian telescope]] design with two convex
lenses, producing an inverted image but with greater magnification than
[[Galileo Galilei|Galileo]]'s Galilean design. Modern astronomical
telescopes use Kepler's design, not Galileo's.

## Mysticism

Kepler combined rigorous mathematics with mystical convictions. *Mysterium
Cosmographicum* (1596) attempted to explain planetary spacing via nested
Platonic solids — an idea that turned out to be wrong but that motivated
his later, correct, mathematical work. #mysticism #early-work

## Connections

- [[Tycho Brahe]] hired Kepler in 1600; Kepler inherited Tycho's data after
  his death and used it to derive the laws
- [[Galileo Galilei]] corresponded with Kepler about telescope observations,
  though they disagreed about ocean tides

#people #german #astronomer #mathematician
