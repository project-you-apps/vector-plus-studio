---
title: Telescope
tags: [concept, instrument, optics]
aliases: [optical telescope, refracting telescope]
---

# Telescope

An optical instrument that uses lenses (or mirrors, in later designs) to
gather and focus light, producing a magnified image of distant objects.
The 17th-century introduction of the telescope was the single most
important empirical development in the shift from [[Geocentrism]] to
[[Heliocentrism]].

## Two early designs

### Galilean telescope

[[Galileo Galilei]]'s improved design (1609) used a convex objective and
a concave eyepiece, producing an upright image with ~30x magnification at
its best. Limited field of view and chromatic aberration. Used for the
first telescopic observations of:

- Jovian moons (1610)
- Phases of Venus (1610)
- Sunspots (1611)
- Lunar mountains (1610)

### Keplerian telescope

[[Johannes Kepler]]'s design described in *Dioptrice* (1611) used two
convex lenses, producing an inverted image but with significantly greater
magnification and a wider field of view. The inversion is acceptable for
astronomy (orientation of stars is arbitrary) but inconvenient for
terrestrial use, leading to terrestrial telescopes adding inverter prisms.
Modern astronomical telescopes are descended from the Keplerian design.

## Why it broke geocentrism

Pre-telescope astronomy had no way to distinguish [[Geocentrism]] from
[[Heliocentrism]] empirically — both models could be tuned (epicycles
for the geocentric, eccentrics for the heliocentric) to predict the
naked-eye motions adequately. The telescope changed this by revealing
phenomena that demanded specific orbital geometries:

- Jupiter's moons → not everything orbits Earth
- Phases of Venus → Venus orbits the Sun, not Earth
- Sunspots → the Sun is not a perfect sphere; it rotates

```python
# A toy estimate of how much resolution the telescope adds
# Naked eye: ~1 arcminute resolution (Tycho Brahe's instruments hit this limit)
# Galileo's 30x telescope: roughly 1/30 arcmin = 2 arcseconds
# Modern amateur 8" telescope: ~0.6 arcsec (diffraction limit)
naked_eye_arcsec = 60.0
galileo_arcsec = naked_eye_arcsec / 30
modern_amateur_arcsec = 0.6
print(f"Galileo gain over naked eye: {naked_eye_arcsec / galileo_arcsec:.0f}x")
```

## Connections

- [[Tycho Brahe]] achieved naked-eye precision approaching the resolution
  limit of the human eye, but was the last great astronomer to work
  without a telescope. Once Galileo demonstrated what telescopic
  observation could reveal, all serious work switched to instruments.

#concept #instrument #optics #1610
