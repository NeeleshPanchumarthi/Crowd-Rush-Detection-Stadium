"""Choose sparse (YOLO + tracks) or dense (DM-Count) counting, with a hysteresis band."""

from __future__ import annotations

SPARSE, DENSE = "sparse", "dense"


def select_mode(count: float, density: float, current: str = SPARSE,
                switch_density: float = 2.0, hysteresis: float = 0.5,
                first_count: float | None = None, detector_count: float | None = None,
                time_in_mode: float = 1e9, min_dwell_s: float = 0.0) -> str:
    """Decide which counter runs next.

    sparse -> dense when the density (people/m^2) reaches switch + band, or when the cheap DM-Count
                    first count clearly exceeds what YOLO finds (the detector is losing people in
                    groups, which happens before the density limit on far or occluded views).
    dense -> sparse when the density falls to switch - band and we have been dense for min_dwell_s,
                    so the mode does not flip back and forth every frame.
    `count` is kept for the plan's signature and for logging.
    """
    if current == SPARSE:
        if density >= switch_density + hysteresis:
            return DENSE
        if first_count is not None and detector_count is not None:
            if first_count >= 8 and first_count >= 2.0 * detector_count:
                return DENSE
        return SPARSE
    low = switch_density - hysteresis
    # A few people only, or clearly below the band: YOLO can cope again. (The count check stops
    # sparse/dense flipping when we went dense because of the detector mismatch rule.)
    if time_in_mode >= min_dwell_s and density <= low and (count < 8 or density <= low / 2):
        return SPARSE
    return DENSE
