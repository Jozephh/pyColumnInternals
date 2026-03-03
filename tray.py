"""
Towler & Sinnott (3e) – Sieve tray hydraulic design procedure (Section 17.13)
Single-file implementation: loads K1/K2/Entrainment J charts from CSVs in the SAME FOLDER as this script.

Expected CSVs in the same directory:
  K1_s0p15.csv, K1_s0p25.csv, K1_s0p30.csv, K1_s0p45.csv, K1_s0p60.csv, K1_s0p90.csv
    columns: FLV, K1

  K2.csv
    columns: hclear_mm, K2

  J_pf30.csv, J_pf35.csv, J_pf40.csv, J_pf45.csv, J_pf50.csv, J_pf60.csv, J_pf70.csv, J_pf80.csv, J_pf90.csv, J_pf95.csv
    columns: FLV, J

What this script does:
- Implements the Towler & Sinnott iterative procedure (Steps 1–14) for sieve-tray hydraulics.
- Uses your digitised K1(FLV, spacing), K2(hclear), and entrainment J(FLV, %flood) from CSVs.
- Performs a simple grid-search optimisation (Step 13) to pick a feasible low-diameter design.

Notes / TODOs:
- C0 is left as a constant (0.82). If you digitise Fig 17.42, you can replace C0_const() with a lookup.
- Flow arrangement (single/reverse/double pass) selection is a rule-stub unless you set inp.flow_arrangement.
- “Acceptable limits” (ΔP limit, J limit, backup criterion) are exposed as inputs.
"""

from __future__ import annotations

import os
import re
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# -------------------------
# Constants
# -------------------------
g = 9.81


# -------------------------
# CSV curve utilities
# -------------------------
def _read_csv(path: str) -> Tuple[List[str], List[dict]]:
    # encoding="utf-8-sig" strips the BOM if present
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        rows = [row for row in r]
    return cols, rows


def _to_float_arrays(rows: List[dict], xcol: str, ycol: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        xv, yv = row.get(xcol), row.get(ycol)
        if xv in (None, "") or yv in (None, ""):
            continue
        xs.append(float(xv))
        ys.append(float(yv))
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        raise ValueError("Curve has too few points.")
    idx = np.argsort(xs)
    return xs[idx], ys[idx]


def _interp_clamped(xq: float, xs: np.ndarray, ys: np.ndarray) -> float:
    if xq <= xs[0]:
        return float(ys[0])
    if xq >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(xq, xs, ys))


@dataclass
class Curve1D:
    xs: np.ndarray
    ys: np.ndarray

    @classmethod
    def from_csv(cls, path: str, xcol: str, ycol: str) -> "Curve1D":
        cols, rows = _read_csv(path)
        if xcol not in cols or ycol not in cols:
            raise ValueError(f"{os.path.basename(path)} missing {xcol}/{ycol}. Has: {cols}")
        xs, ys = _to_float_arrays(rows, xcol, ycol)
        return cls(xs, ys)

    def __call__(self, x: float) -> float:
        return _interp_clamped(x, self.xs, self.ys)


class K1Family:
    """K1(FLV, tray_spacing_m) from digitised K1_s0pXX.csv files."""

    def __init__(self, spacing_to_curve: Dict[float, Curve1D]):
        if not spacing_to_curve:
            raise ValueError("No K1 curves loaded.")
        self.spacing_to_curve = dict(sorted(spacing_to_curve.items(), key=lambda kv: kv[0]))
        self.spacings = np.asarray(list(self.spacing_to_curve.keys()), dtype=float)

    def __call__(self, flv: float, tray_spacing_m: float) -> float:
        s = float(tray_spacing_m)

        if s <= self.spacings[0]:
            return self.spacing_to_curve[float(self.spacings[0])](flv)
        if s >= self.spacings[-1]:
            return self.spacing_to_curve[float(self.spacings[-1])](flv)

        i_hi = int(np.searchsorted(self.spacings, s))
        s0 = float(self.spacings[i_hi - 1])
        s1 = float(self.spacings[i_hi])
        k0 = self.spacing_to_curve[s0](flv)
        k1 = self.spacing_to_curve[s1](flv)
        w = (s - s0) / (s1 - s0)
        return (1 - w) * k0 + w * k1


class JFamily:
    """J(FLV, %flood) from digitised J_pfXX.csv files."""

    def __init__(self, pf_to_curve: Dict[float, Curve1D]):
        if not pf_to_curve:
            raise ValueError("No J curves loaded.")
        self.pf_to_curve = dict(sorted(pf_to_curve.items(), key=lambda kv: kv[0]))
        self.pfs = np.asarray(list(self.pf_to_curve.keys()), dtype=float)

    def __call__(self, flv: float, percent_flood: float) -> float:
        pf = float(percent_flood)

        if pf <= self.pfs[0]:
            return self.pf_to_curve[float(self.pfs[0])](flv)
        if pf >= self.pfs[-1]:
            return self.pf_to_curve[float(self.pfs[-1])](flv)

        i_hi = int(np.searchsorted(self.pfs, pf))
        pf0 = float(self.pfs[i_hi - 1])
        pf1 = float(self.pfs[i_hi])
        j0 = self.pf_to_curve[pf0](flv)
        j1 = self.pf_to_curve[pf1](flv)
        w = (pf - pf0) / (pf1 - pf0)
        return (1 - w) * j0 + w * j1


@dataclass(frozen=True)
class Charts:
    K1: K1Family
    K2: Curve1D
    J: JFamily


def _parse_spacing(fname: str) -> float:
    # K1_s0p45.csv -> 0.45
    m = re.search(r"K1_s(\d+)p(\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse spacing from filename: {fname}")
    a, b = m.group(1), m.group(2)
    return float(f"{int(a)}.{int(b)}")


def _parse_pf(fname: str) -> float:
    # J_pf80.csv -> 80
    m = re.search(r"J_pf(\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse %flood from filename: {fname}")
    return float(m.group(1))


def load_charts_from_script_folder() -> Charts:
    folder = os.path.dirname(os.path.abspath(__file__))

    # K1 curves
    spacing_to_curve: Dict[float, Curve1D] = {}
    for fn in os.listdir(folder):
        if fn.startswith("K1_s") and fn.lower().endswith(".csv"):
            s = _parse_spacing(fn)
            spacing_to_curve[s] = Curve1D.from_csv(os.path.join(folder, fn), xcol="FLV", ycol="K1")

    # J curves
    pf_to_curve: Dict[float, Curve1D] = {}
    for fn in os.listdir(folder):
        if fn.startswith("J_pf") and fn.lower().endswith(".csv"):
            pf = _parse_pf(fn)
            pf_to_curve[pf] = Curve1D.from_csv(os.path.join(folder, fn), xcol="FLV", ycol="J")

    # K2 curve
    k2_path = os.path.join(folder, "K2.csv")
    if not os.path.exists(k2_path):
        raise FileNotFoundError(f"Missing K2.csv in {folder}")
    k2_curve = Curve1D.from_csv(k2_path, xcol="hclear_mm", ycol="K2")

    return Charts(K1=K1Family(spacing_to_curve), K2=k2_curve, J=JFamily(pf_to_curve))


# -------------------------
# Tray design model
# -------------------------
@dataclass(frozen=True)
class TrayHydraulicsInputs:
    # Step 1: turndown max/min flows (mass rates)
    Lw_max: float  # kg/s
    Lw_min: float  # kg/s
    Vw_max: float  # kg/s
    Vw_min: float  # kg/s

    # Step 2: properties at tray conditions
    rhoL: float  # kg/m3
    rhoV: float  # kg/m3
    sigma: float = 0.02  # N/m (used for K1 correction)

    # Step 3: tray spacing candidates (m)
    tray_spacings_m: Tuple[float, ...] = (0.15,0.25 ,0.3,0.45, 0.60, 0.90)

    # Step 4: flooding design fraction
    flood_design_frac: float = 0.8

    # Step 5: arrangement override (else rule-stub)
    flow_arrangement: Optional[str] = None  # "single","reverse","double"

    # Step 6: trial layout guesses
    downcomer_area_frac_guess: float = 0.12
    hole_area_fracs: Tuple[float, ...] = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)  # Ah/Aa search grid

    hw_mm: float = 50.0
    dh_mm: float = 5.0
    t_mm: float = 5.0
    hap_delta_mm: float = 10.0  # hap = hw - delta

    # Acceptance criteria / constraints
    weir_crest_min_mm: float = 10.0
    ht_limit_mm: float = 120.0
    downcomer_res_time_min_s: float = 3.0
    entrainment_J_limit: float = 0.10


@dataclass
class TrayDesignResult:
    ok: bool
    messages: List[str]

    tray_spacing_m: float
    hole_area_frac_active: float
    arrangement: str

    Dc_m: float
    Ac_m2: float
    Ad_m2: float
    An_m2: float
    Aa_m2: float

    lw_m: float
    Ah_m2: float
    Ap_m2: float

    percent_flood_max: float
    weeping_ok: bool
    dp_ok: bool
    downcomer_ok: bool
    entrainment_ok: bool

    ht_mm_liq: float
    dP_Pa: float
    hb_mm: float
    tr_s: float
    J: float


# -------------------------
# Towler-style equations
# -------------------------
def FLV(Lw: float, Vw: float, rhoV: float, rhoL: float) -> float:
    return (Lw / Vw) * math.sqrt(rhoV / rhoL)


def flooding_velocity(K1: float, rhoL: float, rhoV: float) -> float:
    return K1 * math.sqrt((rhoL - rhoV) / rhoV)


def weir_crest_how_mm(Lw: float, rhoL: float, lw_m: float) -> float:
    return 750.0 * Lw / ((rhoL * lw_m) ** (2.0 / 3.0))


def weep_point_uh_min_mps(K2: float, dh_mm: float, rhoV: float) -> float:
    return (K2 - 0.90 * (25.4 - dh_mm)) * math.sqrt(rhoV)


def residual_head_hr_mm(rhoL: float) -> float:
    return 12.5e3 / rhoL


def dry_tray_drop_hd_mm(uh_mps: float, C0: float, rhoV: float, rhoL: float) -> float:
    return 51.0 * (uh_mps / (C0 ** 2)) * (rhoV / rhoL)


def total_tray_drop_ht_mm(hd_mm: float, hw_mm: float, how_mm: float, hr_mm: float) -> float:
    return hd_mm + (hw_mm + how_mm) + hr_mm


def tray_dP_Pa(ht_mm: float, rhoL: float) -> float:
    return 9.81e3 * (ht_mm/1000) * rhoL


def downcomer_headloss_hdc_mm(Lwd: float, rhoL: float, Am_m2: float) -> float:
    return 166.0 * Lwd / (rhoL * (Am_m2 ** 2))


def downcomer_backup_hb_mm(hw_mm: float, how_mm: float, ht_mm: float, hdc_mm: float) -> float:
    return (hw_mm + how_mm) + ht_mm + hdc_mm


def downcomer_res_time_s(Ad_m2: float, hbc_m: float, rhoL: float, Lwd: float) -> float:
    return Ad_m2 * hbc_m * rhoL / Lwd


# -------------------------
# Design logic (Steps 3–13)
# -------------------------
def k1_area_multiplier(ah_over_aa: float) -> float:
    # Towler table correction when Ah/Aa < 0.10
    if ah_over_aa >= 0.10:
        return 1.0
    if ah_over_aa >= 0.08:
        return 0.9
    if ah_over_aa >= 0.06:
        return 0.8
    return 0.8


def select_flow_arrangement(Dc_m: float, QL_m3s_max: float, user_choice: Optional[str]) -> str:
    if user_choice:
        return user_choice.lower()
    # rule-stub (replace if you digitise the arrangement chart)
    if Dc_m >= 2.0 and QL_m3s_max > 0.01:
        return "double"
    if Dc_m >= 1.2 and QL_m3s_max > 0.005:
        return "reverse"
    return "single"


def weir_length_rule(Dc_m: float, arrangement: str) -> float:
    base = 0.77 * Dc_m
    if arrangement == "double":
        return 0.5 * base
    return base


def C0_const(t_over_d: float, Ah_over_Ap: float) -> float:
    # Placeholder: replace with digitised Fig 17.42 if you have it
    return 0.82


def initial_diameter_from_flooding(inp: TrayHydraulicsInputs, charts: Charts, tray_spacing_m: float, ah_over_aa: float):
    flv = FLV(inp.Lw_max, inp.Vw_max, inp.rhoV, inp.rhoL)
    K1_base = charts.K1(flv, tray_spacing_m)
    K1_sig = K1_base * ((inp.sigma / 0.02) ** 0.2)
    K1_corr = K1_sig * k1_area_multiplier(ah_over_aa)

    uf = flooding_velocity(K1_corr, inp.rhoL, inp.rhoV)
    u_design = inp.flood_design_frac * uf

    Qv_max = inp.Vw_max / inp.rhoV
    An_req = Qv_max / u_design

    Ac = An_req / (1.0 - inp.downcomer_area_frac_guess)
    Dc = math.sqrt(4.0 * Ac / math.pi)
    Ad = inp.downcomer_area_frac_guess * Ac
    An = Ac - Ad
    return Dc, Ac, Ad, An


def tray_areas(Ac: float, Ad: float):
    An = Ac - Ad
    Aa = Ac - 2.0 * Ad
    if Aa <= 0:
        raise ValueError("Active area <= 0; reduce downcomer fraction or increase diameter.")
    return An, Aa


def design_one_trial(inp: TrayHydraulicsInputs, charts: Charts, tray_spacing_m: float, hole_area_frac_active: float) -> TrayDesignResult:
    msgs: List[str] = []

    # Step 4: diameter from flooding
    Dc, Ac, Ad, _ = initial_diameter_from_flooding(inp, charts, tray_spacing_m, hole_area_frac_active)
    An, Aa = tray_areas(Ac, Ad)

    # Step 5: arrangement
    arrangement = select_flow_arrangement(Dc, inp.Lw_max / inp.rhoL, inp.flow_arrangement)

    # Step 6: layout
    lw_m = weir_length_rule(Dc, arrangement)
    hw_mm, dh_mm, t_mm = inp.hw_mm, inp.dh_mm, inp.t_mm

    Ap = Aa  # first-pass
    Ah = hole_area_frac_active * Aa

    # Step 7: weeping (min case)
    how_min = weir_crest_how_mm(inp.Lw_min, inp.rhoL, lw_m)
    if how_min < inp.weir_crest_min_mm:
        msgs.append(f"Low weir crest @ min flow: how={how_min:.1f} mm (<{inp.weir_crest_min_mm} mm).")

    hclear_min = hw_mm + how_min
    K2 = charts.K2(hclear_min)
    uh_min_req = weep_point_uh_min_mps(K2, dh_mm, inp.rhoV)
    uh_min_act = (inp.Vw_min / inp.rhoV) / Ah
    weeping_ok = uh_min_act >= uh_min_req
    if not weeping_ok:
        msgs.append(f"Weeping risk: uh(min)={uh_min_act:.2f} < {uh_min_req:.2f} m/s.")

    # Step 8: pressure drop (max)
    how_max = weir_crest_how_mm(inp.Lw_max, inp.rhoL, lw_m)
    uh_max = (inp.Vw_max / inp.rhoV) / Ah

    C0 = C0_const(t_mm / dh_mm, Ah / Ap)
    hd = dry_tray_drop_hd_mm(uh_max, C0, inp.rhoV, inp.rhoL)
    hr = residual_head_hr_mm(inp.rhoL)
    ht = total_tray_drop_ht_mm(hd, hw_mm, how_max, hr)
    dP = tray_dP_Pa(ht, inp.rhoL)

    dp_ok = ht <= inp.ht_limit_mm
    if not dp_ok:
        msgs.append(f"High ΔP: ht={ht:.1f} mm (> {inp.ht_limit_mm} mm).")

    # Step 9: downcomer backup
    hap_mm = max(1.0, hw_mm - inp.hap_delta_mm)
    Aap = lw_m * (hap_mm * 1e-3)
    Am = min(Ad, Aap)

    hdc = downcomer_headloss_hdc_mm(inp.Lw_max, inp.rhoL, Am)
    hb = downcomer_backup_hb_mm(hw_mm, how_max, ht, hdc)

    lt_mm = tray_spacing_m * 1000.0
    downcomer_ok = hb <= 0.5 * (lt_mm + hw_mm)
    if not downcomer_ok:
        msgs.append(f"Downcomer backup high: hb={hb:.1f} mm > 0.5*(lt+hw)={0.5*(lt_mm+hw_mm):.1f} mm.")

    tr = downcomer_res_time_s(Ad, hb * 1e-3, inp.rhoL, inp.Lw_max)
    if tr < inp.downcomer_res_time_min_s:
        msgs.append(f"Downcomer residence time low: tr={tr:.2f} s (<{inp.downcomer_res_time_min_s} s).")

    # Step 11: recompute % flooding with chosen diameter
    u_net = (inp.Vw_max / inp.rhoV) / An
    flv = FLV(inp.Lw_max, inp.Vw_max, inp.rhoV, inp.rhoL)

    K1_use = charts.K1(flv, tray_spacing_m) * ((inp.sigma / 0.02) ** 0.2) * k1_area_multiplier(hole_area_frac_active)
    uf = flooding_velocity(K1_use, inp.rhoL, inp.rhoV)
    pf = 100.0 * (u_net / uf)

    # Step 12: entrainment from J curves
    J = charts.J(flv, pf)
    entrainment_ok = J <= inp.entrainment_J_limit
    if not entrainment_ok:
        msgs.append(f"Entrainment high: J={J:.3f} (> {inp.entrainment_J_limit}).")

    ok = weeping_ok and dp_ok and downcomer_ok and entrainment_ok

    return TrayDesignResult(
        ok=ok,
        messages=msgs,
        tray_spacing_m=tray_spacing_m,
        hole_area_frac_active=hole_area_frac_active,
        arrangement=arrangement,
        Dc_m=Dc,
        Ac_m2=Ac,
        Ad_m2=Ad,
        An_m2=An,
        Aa_m2=Aa,
        lw_m=lw_m,
        Ah_m2=Ah,
        Ap_m2=Ap,
        percent_flood_max=pf,
        weeping_ok=weeping_ok,
        dp_ok=dp_ok,
        downcomer_ok=downcomer_ok,
        entrainment_ok=entrainment_ok,
        ht_mm_liq=ht,
        dP_Pa=dP,
        hb_mm=hb,
        tr_s=tr,
        J=J,
    )


def optimise_tray_design(inp: TrayHydraulicsInputs, charts: Charts) -> TrayDesignResult:
    feasible: List[TrayDesignResult] = []
    best_any: Optional[TrayDesignResult] = None
    best_score = 1e99

    for lt in inp.tray_spacings_m:
        for frac in inp.hole_area_fracs:
            res = design_one_trial(inp, charts, lt, frac)

            if res.ok:
                feasible.append(res)

            fails = sum([not res.weeping_ok, not res.dp_ok, not res.downcomer_ok, not res.entrainment_ok])
            score = fails * 1e6 + res.Dc_m * 1e3 + res.ht_mm_liq
            if score < best_score:
                best_score = score
                best_any = res

    if feasible:
        feasible.sort(key=lambda r: (r.Dc_m, r.tray_spacing_m, r.ht_mm_liq))
        return feasible[0]

    assert best_any is not None
    best_any.messages.insert(0, "No fully feasible design found in search grid; returning closest candidate.")
    return best_any


# -------------------------
# Run example
# -------------------------
if __name__ == "__main__":
    charts = load_charts_from_script_folder()

    # Replace these with your real max/min section flows + properties (Step 1 & 2).
    # IMPORTANT: Vw and Lw are MASS flow rates (kg/s) in this implementation.
    inp = TrayHydraulicsInputs(
        Lw_max=37.81,
        Lw_min=(0.7*37.81),
        Vw_max=37.83,
        Vw_min=(0.7*37.83),
        rhoL=1150.3,
        rhoV=3.424,
        sigma=0.020,

        tray_spacings_m=(0.15, 0.25, 0.30, 0.45, 0.60, 0.90),  # should match your K1 spacings
        hole_area_fracs=(0.03, 0.04 ,0.05 ,0.06, 0.07, 0.08, 0.09, 0.10),

        flow_arrangement=None,  # or "single"/"reverse"/"double" to force
        ht_limit_mm=120.0,
        entrainment_J_limit=0.10,
    )

    best = optimise_tray_design(inp, charts)

    print("\n=== BEST RESULT ===")
    print(f"OK: {best.ok}")
    print(f"Spacing lt: {best.tray_spacing_m:.2f} m | Ah/Aa: {best.hole_area_frac_active:.2f} | arrangement: {best.arrangement}")
    print(f"Dc: {best.Dc_m:.3f} m | %flood(max): {best.percent_flood_max:.1f}% | J: {best.J:.3f}")
    print(f"Weeping: {best.weeping_ok} | ΔP: {best.dp_ok} (ht={best.ht_mm_liq:.1f} mm, dP={best.dP_Pa/1000:.2f} kPa)")
    print(f"Downcomer: {best.downcomer_ok} (hb={best.hb_mm:.1f} mm, tr={best.tr_s:.2f} s)")
    print(f"Entrainment: {best.entrainment_ok}")

    if best.messages:
        print("\nMessages:")
        for m in best.messages:
            print(" -", m)