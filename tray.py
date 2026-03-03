import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

g = 9.81

# -----------------------------
# 1) Load digitised curves
# -----------------------------
def load_curve_xy(csv_path, xcol, ycol):
    df = pd.read_csv(csv_path)
    df = df[[xcol, ycol]].dropna().sort_values(xcol)
    return df[xcol].to_numpy(), df[ycol].to_numpy()

def interp1(xq, x, y):
    """
    Safe linear interpolation for scalars or ndarrays.
    Clamps outside range. Works with 2D grids.
    """
    xq_arr = np.asarray(xq)
    flat = xq_arr.ravel()
    out = np.interp(flat, x, y, left=y[0], right=y[-1])
    return out.reshape(xq_arr.shape)

# K1 curves: map tray spacing -> (FLV array, K1 array)
K1_FILES = {
    0.15: "K1_s0p15.csv",
    0.25: "K1_s0p25.csv",
    0.30: "K1_s0p30.csv",
    0.45: "K1_s0p45.csv",
    0.60: "K1_s0p60.csv",
    0.90: "K1_s0p90.csv",
}
K1_DATA = {s: load_curve_xy(f, "FLV", "K1") for s, f in K1_FILES.items()}

# K2 curve
K2_x, K2_y = load_curve_xy("K2.csv", "hclear_mm", "K2")

# Entrainment curves: map percent flood -> (FLV array, J array)
J_DIR = Path(".")
J_DATA = {}
for pf in [30, 35, 40, 45, 50, 60, 70, 80, 90, 95]:
    fn = J_DIR / f"J_pf{pf}.csv"
    if fn.exists():
        J_DATA[pf] = load_curve_xy(str(fn), "FLV", "J")

print("Loaded K1 spacings:", sorted(K1_DATA.keys()))
print("Loaded J percent-flood curves:", sorted(J_DATA.keys()))
print("K2 points:", len(K2_x))
if len(J_DATA) < 3:
    print("WARNING: You have fewer than ~3 entrainment curves. Entrainment check may be weak/disabled.")

# -----------------------------
# 2) Tray geometry (SI)
# -----------------------------
Dc = 3.79                 # column ID, m
tray_spacing = 0.60       # m  (must match one of your digitised K1 spacings, or be between them)

downcomer_frac = 0.12     # Ad/Ac
hole_active_frac = 0.10   # Ah/Aa
dh_mm = 5.0               # hole diameter, mm

weir_height_m = 0.050     # m
weir_length_m = 0.77 * Dc # m

# Towler downcomer backup inputs
apron_offset_mm = 7.5     # apron bottom is 5–10 mm below outlet weir height
t_res_min_s = 3.0         # Towler recommends >= 3 s residence time

Ac = np.pi * Dc**2 / 4
Ad = downcomer_frac * Ac
An = Ac - Ad
Aa = Ac - 2*Ad
Ah = hole_active_frac * Aa

# -----------------------------
# 3) Section properties (SI) – use representative stage values from HYSYS
# -----------------------------
rho_v = 3.42      # kg/m3
rho_L = 1150    # kg/m3

# -----------------------------
# 4) Towler helper equations
# -----------------------------
def FLV(Lw, Vw):
    # Towler Eq 17.50
    return (Lw / Vw) * np.sqrt(rho_v / rho_L)

def K1_from_Fig1734(flv):
    spacings = np.array(sorted(K1_DATA.keys()))

    # clamp to available spacing range
    if tray_spacing <= spacings.min():
        x, y = K1_DATA[spacings.min()]
        return interp1(flv, x, y)
    if tray_spacing >= spacings.max():
        x, y = K1_DATA[spacings.max()]
        return interp1(flv, x, y)

    hi = spacings[spacings >= tray_spacing].min()
    lo = spacings[spacings <= tray_spacing].max()

    xlo, ylo = K1_DATA[lo]
    Klo = interp1(flv, xlo, ylo)

    # if exactly on a digitised spacing, no interpolation needed
    if hi == lo:
        return Klo

    xhi, yhi = K1_DATA[hi]
    Khi = interp1(flv, xhi, yhi)

    w = (tray_spacing - lo) / (hi - lo)
    return Klo + w * (Khi - Klo)

def flooding_velocity_uf(K1):
    # Towler Eq 17.49
    return K1 * np.sqrt((rho_L - rho_v) / rho_v)

def percent_flood(u, uf):
    # Towler Eq 17.51
    return 100.0 * (u / uf)

def weir_crest_how_mm(Lw):
    # Towler Eq 17.53 gives how in mm
    return 750.0 * ((Lw / (rho_L * weir_length_m)) ** (2.0/3.0))

def K2_from_Fig1737(hclear_mm):
    return interp1(hclear_mm, K2_x, K2_y)

def weep_point_uh_mps(K2):
    # Towler Eq 17.52
    return (K2 - 0.90*(25.4 - dh_mm)) * np.sqrt(rho_v)

def entrainment_J(flv, pf):
    # interpolate across digitised J curves: first in FLV, then in %flood
    if len(J_DATA) < 3 or np.isnan(flv) or np.isnan(pf):
        return np.nan

    pfs = np.array(sorted(J_DATA.keys()), dtype=float)
    pf_c = np.clip(pf, pfs.min(), pfs.max())

    hi = pfs[pfs >= pf_c].min()
    lo = pfs[pfs <= pf_c].max()

    xlo, ylo = J_DATA[int(lo)]
    Jlo = np.interp(flv, xlo, ylo, left=ylo[0], right=ylo[-1])

    if hi == lo:
        return Jlo

    xhi, yhi = J_DATA[int(hi)]
    Jhi = np.interp(flv, xhi, yhi, left=yhi[0], right=yhi[-1])

    w = (pf_c - lo) / (hi - lo)
    return Jlo + w*(Jhi - Jlo)

# -----------------------------
# 4b) Towler downcomer backup (Eqs 17.59–17.63)
# -----------------------------
def apron_clearance_area_Aap():
    """
    Towler Eq 17.61: A_ap = h_ap * l_w
    Using weir_length_m as l_w (good first-pass for envelope).
    """
    h_ap_m = max(weir_height_m - apron_offset_mm/1000.0, 1e-6)
    return h_ap_m * weir_length_m  # m^2

def downcomer_headloss_hdc_m(Lw):
    """
    Towler Eq 17.60:
      h_dc = 166 * ( L_ad / (rho_L * A_m) )^2   [mm]
    Convert to meters.
    Where A_m = min(A_d, A_ap)
    """
    A_ap = apron_clearance_area_Aap()
    A_m = np.minimum(Ad, A_ap)

    v_like = Lw / (rho_L * A_m)     # m/s
    hdc_mm = 166.0 * (v_like**2)    # mm
    return hdc_mm / 1000.0          # m

def downcomer_backup_hb_m(Lw, how_mm, hL_extra_m=0.0):
    """
    Towler Eq 17.59 (clear liquid basis):
      h_b = (h_w + h_ow) + h_l + h_dc
    """
    how_m = how_mm / 1000.0
    hdc_m = downcomer_headloss_hdc_m(Lw)
    return (weir_height_m + how_m) + hL_extra_m + hdc_m

def ok_downcomer_backup_Towler(Lw, how_mm):
    """
    Towler Eq 17.62:
      h_b <= 0.5 * (tray_spacing + h_w)

    Towler Eq 17.63:
      t_r = A_d * h_b * rho_L / L_ad  >= 3 s recommended
    """
    hb_m = downcomer_backup_hb_m(Lw, how_mm, hL_extra_m=0.0)

    ok_hb = hb_m <= 0.5 * (tray_spacing + weir_height_m)

    t_r = (Ad * hb_m * rho_L) / np.maximum(Lw, 1e-12)  # s
    ok_tr = t_r >= t_res_min_s

    return ok_hb & ok_tr, hb_m, t_r

# -----------------------------
# 5) Envelope sweep
# -----------------------------
def compute_envelope(
    L_range, V_range, n=180,
    flood_design_frac=0.85,
    weep_margin=1.20,
    J_max=0.10
):
    L_vals = np.linspace(*L_range, n)  # kg/s
    V_vals = np.linspace(*V_range, n)  # kg/s
    Lg, Vg = np.meshgrid(L_vals, V_vals)

    flv = FLV(Lg, Vg)
    K1 = K1_from_Fig1734(flv)
    print("K1 min/max:", float(np.nanmin(K1)), float(np.nanmax(K1)))
    print("FLV min/max:", float(np.nanmin(flv)), float(np.nanmax(flv)))
    uf = flooding_velocity_uf(K1)

    Qv = Vg / rho_v               # m3/s
    u = Qv / An                   # m/s (net area superficial)

    ok_flood = u <= flood_design_frac * uf

    how_mm = weir_crest_how_mm(Lg)
    hclear_mm = weir_height_m*1000 + how_mm
    K2 = K2_from_Fig1737(hclear_mm)

    uh_weep = weep_point_uh_mps(K2)
    uh_act = Qv / Ah              # m/s (through hole area)

    ok_weep = uh_act >= weep_margin * uh_weep

    # Entrainment check (optional)
    pf = percent_flood(u, uf)
    J = np.full_like(pf, np.nan, dtype=float)
    ok_J = np.ones_like(pf, dtype=bool)
    if len(J_DATA) >= 3:
        it = np.nditer(pf, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            J[idx] = entrainment_J(float(flv[idx]), float(pf[idx]))
            it.iternext()
        ok_J = (J <= J_max)

    # Downcomer backup limitation (Towler)
    ok_dc, hb_m, t_r = ok_downcomer_backup_Towler(Lg, how_mm)

    ok = ok_flood & ok_weep & ok_J & ok_dc

    diag = {
        "L": L_vals, "V": V_vals,
        "ok_flood": ok_flood, "ok_weep": ok_weep, "ok_J": ok_J, "ok_dc": ok_dc,
        "u": u, "uf": uf, "pf": pf, "J": J,
        "uh_act": uh_act, "uh_weep": uh_weep,
        "how_mm": how_mm, "hclear_mm": hclear_mm,
        "hb_m": hb_m, "t_r": t_r
    }
    return Lg, Vg, ok, diag



def plot_full_envelope(Lg, Vg, ok, diag, operating_points=None,
                       title="Sieve plate performance envelope (Towler digitised)"):

    plt.figure(figsize=(9,7))

    # --- Shade operating region ---
    plt.contourf(Lg, Vg, ok.astype(int),
                 levels=[-0.5, 0.5, 1.5],
                 colors=["#f8d7da", "#d4edda"], alpha=0.6)

    # --- Individual boundaries ---
    flood = plt.contour(Lg, Vg, diag["ok_flood"].astype(int),
                        levels=[0.5], colors="red", linewidths=2)

    weep = plt.contour(Lg, Vg, diag["ok_weep"].astype(int),
                       levels=[0.5], colors="blue", linewidths=2)

    dc = plt.contour(Lg, Vg, diag["ok_dc"].astype(int),
                     levels=[0.5], colors="purple", linewidths=2)

    if np.isfinite(np.nanmax(diag["J"])):
        entr = plt.contour(Lg, Vg, diag["ok_J"].astype(int),
                           levels=[0.5], colors="orange", linewidths=2)
    else:
        entr = None

    # --- Labels ---
    plt.xlabel("Liquid mass flow Lw (kg/s)")
    plt.ylabel("Vapor mass flow Vw (kg/s)")
    plt.title(title)

    # --- Operating points ---
    if operating_points:
        for Lw, Vw, lab in operating_points:
            plt.scatter([Lw], [Vw], marker="x", s=80)
            plt.text(Lw, Vw, f" {lab}", fontsize=10)

    # --- Manual legend ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Flooding limit"),
        Line2D([0], [0], color="blue", lw=2, label="Weeping limit"),
        Line2D([0], [0], color="purple", lw=2, label="Downcomer backup"),
    ]

    if entr is not None:
        legend_elements.append(
            Line2D([0], [0], color="orange", lw=2, label="Entrainment limit")
        )

    legend_elements.append(
        Line2D([0], [0], marker='s', color='#d4edda',
               markersize=10, linestyle='None',
               label="Satisfactory operation")
    )

    plt.legend(handles=legend_elements, loc="best")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6) Example run
# -----------------------------
if __name__ == "__main__":
    ops = [
        (37.83, 37.812, "Top"),
        (44.97, 47.077, "Bottom"),
    ]

    # Auto sweep bounds (±30% around your points, with a minimum span)
    L_pts = np.array([p[0] for p in ops], dtype=float)
    V_pts = np.array([p[1] for p in ops], dtype=float)

    Lmin = max(0.1, 0.7 * L_pts.min())
    Lmax = 1.3 * L_pts.max()
    Vmin = max(0.1, 0.7 * V_pts.min())
    Vmax = 1.3 * V_pts.max()

    Lg, Vg, ok, diag = compute_envelope(L_range=(Lmin, Lmax), V_range=(Vmin, Vmax), n=400)

    plot_full_envelope(Lg, Vg, ok, diag, operating_points=ops)