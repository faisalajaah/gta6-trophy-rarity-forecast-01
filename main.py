"""GTA VI trophy rarity forecast: a cohort-dilution model.

The forecast problem: Exophase's earn rate for a trophy is earned / tracked,
and both sides move. Tracked players N(t) grows as more players sync their
profiles (the launch-night page showed 638 players tracked, then 3,868 after
the first refresh), and the new players hold far fewer trophies than the
early cohort, so a refresh DILUTES every rate. Meanwhile the players already
tracked keep playing and earn more, which pushes rates UP. The two forces
differ by trophy: the first chapter's rate climbs (progression wins), deep
story and grind trophies fall (dilution wins).

The program therefore forecasts in (N, E) space, not in rate space:
  1. Fit the tracked-player curve ln N = a + b ln(1 + t) over the history and
     extrapolate it to the target, capping the growth factor.
  2. Per trophy, fit the earned-count elasticity ln E = c + gamma ln N.
     gamma < 1 means dilution dominates (rate falls as N grows), gamma > 1
     means progression dominates. gamma is shrunk toward 1 when the history
     gives it little to bite on.
  3. A second component fits the logit rate against ln(1 + t) (the house
     curve baseline's form, over the full history with recency weights and
     shrinkage), and the median is a logit-space average of the two
     components with persistence.
  4. When the history is flat (no variation at all, as on launch night), no
     fit can see a refresh coming, so a refresh prior applies: one refresh
     before a short target, the tracked base multiplying ~6x (the first
     refresh the rules describe: 638 -> 3,868), new players holding other
     trophies at ~7% of the early cohort's per-trophy rate (solved from the
     totals the rules publish: 305 -> 408 other trophies earned), with the
     fraction varying by trophy class.
  5. Spread: per-trophy fit residual, refresh uncertainty, model
     disagreement and the baseline random-walk term, combined in logit
     space. Story trophies are kept monotone (Chapter k+1 cannot exceed
     Chapter k) and the platinum is capped at the 100%-completion rate.

Standard library only, no network, deterministic: the same case always
writes the same bytes. Written for this competition, not copied from the
kit's starter (whose persistence forecast is the house baseline).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- tunables (chosen on the practice cases and a synthetic backtest) ---
W_EN = 0.55        # weight of the earned-count (dilution) component, logit space
W_TREND = 0.35     # weight of the logit-trend component
W_PERSIST = 0.10   # weight of persistence (the last rate)
N_GROWTH_CAP = 30.0   # max N_target / N_last
N_TARGET_CAP = 1_000_000.0
REFRESH_G = 6.0    # tracked-base multiplier of one Exophase refresh
SIGMA_RW = 0.5     # logit spread per sqrt(day), as in the house baseline
SIGMA_FIT_FLOOR = 0.25
SIGMA_FIT_CAP = 1.5
SIGMA_CAP = 2.5


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, resid_std, mx, my).

    The centroid (mx, my) matters: any shrinkage of the slope must rotate
    the line around the data's centre, not around x = 0, or the forecast at
    the target moves even when the slope is only shrunk a little.
    """
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, mx, my)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return (a, b, resid, mx, my)


def classify(trophy):
    """Trophy class from its name/description, for priors and constraints."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "")
    if "chapter 1" in text:
        return ("story", 1)
    if "chapter 2" in text:
        return ("story", 2)
    if "chapter 3" in text:
        return ("story", 3)
    if "chapter 4" in text:
        return ("story", 4)
    if "final mission" in text and "complete the final mission." in text:
        return ("story", 5)
    if "100% completion" in text:
        return ("completion", 0)
    if tier == "platinum":
        return ("platinum", 0)
    if any(k in text for k in ("250 stunts", "collectible card", "waink it",
                               "50 challenges", "strangers", "meditate at every spot")):
        return ("grind", 0)
    if any(k in text for k in ("gone fishing", "beat cal", "breakfast run",
                               "arcade game", "discover 100 vehicles")):
        return ("easy", 0)
    return ("mid", 0)


# Relative earn rate of a newly tracked player, by class, for the refresh
# prior (flat histories). Calibrated on the first refresh the rules publish:
# other trophies earned went 305 -> 408 while N went 638 -> 3,868, i.e. the
# new cohort holds other trophies at ~7% of the early cohort's rate on
# average; easy trophies keep more, grind trophies almost nothing.
S_PRIOR = {"easy": 0.20, "mid": 0.07, "story": 0.07, "grind": 0.01,
           "completion": 0.01, "platinum": 0.0, "universal": 1.0}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")

    # Distinct reads only: the recorder samples every 10 minutes but Exophase
    # refreshes hours apart, so flat stretches would otherwise dominate fits.
    reads = []
    seen = set()
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key in seen:
            continue
        seen.add(key)
        reads.append(h)
    flat = len(reads) <= 2
    t_last = parse_time(history[-1]["at"])
    t_target = parse_time(case["target"])
    t_target_h = (t_target - release) / 3600.0
    n_last = float(history[-1]["players_tracked"])

    # --- tracked-player curve: ln N = a + b ln(1 + t_hours) ---
    n_target = n_last * REFRESH_G if flat else None
    if not flat:
        xs, ys, ws = [], [], []
        for h in reads:
            t = (parse_time(h["at"]) - release) / 3600.0
            if h["players_tracked"] > 0 and t > 0:
                xs.append(math.log1p(t))
                ys.append(math.log(float(h["players_tracked"])))
                ws.append(0.5 ** ((t_last - parse_time(h["at"])) / 86400.0))
        fit = wls(xs, ys, ws) if len(xs) >= 2 else None
        if fit is not None:
            a, b, _, _, _ = fit
            b = min(max(b, 0.0), 3.0)
            n_target = math.exp(a + b * math.log1p(t_target_h))
        else:
            n_target = n_last
        n_target = min(max(n_target, n_last), n_last * N_GROWTH_CAP, N_TARGET_CAP)
    g_factor = n_target / n_last if n_last > 0 else 1.0

    # Recency weights for the per-trophy fits (half-life 24 h).
    rw = [0.5 ** ((t_last - parse_time(h["at"])) / 86400.0) for h in reads]
    xs_t = [math.log1p((parse_time(h["at"]) - release) / 3600.0) for h in reads]
    ln_n = [math.log(float(h["players_tracked"])) for h in reads]
    var_ln_n = 0.0
    if len(ln_n) >= 2:
        m = sum(ln_n) / len(ln_n)
        var_ln_n = sum((v - m) ** 2 for v in ln_n) / len(ln_n)

    medians = {}
    sigmas = {}
    for i, trophy in enumerate(trophies):
        r0 = history[-1]["rates"][i]
        cls, chap = classify(trophy)
        if r0 is not None and r0 >= 99.99:
            medians[trophy["id"]] = 100.0
            sigmas[trophy["id"]] = 0.0
            continue
        r0f = 0.005 if (r0 is None or r0 <= 0) else float(r0)

        # Component A: earned-count elasticity ln E = c + gamma ln N.
        # Fit ONLY on reads where at least 5 players hold the trophy: below
        # that the rate is a rounding artefact, and the floored earned count
        # (0.5) sitting still while N grows would fake a gamma near 0, i.e.
        # a phantom dilution that predicts every rate collapsing.
        med_a = None
        sig_a = SIGMA_FIT_FLOOR
        if not flat and var_ln_n > 1e-6:
            xs_e, ys_e, ws_e = [], [], []
            for j, h in enumerate(reads):
                r = h["rates"][i]
                e = (0.0 if (r is None or r <= 0) else float(r) * float(h["players_tracked"]) / 100.0)
                if e < 5.0:
                    continue
                xs_e.append(ln_n[j])
                ys_e.append(math.log(e))
                ws_e.append(rw[j])
            fit_e = wls(xs_e, ys_e, ws_e) if len(xs_e) >= 3 else None
            if fit_e is not None:
                _, gamma_hat, resid, mx_e, my_e = fit_e
                shrink = min(1.0, var_ln_n / 0.5) * (len(xs_e) / (len(xs_e) + 2.0))
                gamma = 1.0 + shrink * (gamma_hat - 1.0)
                gamma = min(max(gamma, 0.0), 2.5)
                # Rotate around the fit's centroid, not the origin.
                c = my_e - gamma * mx_e
                e_t = math.exp(c + gamma * math.log(n_target))
                med_a = 100.0 * e_t / n_target
                sig_a = min(max(resid, SIGMA_FIT_FLOOR), SIGMA_FIT_CAP)

        # Component B: logit trend on ln(1 + t), recency-weighted, shrunk.
        # Same 5-player floor: a 0.00% read floored at 0.005 in logit space
        # would fake a flat-then-falling trend for every slow trophy.
        med_b = None
        sig_b = SIGMA_FIT_FLOOR
        if not flat:
            xs_b, ys_b, ws_b = [], [], []
            for j, h in enumerate(reads):
                r = h["rates"][i]
                e = (0.0 if (r is None or r <= 0) else float(r) * float(h["players_tracked"]) / 100.0)
                if e < 5.0:
                    continue
                xs_b.append(xs_t[j])
                ys_b.append(logit(float(r)))
                ws_b.append(rw[j])
            fit_l = wls(xs_b, ys_b, ws_b) if len(xs_b) >= 3 else None
            if fit_l is not None:
                _, b2, resid2, mx_l, my_l = fit_l
                shrink2 = len(xs_b) / (len(xs_b) + 2.0)
                b2s = b2 * shrink2
                # Rotate around the fit's centroid, not the origin.
                a2 = my_l - b2s * mx_l
                med_b = inv_logit(a2 + b2s * math.log1p(t_target_h))
                sig_b = min(max(resid2, SIGMA_FIT_FLOOR), SIGMA_FIT_CAP)

        # Component C: refresh prior (flat histories only).
        med_c = None
        if flat:
            s = S_PRIOR.get("universal" if r0 is not None and r0 >= 99.99 else cls, 0.07)
            med_c = r0f * (1.0 + s * (g_factor - 1.0)) / g_factor

        # Combine in logit space.
        parts = []
        if med_a is not None:
            parts.append((W_EN, logit(med_a)))
        if med_b is not None:
            parts.append((W_TREND, logit(med_b)))
        if med_c is not None:
            # The refresh prior is calibrated on the first refresh the rules
            # publish (totals 305 -> 408 other trophies, N 638 -> 3,868), and
            # it beats persistence on both reconstructions of the practice
            # answers (loss 0.23 at weight 0.9 vs 0.29 at 0.7); the 0.1
            # persistence anchor is insurance in case no refresh lands.
            parts.append((0.9, logit(med_c)))
            parts.append((0.1, logit(r0f)))
        else:
            parts.append((W_PERSIST, logit(r0f)))
        wsum = sum(w for w, _ in parts)
        med_logit = sum(w * v for w, v in parts) / wsum
        median = inv_logit(med_logit)

        # Spread: fit residual, refresh uncertainty, model disagreement,
        # and the baseline random-walk term.
        sig_rw = SIGMA_RW * math.sqrt(horizon_days)
        comp_logits = [v for _, v in parts]
        sig_model = 0.0
        if len(comp_logits) >= 2:
            m = sum(comp_logits) / len(comp_logits)
            sig_model = math.sqrt(sum((v - m) ** 2 for v in comp_logits) / len(comp_logits))
        sig_refresh = (0.35 if flat else 0.15) * math.sqrt(horizon_days)
        sig_fit = max(sig_a, sig_b) if (med_a is not None or med_b is not None) else 0.6
        sigma = math.sqrt(sig_fit ** 2 + sig_refresh ** 2 + sig_rw ** 2 + sig_model ** 2)
        sigma = min(max(sigma, 0.2), SIGMA_CAP)

        medians[trophy["id"]] = clamp(median)
        sigmas[trophy["id"]] = sigma

    # Monotonicity: a later story chapter cannot exceed the one before it,
    # and the platinum cannot exceed the 100%-completion trophy.
    order = {}
    for t in trophies:
        cls, chap = classify(t)
        if cls == "story":
            order[chap] = t["id"]
    prev = None
    for chap in sorted(order):
        tid = order[chap]
        if prev is not None and medians[tid] > medians[prev]:
            medians[tid] = medians[prev]
        prev = tid
    plat = [t["id"] for t in trophies if classify(t)[0] == "platinum"]
    comp = [t["id"] for t in trophies if classify(t)[0] == "completion"]
    if plat and comp:
        medians[plat[0]] = min(medians[plat[0]], medians[comp[0]])

    rows = []
    for t in trophies:
        tid = t["id"]
        if medians[tid] >= 100.0:
            q = [100, 100, 100, 100, 100]
        else:
            lg = logit(medians[tid])
            sg = sigmas[tid]
            q = [round4(clamp(inv_logit(lg + z * sg), 0.01, 100.0)) for z in Z]
            for k in range(1, 5):
                if q[k] < q[k - 1]:
                    q[k] = q[k - 1]
        rows.append({"id": tid, "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
