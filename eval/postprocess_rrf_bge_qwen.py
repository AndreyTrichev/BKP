import argparse
import json
from pathlib import Path

import numpy as np

DEPTS = ["urban_economy", "urban_development", "education", "culture"]
SUBS = ["gas", "water", "heat", "housing", "roads", "waste", "ecology",
        "construction", "land", "planning", "trade", "beaches", "tourism",
        "preschool", "school", "custody", "institutions", "heritage"]
RRF_K = 60


def hybrid_alpha_05(in_d_ce):
    if not in_d_ce:
        return None
    ce_v = [c["ce_score"] for c in in_d_ce]
    l2_v = [c["l2_score"] or 0 for c in in_d_ce]
    ce_min, ce_max = min(ce_v), max(ce_v)
    l2_min, l2_max = min(l2_v), max(l2_v)
    ce_r = ce_max - ce_min if ce_max > ce_min else 1
    l2_r = l2_max - l2_min if l2_max > l2_min else 1
    best, bs = -1e18, None
    for c in in_d_ce:
        sc = 0.5 * (c["ce_score"] - ce_min) / ce_r + 0.5 * ((c["l2_score"] or 0) - l2_min) / l2_r
        if sc > best:
            best = sc
            bs = c["subdepartment"]
    return bs


def f1_mw(y_true, y_pred, classes):
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        sup = sum(1 for t in y_true if t == c)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[c] = {"f1": f1, "support": sup}
    sup_list = [v for v in out.values() if v["support"] > 0]
    macro = float(np.mean([v["f1"] for v in sup_list])) if sup_list else 0.0
    total = sum(v["support"] for v in out.values())
    weighted = sum(v["f1"] * v["support"] for v in out.values()) / total if total else 0.0
    return macro, weighted


def metrics(gd, gs, pd, ps):
    n = len(gd)
    acc_d = sum(1 for t, p in zip(gd, pd) if t == p) / n
    correct = sum(1 for td, ts, xd, xs in zip(gd, gs, pd, ps) if td == xd and ts == xs)
    f1d_m, f1d_w = f1_mw(gd, pd, DEPTS)
    f1s_m, f1s_w = f1_mw(gs, ps, SUBS)
    return {"n": n, "acc_dept": acc_d, "acc_sub": correct / n,
            "errors_sub": n - correct,
            "f1_dept_macro": f1d_m, "f1_dept_weighted": f1d_w,
            "f1_sub_macro": f1s_m, "f1_sub_weighted": f1s_w}


def add_rrf_bge_qwen(items):
    rrf = {i: 0.0 for i in range(len(items))}
    bge_sorted = sorted(range(len(items)), key=lambda i: items[i]["bge_score"], reverse=True)
    for r, i in enumerate(bge_sorted, 1):
        rrf[i] += 1.0 / (RRF_K + r)
    valid_q = [(i, c["qwen3_score"]) for i, c in enumerate(items) if c.get("qwen3_score") is not None]
    valid_q.sort(key=lambda x: x[1], reverse=True)
    for r, (i, _) in enumerate(valid_q, 1):
        rrf[i] += 1.0 / (RRF_K + r)
    for i, c in enumerate(items):
        c["rrf_bgeqwen"] = rrf[i]


def maxp_top1(items, key):
    if not items:
        return "unknown", "unknown"
    dm = {}
    for c in items:
        v = c.get(key)
        if v is None:
            continue
        if c["department"] not in dm or v > dm[c["department"]]:
            dm[c["department"]] = v
    if not dm:
        return "unknown", "unknown"
    wd = max(dm.items(), key=lambda x: x[1])[0]
    in_d = sorted([c for c in items if c["department"] == wd and c.get(key) is not None],
                  key=lambda x: x[key], reverse=True)
    return wd, in_d[0]["subdepartment"]


def hybrid_alpha(items, key_ce):
    if not items:
        return "unknown", "unknown"
    dm = {}
    for c in items:
        v = c.get(key_ce)
        if v is None:
            continue
        if c["department"] not in dm or v > dm[c["department"]]:
            dm[c["department"]] = v
    if not dm:
        return "unknown", "unknown"
    wd = max(dm.items(), key=lambda x: x[1])[0]
    in_d = [c for c in items if c["department"] == wd and c.get(key_ce) is not None]
    if not in_d:
        return wd, "unknown"
    in_d_remap = [{"ce_score": c[key_ce], "l2_score": c["l2_score"],
                   "subdepartment": c["subdepartment"]} for c in in_d]
    return wd, hybrid_alpha_05(in_d_remap)


ALL_MODES = ["bge_top1", "bge_hybrid", "qwen3_cascade_top1", "bge_dept_qwen_sub",
             "rrf_bgeqwen_top1", "rrf_bgeqwen_dept_qwen_sub",
             "rrf_bgeqwen_dept_hyb_sub", "bge_dept_rrf_sub"]


def predict_modes(items, restrict=None):
    if restrict:
        items = [c for c in items if c["department"] == restrict]
    if not items:
        return {m: ("unknown", "unknown") for m in ALL_MODES}

    add_rrf_bge_qwen(items)
    out = {}

    out["bge_top1"] = maxp_top1(items, "bge_score")
    out["bge_hybrid"] = hybrid_alpha(items, "bge_score")

    cascade = [c for c in items if c.get("qwen3_score") is not None]
    out["qwen3_cascade_top1"] = maxp_top1(cascade, "qwen3_score") if cascade else ("unknown", "unknown")

    wd, _ = maxp_top1(items, "bge_score")
    in_d = [c for c in items if c["department"] == wd]
    in_d_q = [c for c in in_d if c.get("qwen3_score") is not None]
    if in_d_q:
        sub = sorted(in_d_q, key=lambda x: x["qwen3_score"], reverse=True)[0]["subdepartment"]
    elif in_d:
        sub = sorted(in_d, key=lambda x: x["bge_score"], reverse=True)[0]["subdepartment"]
    else:
        sub = "unknown"
    out["bge_dept_qwen_sub"] = (wd, sub)

    out["rrf_bgeqwen_top1"] = maxp_top1(items, "rrf_bgeqwen")

    wd, _ = maxp_top1(items, "rrf_bgeqwen")
    in_d = [c for c in items if c["department"] == wd]
    in_d_q = [c for c in in_d if c.get("qwen3_score") is not None]
    if in_d_q:
        sub = sorted(in_d_q, key=lambda x: x["qwen3_score"], reverse=True)[0]["subdepartment"]
    elif in_d:
        sub = sorted(in_d, key=lambda x: x["bge_score"], reverse=True)[0]["subdepartment"]
    else:
        sub = "unknown"
    out["rrf_bgeqwen_dept_qwen_sub"] = (wd, sub)

    in_d_full = [c for c in items if c["department"] == wd]
    if in_d_full:
        in_d_remap = [{"ce_score": c["bge_score"], "l2_score": c["l2_score"],
                       "subdepartment": c["subdepartment"]} for c in in_d_full]
        sub = hybrid_alpha_05(in_d_remap)
    else:
        sub = "unknown"
    out["rrf_bgeqwen_dept_hyb_sub"] = (wd, sub)

    wd, _ = maxp_top1(items, "bge_score")
    in_d = [c for c in items if c["department"] == wd]
    if in_d:
        sub = sorted(in_d, key=lambda x: x["rrf_bgeqwen"], reverse=True)[0]["subdepartment"]
    else:
        sub = "unknown"
    out["bge_dept_rrf_sub"] = (wd, sub)

    return out


def evaluate(rows, restrict=None):
    in_scope = [r for r in rows
                if r.get("gt_dept") not in ("unknown", None)
                and r.get("gt_sub") and r.get("ce_stage")]
    gd = [r["gt_dept"] for r in in_scope]
    gs = [r["gt_sub"] for r in in_scope]
    all_preds = {}
    for r in in_scope:
        preds = predict_modes(r["ce_stage"], restrict=restrict)
        for mode, (d, s) in preds.items():
            all_preds.setdefault(mode, ([], []))
            all_preds[mode][0].append(d)
            all_preds[mode][1].append(s)
    out = {}
    for mode, (pds, pss) in all_preds.items():
        out[mode] = metrics(gd, gs, pds, pss)
    return out


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", required=True, help="path to master_live.jsonl")
    parser.add_argument("--valtest", required=True, help="path to master_valtest.jsonl")
    parser.add_argument("--out", required=True, help="path to write metrics json")
    args = parser.parse_args()

    live = load_jsonl(args.live)
    valtest = load_jsonl(args.valtest)
    val = [r for r in valtest if r["split"] == "val"]
    test = [r for r in valtest if r["split"] == "test"]

    sets = [
        ("Live full (n=70)", live, None),
        ("Live UGX restrict", live, "urban_economy"),
        (f"Val (n={len(val)})", val, None),
        (f"Test (n={len(test)})", test, None),
    ]

    all_results = {}
    for label, rows, restrict in sets:
        res = evaluate(rows, restrict=restrict)
        all_results[label] = res
        print(f"\n--- {label} ---")
        for mode, m in res.items():
            print(f"  {mode:<32} acc_d={m['acc_dept']:.3f}  acc_s={m['acc_sub']:.3f}  "
                  f"F1m(s)={m['f1_sub_macro']:.3f}  err={m['errors_sub']:>3}")

    print("\n\n" + "=" * 100)
    print("  Delta acc_sub vs bge_hybrid (baseline)")
    print("=" * 100)
    methods = ["bge_top1", "qwen3_cascade_top1", "bge_dept_qwen_sub",
               "rrf_bgeqwen_top1", "rrf_bgeqwen_dept_qwen_sub",
               "rrf_bgeqwen_dept_hyb_sub", "bge_dept_rrf_sub"]
    print(f"\n{'Method':<32}", end="")
    for label, _, _ in sets:
        print(f" {label.split('(')[0].strip():>16}", end="")
    print()
    for m in methods:
        print(f"{m:<32}", end="")
        for label, _, _ in sets:
            base = all_results[label].get("bge_hybrid", {}).get("acc_sub", 0)
            v = all_results[label].get(m, {}).get("acc_sub", 0)
            d = (v - base) * 100
            sign = "+" if d >= 0 else ""
            print(f" {sign}{d:>14.1f}pp", end="")
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
