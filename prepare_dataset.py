"""
prepare_dataset.py
==================
Builds a de-identified, labelled ECG dataset from:

  (a) the clinical Excel database  (one row per patient; recurrence label), and
  (b) the per-patient ZIP archives of CARTO exports, organised by ablation
      approach (e.g. "SAG YAKLASIM" / "SOL YAKLASIM" / "SAG VE SOL YAKLASIM").

The ZIP files are named with real patient names, and the Excel contains names
and national ID numbers. Neither may leave the local machine. This script:

  * matches each ZIP to its Excel row on a normalised name,
  * assigns a pseudonymous ID (PT0001, PT0002, ...),
  * extracts the ECG .txt files under that pseudonymous ID,
  * writes `manifest.csv` containing NO identifying information, and
  * writes `crosswalk_PRIVATE.csv` (pseudonym <-> real name) which must be kept
    off GitHub and out of any supplementary material.

The national ID column is never read or written.

Usage
-----
Always start with a dry run to check the name matching:

    python prepare_dataset.py \
        --excel "/content/drive/MyDrive/Kopya VES1 son.xlsx" \
        --zips  "/content/drive/MyDrive/SAG YAKLASIM" \
                "/content/drive/MyDrive/SOL YAKLASIM" \
                "/content/drive/MyDrive/SAG VE SOL YAKLASIM" \
        --out   /content/dataset \
        --dry-run

Then drop --dry-run to extract. Afterwards:

    python recurrence_cv.py --manifest /content/dataset/manifest.csv --sanity-check
"""

import argparse
import difflib
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter

import pandas as pd

NAME_COL_HINT = "İSİM"
RECUR_COL_HINT = "Recurrence"
LOC_COL_HINT = "Localisation"
FORBIDDEN = ("TC KİMLİK", "TC KIMLIK")   # never read, never written

TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


def norm_name(x: str) -> str:
    """Normalise a Turkish name for matching: lowercase, de-accent, squeeze."""
    s = str(x).strip().translate(TR_MAP).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_col(df: pd.DataFrame, hint: str) -> str:
    for c in df.columns:
        if hint.lower() in str(c).lower():
            return c
    raise SystemExit(f"Could not find a column containing {hint!r}. "
                     f"Columns are:\n" + "\n".join(f"  {c}" for c in df.columns))


def load_clinical(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    name_c = find_col(df, NAME_COL_HINT)
    rec_c = find_col(df, RECUR_COL_HINT)
    loc_c = find_col(df, LOC_COL_HINT)

    out = pd.DataFrame({
        "name_raw": df[name_c].astype(str).str.strip(),
        "label": pd.to_numeric(df[rec_c], errors="coerce"),
        # collapse whitespace/case variants, e.g. "RVOT" vs "RVOT "
        "localisation": df[loc_c].astype(str).str.strip().str.upper(),
    })
    out["key"] = out["name_raw"].map(norm_name)

    bad = out["label"].isna().sum()
    if bad:
        print(f"WARNING: {bad} rows have a missing recurrence label and will be dropped.")
        out = out[out["label"].notna()]
    out["label"] = out["label"].astype(int)

    dupes = out["key"].duplicated(keep=False)
    if dupes.any():
        print(f"WARNING: {dupes.sum()} rows share a normalised name. "
              "These cannot be matched unambiguously and will be skipped.")
        out = out[~dupes]
    return out.reset_index(drop=True)


ZIP_STEM_RE = re.compile(r"^(.*?)\.zip", re.IGNORECASE)
COPY_PREFIX_RE = re.compile(r"^(copy of|kopya|kopyası)\s+", re.IGNORECASE)


def zip_stem(filename: str) -> str:
    """Recover the patient name from a possibly-mangled archive filename.

    Cloud storage often renames duplicates, e.g.
        'ahmet yilmaz.zip adli dosyanin kopyasi'   (Google Drive, Turkish)
        'ahmet yilmaz.zip (1)'
        'Copy of ahmet yilmaz.zip'
    Taking everything before the first '.zip' recovers the name in all of these.
    """
    name = COPY_PREFIX_RE.sub("", filename)
    m = ZIP_STEM_RE.match(name)
    return m.group(1) if m else os.path.splitext(name)[0]


def scan_zips(dirs):
    """Return [(archive_path, source_folder, normalised_stem), ...].

    Scans recursively, and accepts archives whose filename has been altered by
    cloud-storage duplication (so the name no longer *ends* in '.zip'). Every
    candidate is verified with zipfile.is_zipfile before being accepted.
    """
    found, not_archives = [], []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"WARNING: not a directory, skipping: {d}")
            continue
        top = os.path.basename(os.path.normpath(d))
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if ".zip" not in f.lower():
                    continue
                p = os.path.join(root, f)
                if not zipfile.is_zipfile(p):
                    not_archives.append(p)
                    continue
                found.append((p, top, norm_name(zip_stem(f))))
    if not_archives:
        print(f"NOTE: {len(not_archives)} file(s) look like archives by name but "
              f"are not readable ZIPs, e.g.\n  {os.path.basename(not_archives[0])}")
    return found


def dedupe(matched):
    """Keep one archive per patient; report patients found in several folders."""
    by_key = {}
    dupes = []
    for path, folder, key, how in matched:
        if key in by_key:
            dupes.append((key, by_key[key][1], folder))
        else:
            by_key[key] = (path, folder, key, how)
    if dupes:
        print(f"\nSame patient present in more than one folder : {len(dupes)}")
        print("  (keeping the first occurrence, ignoring the copy)")
        for key, first, second in dupes[:15]:
            print(f"    {mask(key)}: {first}  +  {second}")
    return list(by_key.values()), dupes


def provenance_report(matched, clin):
    """Check whether the outcome label is confounded with the source folder.

    If every recurrence archive comes from one folder and every cured archive
    from another, any classifier can separate the two by learning folder- or
    batch-specific signal rather than physiology.
    """
    lab = clin.set_index("key")["label"].to_dict()
    rows = [{"folder": f, "label": lab[k]} for _, f, k, _ in matched if k in lab]
    if not rows:
        return
    t = pd.crosstab(pd.DataFrame(rows)["folder"], pd.DataFrame(rows)["label"])
    for c in (0, 1):
        if c not in t.columns:
            t[c] = 0
    t = t[[0, 1]].rename(columns={0: "cured", 1: "recurrence"})
    t["total"] = t.sum(axis=1)
    print("\n=== PROVENANCE: source folder x outcome ===")
    print(t.to_string())

    # how separable is the label from the folder alone?
    n = int(t["total"].sum())
    majority = int(t[["cured", "recurrence"]].max(axis=1).sum())
    acc = majority / n if n else 0
    print(f"\nA classifier that only looked at the source folder would be "
          f"{100 * acc:.1f}% accurate.")
    if acc > 0.90:
        print("  !! WARNING: the label is almost perfectly predicted by the folder.")
        print("     Any model result is then uninterpretable: the network may be")
        print("     learning acquisition/batch differences rather than ECG features.")
        print("     Before trusting results, confirm that recurrence and cured")
        print("     recordings share the same export settings, devices and era,")
        print("     and ideally assemble both classes from the same source.")
    elif acc > 0.75:
        print("  ! Note: folder and outcome overlap substantially. Keep this in mind")
        print("    when interpreting performance.")


def match(zips, clin, fuzzy_cutoff=0.90):
    keys = clin["key"].tolist()
    rows, unmatched = [], []
    for path, approach, key in zips:
        if key in keys:
            rows.append((path, approach, key, "exact"))
            continue
        close = difflib.get_close_matches(key, keys, n=1, cutoff=fuzzy_cutoff)
        if close:
            rows.append((path, approach, close[0], "fuzzy"))
        else:
            unmatched.append((path, key))
    return rows, unmatched


def extract_patient(zpath: str, dest: str, validate: bool):
    """Extract every .txt from a ZIP into dest. Returns list of written paths."""
    os.makedirs(dest, exist_ok=True)
    written = []
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist()
                   if m.lower().endswith(".txt") and not m.endswith("/")]
        for i, m in enumerate(members):
            # flatten: keep only the basename, prefix with index to avoid clashes
            base = f"{i:03d}_{os.path.basename(m)}"
            target = os.path.join(dest, base)
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written.append(target)
    if validate and written:
        check_carto(written[0])
    return written


def check_carto(path: str):
    """Warn (once) if the file does not look like the expected CARTO export."""
    need = ["V1(22)", "I(110)", "aVF(173)"]
    try:
        with open(path, "r", errors="ignore") as f:
            lines = [next(f) for _ in range(3)]
    except Exception as e:
        print(f"  NOTE: could not read {path}: {e}")
        return
    hdr = lines[2].split() if len(lines) > 2 else []
    missing = [n for n in need if n not in hdr]
    if missing:
        print("\n  !! The extracted files may not match the expected CARTO layout.")
        print(f"     checked: {path}")
        print(f"     missing from header line 3: {missing}")
        print(f"     header line 3 begins: {hdr[:12]}")
        print("     If this is wrong, recurrence_cv.py will not parse them.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", required=True)
    ap.add_argument("--zips", nargs="+", required=True,
                    help="one or more folders containing per-patient .zip files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="report matching only; extract nothing")
    ap.add_argument("--fuzzy-cutoff", type=float, default=0.90)
    args = ap.parse_args()

    clin = load_clinical(args.excel)
    print(f"clinical rows usable      : {len(clin)}")
    print(f"  cured (0)               : {(clin.label == 0).sum()}")
    print(f"  recurrence (1)          : {(clin.label == 1).sum()} "
          f"({100 * clin.label.mean():.1f}%)")

    zips = scan_zips(args.zips)
    print(f"zip archives found        : {len(zips)}")
    for a, n in Counter(a for _, a, _ in zips).items():
        print(f"  {a}: {n}")

    matched, unmatched = match(zips, clin, args.fuzzy_cutoff)
    matched, _ = dedupe(matched)
    n_fuzzy = sum(1 for m in matched if m[3] == "fuzzy")
    print(f"\nmatched to a clinical row : {len(matched)} "
          f"({n_fuzzy} by approximate name match)")

    if n_fuzzy:
        print("  approximate matches (verify these are the same person):")
        for p, a, k, how in matched:
            if how == "fuzzy":
                z = norm_name(os.path.splitext(os.path.basename(p))[0])
                print(f"    zip {mask(z)}  ->  excel {mask(k)}")

    if unmatched:
        print(f"\nZIPs with NO clinical row : {len(unmatched)}")
        for p, k in unmatched:
            print(f"    {mask(k)}   ({os.path.basename(os.path.dirname(p))})")

    used = {k for _, _, k, _ in matched}
    missing_ecg = clin[~clin["key"].isin(used)]
    if len(missing_ecg):
        print(f"\nClinical rows with NO ZIP : {len(missing_ecg)}")
        print(f"  of which recurrence      : {(missing_ecg.label == 1).sum()}")
        print("  (these patients cannot contribute to the model)")

    if not matched:
        raise SystemExit("\nNothing matched. Check --zips paths and the name column.")

    # ---- assign pseudonyms in a stable order ------------------------------
    clin_idx = clin.set_index("key")
    order = sorted({k for _, _, k, _ in matched})
    pseudo = {k: f"PT{i:04d}" for i, k in enumerate(order, 1)}

    print(f"\n=== EFFECTIVE COHORT ===")
    eff = clin[clin["key"].isin(order)]
    print(f"patients      : {len(eff)}")
    print(f"  cured       : {(eff.label == 0).sum()}")
    print(f"  recurrence  : {(eff.label == 1).sum()} ({100 * eff.label.mean():.1f}%)")

    provenance_report(matched, clin)

    if args.dry_run:
        print("\n--dry-run set; nothing was extracted.")
        print("Review the matching above, then re-run without --dry-run.")
        return

    # ---- extract -----------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    ecg_root = os.path.join(args.out, "ecg")
    rows, crosswalk, validated = [], [], False
    for zpath, approach, key, how in matched:
        pid = pseudo[key]
        info = clin_idx.loc[key]
        files = extract_patient(zpath, os.path.join(ecg_root, pid),
                                validate=not validated)
        validated = True
        if not files:
            print(f"  NOTE: no .txt inside {os.path.basename(zpath)}")
        for fp in files:
            rows.append({
                "path": os.path.relpath(fp, args.out),
                "patient": pid,
                "label": int(info["label"]),
                "site": info["localisation"],
                "approach": approach,
            })
        crosswalk.append({"patient": pid, "name": info["name_raw"],
                          "zip": os.path.basename(zpath), "match": how})

    man = pd.DataFrame(rows)
    if man.empty:
        raise SystemExit("No .txt files were extracted from any archive.")
    man_path = os.path.join(args.out, "manifest.csv")
    man.to_csv(man_path, index=False)

    cw_path = os.path.join(args.out, "crosswalk_PRIVATE.csv")
    pd.DataFrame(crosswalk).to_csv(cw_path, index=False)

    per = man.groupby("patient").size()
    print(f"\nrecordings extracted : {len(man)}")
    print(f"recordings/patient   : mean {per.mean():.2f}  min {per.min()}  max {per.max()}")
    print(f"\nmanifest  -> {man_path}")
    print(f"crosswalk -> {cw_path}")
    print("\n" + "!" * 70)
    print("crosswalk_PRIVATE.csv links pseudonyms to real patient names.")
    print("Keep it OFF GitHub and out of any supplementary file.")
    print("manifest.csv contains no identifying information and is safe to use.")
    print("!" * 70)
    print(f"\nNext:\n  python recurrence_cv.py --manifest {man_path} --sanity-check")


def mask(n: str) -> str:
    return " ".join(w[0] + "*" * (len(w) - 1) for w in str(n).split())


if __name__ == "__main__":
    main()
