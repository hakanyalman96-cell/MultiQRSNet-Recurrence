"""
Extract electroanatomic-mapping (CARTO) point features per patient, to test whether
procedural mapping data predict recurrence better than the surface ECG.

Each mapped point is an XML file named like  *_P<n>_Point_Export.xml  containing:
  <Annotations Reference_Annotation Map_Annotation/>   -> LAT = Map - Reference (ms)
  <Voltages Unipolar Bipolar/>                          -> local voltages
The earliest endocardial site (most negative LAT) and its voltage are the classic
predictors of ablation success / recurrence.

USAGE
  Put this next to your CARTO export ROOT folder (one sub-folder per patient/study,
  each containing that patient's *_Point_Export.xml files), then:
      python3 carto_mapping_features.py  /path/to/CARTO_root
  (defaults to the current folder if no path given)

OUTPUT  mapping_features.csv  with one row per patient folder:
  patient_folder, n_points, earliest_LAT, LAT_median,
  bip_at_earliest, uni_at_earliest, bip_min, bip_median, uni_median
"""
import sys, os, glob, csv, xml.etree.ElementTree as ET, numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

def parse_point(fp):
    """Return (LAT_ms, unipolar, bipolar) for one point XML, or None on failure."""
    try:
        r = ET.parse(fp).getroot()
        a = r.find("Annotations"); v = r.find("Voltages")
        ref = float(a.get("Reference_Annotation")); mp = float(a.get("Map_Annotation"))
        lat = mp - ref                                   # ms relative to reference
        uni = float(v.get("Unipolar")); bip = float(v.get("Bipolar"))
        # guard against sentinel/invalid values
        uni = uni if abs(uni) < 9000 else np.nan
        bip = bip if abs(bip) < 9000 else np.nan
        return lat, uni, bip
    except Exception:
        return None

def main():
    xmls = glob.glob(os.path.join(ROOT, "**", "*_Point_Export.xml"), recursive=True)
    if not xmls:
        print(f"No *_Point_Export.xml found under {ROOT!r}. Point this at your CARTO export root."); return
    # group points by the TOP-LEVEL folder under ROOT (= one patient/study);
    # robust to nested structures like  ROOT/<patient>/<study>/<map>/*_Point_Export.xml
    by_pt = {}
    for fp in xmls:
        rel = os.path.relpath(fp, ROOT); parts = rel.split(os.sep)
        pat = parts[0] if len(parts) > 1 else os.path.basename(os.path.dirname(fp))
        by_pt.setdefault(pat, []).append(fp)
    print(f"{len(xmls)} point files across {len(by_pt)} patient folders")
    print("  example patient keys:", list(sorted(by_pt))[:5])

    rows = []
    for pat, files in sorted(by_pt.items()):
        recs = [p for p in (parse_point(f) for f in files) if p]
        if not recs:
            continue
        lat = np.array([r[0] for r in recs]); uni = np.array([r[1] for r in recs]); bip = np.array([r[2] for r in recs])
        j = int(np.nanargmin(lat))                       # earliest-activation point
        rows.append(dict(
            patient_folder=pat, n_points=len(recs),
            earliest_LAT=round(float(np.nanmin(lat)), 1),
            LAT_median=round(float(np.nanmedian(lat)), 1),
            bip_at_earliest=round(float(bip[j]), 3) if bip[j]==bip[j] else "",
            uni_at_earliest=round(float(uni[j]), 3) if uni[j]==uni[j] else "",
            bip_min=round(float(np.nanmin(bip)), 3) if np.isfinite(np.nanmin(bip)) else "",
            bip_median=round(float(np.nanmedian(bip)), 3) if np.isfinite(np.nanmedian(bip)) else "",
            uni_median=round(float(np.nanmedian(uni)), 3) if np.isfinite(np.nanmedian(uni)) else "",
        ))
    cols = ["patient_folder","n_points","earliest_LAT","LAT_median","bip_at_earliest",
            "uni_at_earliest","bip_min","bip_median","uni_median"]
    with open("mapping_features.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote mapping_features.csv  ({len(rows)} patients)")
    # quick peek
    for r in rows[:5]:
        print("  ", r["patient_folder"], "| n_pts", r["n_points"], "| earliest_LAT", r["earliest_LAT"], "ms | bip@earliest", r["bip_at_earliest"])

if __name__ == "__main__":
    main()
