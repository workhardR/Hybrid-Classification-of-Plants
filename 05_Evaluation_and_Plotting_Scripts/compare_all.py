"""Run AFTER training: builds one comparison table of all 7 techniques."""
import glob, json, os
import pandas as pd
rows=[json.load(open(p)) for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),"*","outputs","*","summary.json")))]
if not rows: raise SystemExit("No results yet - run the model folders first.")
df=pd.DataFrame(rows)
for m in ("80:20","5-fold"):
    s=df[df["mode"]==m]
    if len(s): print(f"\n===== {m} : TEST RESULTS (all models) ====="); print(s.drop(columns="mode").to_string(index=False))
df.to_csv("comparison_all_models.csv",index=False)
