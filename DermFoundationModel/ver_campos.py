import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PAD_PATH = os.getenv("PAD_PATH", ".")

df = pd.read_csv(os.path.join(PAD_PATH, "metadata.csv"))
for c in ["grew", "changed", "bleed", "itch", "hurt", "elevation",
          "skin_cancer_history", "cancer_history"]:
    print(c, df[c].value_counts(dropna=False).to_dict())

df["diagnostic"] = df["diagnostic"].str.upper()
print(pd.crosstab(df["diagnostic"], df["cancer_history"].isna(), normalize="index").round(2))

def desconhecido(serie):
    s = serie.astype(str).str.strip().str.upper()
    return ~s.isin(["TRUE", "FALSE", "MALE", "FEMALE"])

for c in ["gender", "grew", "changed"]:
    print(f"\n{c} desconhecido (fração por diagnóstico):")
    print(pd.crosstab(df["diagnostic"], desconhecido(df[c]), normalize="index").round(2))

print("\nage ausente:")
print(pd.crosstab(df["diagnostic"], pd.to_numeric(df["age"], errors="coerce").isna(), normalize="index").round(2))