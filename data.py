import pandas as pd
import json
from pathlib import Path

out_dir = Path("documents")
out_dir.mkdir(exist_ok=True)

with open("data/News_Category_Dataset_v3.json") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
df["text"] = df["headline"].fillna("") + " " + df["short_description"].fillna("")

for i, text in enumerate(df["text"][:1000]):
    if not text.strip():
        continue
    with open(out_dir / f"kaggle_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)
