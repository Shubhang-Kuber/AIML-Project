"""Load UCI Adult, bucket 8 columns into small integer domains (rules learned on TRAIN only)."""
import pandas as pd

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
KEEP = ["age", "education", "marital", "occupation", "sex", "race", "hours", "income"]
DOMAIN = [5, 4, 3, 7, 2, 5, 3, 2]


def load(path, skip):
    df = pd.read_csv(path, names=COLS, skipinitialspace=True, na_values="?", skiprows=skip)
    df = df.dropna()
    df["income"] = df["income"].str.rstrip(".")  # adult.test labels end with '.'
    return df


def edu_group(e):
    if e in ("Bachelors", "Masters", "Doctorate", "Prof-school"):
        return 3
    if e in ("Some-college", "Assoc-acdm", "Assoc-voc"):
        return 2
    if e == "HS-grad":
        return 1
    return 0  # anything below HS-grad = "school"


def marital_group(m):
    if m.startswith("Married"):
        return 0
    if m == "Never-married":
        return 1
    return 2


def encode(df, occ_top, races):
    out = pd.DataFrame()
    out["age"] = pd.cut(df["age"], [0, 24, 34, 44, 59, 200], labels=False, right=True)
    out["education"] = df["education"].map(edu_group)
    out["marital"] = df["marital-status"].map(marital_group)
    # Occupations outside the train top-6 go to "other" (code 6), so test uses the train rule.
    out["occupation"] = df["occupation"].map({o: i for i, o in enumerate(occ_top)}).fillna(6)
    out["sex"] = (df["sex"] == "Male").astype(int)
    out["race"] = df["race"].map({r: i for i, r in enumerate(races)}).fillna(len(races) - 1)
    out["hours"] = pd.cut(df["hours-per-week"], [0, 34, 45, 200], labels=False, right=True)
    out["income"] = (df["income"] == ">50K").astype(int)
    return out.astype(int)


if __name__ == "__main__":
    train = load("data/adult.data", 0)
    test = load("data/adult.test", 1)
    occ_top = train["occupation"].value_counts().index[:6].tolist()  # learned from TRAIN only
    races = sorted(train["race"].unique())
    assert len(races) == 5
    for name, df in [("train", train), ("test", test)]:
        enc = encode(df, occ_top, races)
        for c, d in zip(KEEP, DOMAIN):
            assert enc[c].min() >= 0 and enc[c].max() < d, (name, c)
        enc.to_csv(f"data/adult_{name}_enc.csv", index=False)
        print(name, enc.shape)
