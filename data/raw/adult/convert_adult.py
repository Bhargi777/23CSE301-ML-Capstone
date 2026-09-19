import pandas as pd
from pathlib import Path

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

folder = Path(__file__).parent

train = pd.read_csv(folder / "adult.data", header=None, names=COLUMNS, skipinitialspace=True)
train = train.dropna(how="all")

test = pd.read_csv(folder / "adult.test", header=None, names=COLUMNS, skipinitialspace=True, skiprows=1)
test["income"] = test["income"].str.rstrip(".")

train.to_csv(folder / "adult_train.csv", index=False)
test.to_csv(folder / "adult_test.csv", index=False)

print("adult_train.csv:", train.shape)
print("adult_test.csv:", test.shape)
