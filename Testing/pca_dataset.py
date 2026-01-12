import pandas as pd

df2 = pd.read_csv("./dataIN/Sleep_health_and_lifestyle_dataset.csv")

bp = df2["Blood Pressure"].astype(str).str.split("/", expand=True)
df2["Systolic"] = pd.to_numeric(bp[0], errors="coerce")
df2["Diastolic"] = pd.to_numeric(bp[1], errors="coerce")

bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
df2["BMI_Code"] = df2["BMI Category"].map(bmi_map)

vars_num = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "Systolic",
    "Diastolic",
    "BMI_Code"
]

out = df2[["Person ID"] + vars_num].copy()
out = out.set_index("Person ID").dropna()
out.to_csv("./dataIN/Sleep_PCA.csv")
