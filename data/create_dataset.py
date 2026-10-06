import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CREATE SYNTHETIC PLACEMENT DATASET
# ============================================================

np.random.seed(42)

# Number of students
n = 1000

# ============================================================
# 1. ACADEMIC FEATURES
# ============================================================

data = {}

for sem in range(1, 9):
    data[f"SGPA_Sem{sem}"] = np.round(
        np.clip(
            np.random.normal(7.5, 1.0, n),
            5.0,
            10.0
        ),
        2
    )

data["CGPA"] = np.round(
    np.clip(
        np.mean(
            [data[f"SGPA_Sem{i}"] for i in range(1, 9)],
            axis=0
        ) + np.random.normal(0, 0.25, n),
        5.0,
        10.0
    ),
    2
)

data["AttendancePercent"] = np.round(
    np.clip(
        np.random.normal(82, 10, n),
        50,
        100
    ),
    2
)

# ============================================================
# 2. ACTIVITIES
# ============================================================

data["Internships"] = np.random.randint(0, 4, n)

data["Projects"] = np.random.randint(0, 6, n)

data["Workshops"] = np.random.randint(0, 8, n)

data["Certifications"] = np.random.randint(0, 6, n)

data["Publications"] = np.random.randint(0, 4, n)

data["AptitudeTestScore"] = np.round(
    np.clip(
        np.random.normal(70, 15, n),
        30,
        100
    ),
    2
)

data["SoftSkillsRating"] = np.round(
    np.clip(
        np.random.normal(7, 1.5, n),
        1,
        10
    ),
    2
)

data["CodingTestScore"] = np.round(
    np.clip(
        np.random.normal(68, 16, n),
        20,
        100
    ),
    2
)

data["MockInterviewScore"] = np.round(
    np.clip(
        np.random.normal(7, 1.5, n),
        1,
        10
    ),
    2
)

# ============================================================
# 3. CATEGORICAL FEATURES
# ============================================================

data["Gender"] = np.random.choice(
    ["Male", "Female"],
    n
)

data["City"] = np.random.choice(
    [
        "Hyderabad",
        "Vijayawada",
        "Chennai",
        "Bangalore",
        "Mumbai"
    ],
    n
)

data["CollegeTier"] = np.random.choice(
    [1, 2, 3],
    n,
    p=[0.25, 0.50, 0.25]
)

data["Stream"] = np.random.choice(
    [
        "CSE",
        "ECE",
        "EEE",
        "MECH",
        "CIVIL",
        "IT"
    ],
    n
)

data["Specialisation"] = np.random.choice(
    [
        "AI",
        "Data Science",
        "Web Development",
        "Cyber Security",
        "Cloud Computing",
        "Embedded Systems"
    ],
    n
)

data["ExtraCurricular"] = np.random.randint(0, 10, n)

# ============================================================
# 4. HOSTEL AND BACKLOG
# ============================================================

data["Hostel"] = np.random.choice(
    ["Yes", "No"],
    n
)

data["HistoryOfBacklogs"] = np.random.choice(
    ["Yes", "No"],
    n,
    p=[0.20, 0.80]
)

# ============================================================
# 5. PLACEMENT SCORE
# ============================================================

placement_score = (
    0.35 * data["CGPA"]
    + 0.08 * data["AttendancePercent"]
    + 0.8 * data["Internships"]
    + 0.4 * data["Projects"]
    + 0.25 * data["Certifications"]
    + 0.02 * data["AptitudeTestScore"]
    + 0.03 * data["CodingTestScore"]
    + 0.15 * data["SoftSkillsRating"]
    + 0.15 * data["MockInterviewScore"]
    + 0.1 * data["ExtraCurricular"]
)

# Add random noise
placement_score += np.random.normal(0, 3, n)

# ============================================================
# 6. CREATE ~67% PLACED / ~33% NOT PLACED
# ============================================================

# 33rd percentile means approximately 67% will be above threshold
threshold = np.percentile(placement_score, 33)

data["PlacementStatus"] = np.where(
    placement_score >= threshold,
    "Placed",
    "Not Placed"
)

# ============================================================
# 7. SALARY PACKAGE
# ============================================================

salary = (
    2.5
    + 0.7 * data["CGPA"]
    + 0.08 * data["Internships"]
    + 0.15 * data["Projects"]
    + 0.015 * data["AptitudeTestScore"]
    + 0.02 * data["CodingTestScore"]
    + np.random.normal(0, 1.0, n)
)

salary = np.clip(
    salary,
    2.5,
    25
)

# Salary only for placed students
data["Salary Package"] = np.round(
    np.where(
        data["PlacementStatus"] == "Placed",
        salary,
        0
    ),
    2
)

# ============================================================
# 8. ANOMALY FLAG
# ============================================================

data["IsAnomaly"] = np.random.choice(
    [0, 1],
    n,
    p=[0.95, 0.05]
)

# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)

# ============================================================
# 10. ADD MISSING VALUES
# ============================================================

missing_columns = [
    "Workshops",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]

for column in missing_columns:

    missing_indices = np.random.choice(
        df.index,
        size=int(0.03 * n),
        replace=False
    )

    df.loc[
        missing_indices,
        column
    ] = np.nan

# ============================================================
# 11. SAVE DATASET
# ============================================================

output_path = Path(__file__).parent / "placement_data.csv"

df.to_csv(
    output_path,
    index=False
)

# ============================================================
# 12. DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("PLACEMENT DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nFile saved at:")
print(output_path)

print(f"\nDataset shape:")
print(df.shape)

print("\nPlacement Status:")
print(df["PlacementStatus"].value_counts())

print("\nPlacement Percentage:")
print(
    (df["PlacementStatus"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset creation completed successfully!")