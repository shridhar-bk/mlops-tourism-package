
# for data manipulation
import pandas as pd

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = "ProdTaken"

# List of numerical features in the dataset
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

# List of categorical features in the dataset
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

# Remove the unnecessary identifier column
df = df.drop(columns=["CustomerID"])

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
