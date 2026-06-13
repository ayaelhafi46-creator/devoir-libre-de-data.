# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# ==============================
# 2. LOAD DATA
# ==============================
titanic = pd.read_csv("titanic_survival.csv")

print(titanic.head())
print(titanic.shape)
print(titanic.columns)

# ==============================
# 3. MISSING VALUES
# ==============================

# age
age = titanic["age"]
missing_values = age[age.isnull()]
missing_values_count = len(missing_values)
print("Missing age:", missing_values_count)

# cabin
print("Missing cabin:", titanic["cabin"].isnull().sum())

# all columns
print("Missing all:\n", titanic.isnull().sum())

# ==============================
# 4. HANDLE MISSING VALUES
# ==============================

# drop rows with embarked missing
titanic = titanic.dropna(subset=["embarked"])

# drop cabin column
titanic = titanic.drop("cabin", axis=1)

# fill age with mean
titanic["age"].fillna(titanic["age"].mean(), inplace=True)

# fill embarked with most frequent
titanic["embarked"].fillna(titanic["embarked"].mode()[0], inplace=True)

# ==============================
# 5. ENCODING
# ==============================

# embarked -> OneHot
titanic = pd.get_dummies(titanic, columns=["embarked"])

# sex -> LabelEncoder
le = LabelEncoder()
titanic["sex"] = le.fit_transform(titanic["sex"])

# ==============================
# 6. FEATURE SELECTION
# ==============================
data = titanic[["pclass", "sex", "age", "fare", "survived"]]

# ==============================
# 7. TRAIN TEST SPLIT
# ==============================
X = data.drop("survived", axis=1)
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("Train/Test sizes:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ==============================
# 8. FEATURE SCALING
# ==============================

# StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# MinMaxScaler
mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train)
X_test_mm = mm.transform(X_test)

print("Scaling done successfully!")