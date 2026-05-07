# ================================================
# E-Commerce Quantity Prediction - ML Training
# Features: UnitPrice, Country, CustomerID,
#           InvoiceNo (numeric), Month, Day, Hour
# ================================================

import os
import glob
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# Step 1: Kaggle API Authentication
# -------------------------------
os.environ['KAGGLE_USERNAME'] = 'hamzaabbasikaggle'
os.environ['KAGGLE_KEY']      = 'KGAT_afe811e70d49afc81c558617998413e0'

# -------------------------------
# Step 2: Download dataset if needed
# -------------------------------
dataset_name = 'carrie1/ecommerce-data'
data_path    = 'data'
os.makedirs(data_path, exist_ok=True)

csv_files = glob.glob(os.path.join(data_path, '*.csv'))
if not csv_files:
    print("Downloading dataset from Kaggle...")
    os.system(f'kaggle datasets download -d {dataset_name} -p {data_path} --unzip')
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
else:
    print("Dataset already exists locally. Skipping download.")

if not csv_files:
    raise FileNotFoundError("No CSV file found!")

# -------------------------------
# Step 3: Load CSV
# -------------------------------
csv_file = csv_files[0]
data = pd.read_csv(csv_file, encoding="latin1").head(5000)
print("Dataset loaded successfully!")
print(data.head())
print("\nColumns:", data.columns.tolist())
print("\nDtypes:\n", data.dtypes)

# -------------------------------
# Step 4: Feature Engineering
# -------------------------------

# 4a. Parse InvoiceDate into datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')

# 4b. Extract Month, Day, Hour from InvoiceDate
data['Month'] = data['InvoiceDate'].dt.month
data['Day']   = data['InvoiceDate'].dt.day
data['Hour']  = data['InvoiceDate'].dt.hour

# 4c. Drop cancellation invoices (InvoiceNo starting with 'C')
data = data[~data['InvoiceNo'].astype(str).str.startswith('C')]

# 4d. Convert InvoiceNo to numeric
data['InvoiceNo_num'] = pd.to_numeric(data['InvoiceNo'], errors='coerce')

# 4e. Define features and target
NUMERIC_FEATURES  = ['UnitPrice', 'CustomerID', 'InvoiceNo_num', 'Month', 'Day', 'Hour']
CATEGORY_FEATURES = ['Country']
ALL_FEATURES      = NUMERIC_FEATURES + CATEGORY_FEATURES
TARGET            = 'Quantity'

# 4f. Keep only needed columns
data_clean = data[ALL_FEATURES + [TARGET]].copy()

# 4g. Debug: NaN counts before cleaning
print("\n--- NaN counts BEFORE cleaning ---")
print(data_clean.isnull().sum())

# 4h. Drop rows with any NaN
data_clean = data_clean.dropna(subset=ALL_FEATURES + [TARGET])

# 4i. Remove nonsensical rows (returns, bad entries)
data_clean = data_clean[
    (data_clean['Quantity']   > 0) &
    (data_clean['UnitPrice']  > 0) &
    (data_clean['CustomerID'] > 0)
]

data_clean = data_clean.reset_index(drop=True)

# 4j. Confirm no NaNs remain
print("\n--- NaN counts AFTER cleaning ---")
print(data_clean.isnull().sum())
print(f"\nClean dataset shape: {data_clean.shape}")

# -------------------------------
# Step 5: Features & Target Split
# -------------------------------
X = data_clean[ALL_FEATURES]
y = data_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")

# -------------------------------
# Step 6: Build Pipeline
# -------------------------------

# Numeric features → StandardScaler (improves linear regression performance)
# Country         → OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),                        NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORY_FEATURES),
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor',    LinearRegression())
])

# -------------------------------
# Step 7: Train & Save
# -------------------------------
FORCE_RETRAIN = True
model_path    = "ecommerce_model.pkl"

if os.path.exists(model_path) and not FORCE_RETRAIN:
    print("\nLoading existing pipeline...")
    pipeline = joblib.load(model_path)
else:
    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to '{model_path}'!")

# -------------------------------
# Step 8: Evaluate
# -------------------------------
y_pred = pipeline.predict(X_test)

print("\n--- Model Evaluation (unseen test data) ---")
print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
print(f"MAE      : {mean_absolute_error(y_test, y_pred):.4f}")

# -------------------------------
# Step 9: Sanity Check (mirrors FastAPI input exactly)
# -------------------------------
sample = pd.DataFrame([{
    "UnitPrice"    : 2.55,
    "CustomerID"   : 17850.0,
    "InvoiceNo_num": 536365,
    "Month"        : 12,
    "Day"          : 1,
    "Hour"         : 8,
    "Country"      : "United Kingdom"
}])

pred = pipeline.predict(sample)
print(f"\nSample prediction: {pred[0]:.2f} units")




































































































# # Step 0: Install required packages
# # !pip install kaggle scikit-learn pandas joblib

# import os
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import joblib
# import glob
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# # -------------------------------
# # Step 1: Kaggle API Authentication
# # -------------------------------
# os.environ['KAGGLE_USERNAME'] = 'hamzaabbasikaggle'
# os.environ['KAGGLE_KEY'] = 'KGAT_afe811e70d49afc81c558617998413e0'

# # -------------------------------
# # Step 2: Download dataset ONLY if not already downloaded
# # -------------------------------
# dataset_name = 'carrie1/ecommerce-data'
# data_path = 'data'

# os.makedirs(data_path, exist_ok=True)

# # Check if CSV already exists
# csv_files = glob.glob(os.path.join(data_path, '*.csv'))

# if not csv_files:
#     print("Dataset not found locally. Downloading from Kaggle...")

#     os.system(
#         f'kaggle datasets download -d {dataset_name} -p {data_path} --unzip'
#     )

#     print("Download completed!")

# else:
#     print("Dataset already exists locally. Skipping download.")

# # -------------------------------
# # Step 3: Load CSV
# # -------------------------------
# if not csv_files:
#     csv_files = glob.glob(os.path.join(data_path, '*.csv'))

# if not csv_files:
#     raise FileNotFoundError("No CSV file found in the dataset folder!")

# csv_file = csv_files[0]
# data = pd.read_csv(csv_file, encoding="latin1").head(5000)

# print("Dataset loaded successfully!")
# print(data.head())

# # -------------------------------
# # Step 4: Prepare Features (X) and Target (y)
# # -------------------------------

# # Let's select only numeric columns suitable for regression

# # Select relevant columns
# # data_selected = data[['Quantity', 'UnitPrice', 'Country', 'CustomerID']].copy()

# # # Remove missing values
# # data_selected = data_selected.dropna()

# # # Encode ONLY Country
# # data_encoded = pd.get_dummies(
# #     data_selected,
# #     columns=['Country'],
# #     drop_first=True
# # )
# #-----------------------------------------------------------------------
# # features
# X = data[['UnitPrice', 'CustomerID', 'Country']]
# y = data['Quantity']

# # preprocessing
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country']),
#         ('num', 'passthrough', ['UnitPrice', 'CustomerID'])
#     ]
# )

# #----------------------------------------------------------------

# # # Features
# # X = data_encoded.drop('Quantity', axis=1)

# # # Target
# # y = data_encoded['Quantity']

# # print("\nFeatures shape:", X.shape)
# # print("Target shape:", y.shape)

# # -------------------------------
# # Step 5: Train Linear Regression
# # -------------------------------


# FORCE_RETRAIN = True

# model_path = "ecommerce_model.pkl"

# # -------------------------------
# # Build pipeline (ONLY ON TRAIN)
# # -------------------------------
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country']),
#         ('num', 'passthrough', ['UnitPrice', 'CustomerID'])
#     ]
# )

# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', LinearRegression())
# ])


# if os.path.exists("ecommerce_model.pkl") and not FORCE_RETRAIN:
#     print("Loading existing model...")

#     model = joblib.load("ecommerce_model.pkl")

# else:
#     print("Training model...")

#     model = LinearRegression()
#     model.fit(X, y)

#     joblib.dump(model, "ecommerce_model.pkl")

#     print("Model trained and saved!")
    

# # -------------------------------
# # Step 6: R2 Score, and MAE
# # -------------------------------

# from sklearn.metrics import r2_score, mean_absolute_error

# y_pred = model.predict(X)

# print("R2 Score:", r2_score(y, y_pred))
# print("MAE:", mean_absolute_error(y, y_pred))

# #_________________________________________________________

