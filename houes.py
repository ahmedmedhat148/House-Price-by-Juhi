import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score
import joblib
import json

ti=pd.read_csv(r'house_prices.csv')
print(ti.shape)
print("------------------------------------------")
print(ti.info())
print("------------------------------------------")
print(ti.dtypes)
print("------------------------------------------")
print(ti.head())
print("------------------------------------------")
print(ti.tail())
print("------------------------------------------")
print(ti.duplicated().sum())
print("------------------------------------------")
print(ti.isnull().sum())
print("------------------------------------------")
ti=ti.dropna(thresh=10)
ti=ti.dropna(axis=1,how="all")
print("------------------------------------------")

ti["Bathroom"] = pd.to_numeric(ti["Bathroom"], errors="coerce")
ti["Bathroom"] = ti["Bathroom"].fillna(ti["Bathroom"].median())

ti["Balcony"] = pd.to_numeric(ti["Balcony"], errors="coerce")
ti["Balcony"] = ti["Balcony"].fillna(ti["Balcony"].median())



ti["Price (in rupees)"] = pd.to_numeric(ti["Price (in rupees)"], errors="coerce")
ti["Price (in rupees)"] = ti["Price (in rupees)"].fillna(ti["Price (in rupees)"].median())

ti["Carpet Area"] = (
    ti["Carpet Area"]
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
)
ti["Carpet Area"] = pd.to_numeric(ti["Carpet Area"], errors="coerce")
ti["Carpet Area"] = ti["Carpet Area"].fillna(ti["Carpet Area"].median())

# Super Area
ti["Super Area"] = (
    ti["Super Area"]
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
)
ti["Super Area"] = pd.to_numeric(ti["Super Area"], errors="coerce")
ti["Super Area"] = ti["Super Area"].fillna(ti["Super Area"].median())

print("------------------------------------------")


ti["Description"] = ti["Description"].fillna(ti["Description"].mode()[0])

ti["Amount(in rupees)"] = ti["Amount(in rupees)"].fillna(ti["Amount(in rupees)"].mode()[0])


ti["Status"] = ti["Status"].fillna(ti["Status"].mode()[0])

ti["Floor"] = ti["Floor"].fillna(ti["Floor"].mode()[0])


ti["Transaction"] = ti["Transaction"].fillna(ti["Transaction"].mode()[0])

ti["Furnishing"] = ti["Furnishing"].fillna(ti["Furnishing"].mode()[0])

ti["facing"] = ti["facing"].fillna(ti["facing"].mode()[0])

ti["overlooking"] = ti["overlooking"].fillna(ti["overlooking"].mode()[0])

ti["Society"] = ti["Society"].fillna(ti["Society"].mode()[0])

ti["Ownership"] = ti["Ownership"].fillna(ti["Ownership"].mode()[0])


ti["Car Parking"] = ti["Car Parking"].fillna("Unknown") 

print(ti.isnull().sum())

def parse_amount(x):
    if not isinstance(x, str):
        return None

    x = x.strip().lower()

    try:
        if "lac" in x:
            return float(x.replace("lac", "").strip()) * 1e5

        if "cr" in x:
            return float(x.replace("cr", "").strip()) * 1e7

        return float(x.replace(",", ""))

    except ValueError:
        return None


ti["price_clean"] = ti["Amount(in rupees)"].apply(parse_amount)

ti = ti.dropna(subset=["price_clean"])
print(ti["Amount(in rupees)"],ti["Price (in rupees)"])



Q1 = ti["price_clean"].quantile(0.25)
Q3 = ti["price_clean"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


ti = ti[
    (ti["price_clean"] >= lower_bound) &
    (ti["price_clean"] <= upper_bound)
]


print("After removing outliers:")
print(ti.shape)

sns.histplot(ti["price_clean"], log_scale=True)
plt.title("Price Distribution")
plt.show()

sns.boxplot(x=ti["price_clean"])
plt.title("Price Boxplot")
plt.show()

num_fu=["Balcony","Bathroom","Super Area","Carpet Area"]
cat_fu=["location","Status","Ownership","facing","Transaction","Furnishing",]


x=ti[num_fu+cat_fu]

y=ti["price_clean"]

num_pip=Pipeline(steps=[('imputer',SimpleImputer(strategy="median")),("scaler",StandardScaler())])

cat_pip=Pipeline(steps=[('imputer',SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])

pross=ColumnTransformer([('num',num_pip,num_fu),('cat',cat_pip,cat_fu)])
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
models={'LinearRegression':LinearRegression (),
        'RandomForestRegressor':RandomForestRegressor(n_estimators=5,random_state=42),
        'GradientBoostingRegressor':GradientBoostingRegressor(n_estimators=5,random_state=42)}

for n,m in models.items() :
    model_pip=Pipeline(steps=([("prep",pross),("reg",m)]))
    model_pip.fit(X_train,y_train)
    y_prad=model_pip.predict(X_test)
    print("<-------------------------------------------------------->")
    print("NAME :",n)
    print("MAE :",mean_absolute_percentage_error(y_test,y_prad))
    print("MSE :",mean_squared_error(y_test,y_prad) )
    print("R2",r2_score(y_test,y_prad))


joblib.dump(model_pip, "house_price.pkl")
print("Model saved successfully.")

loaded_model = joblib.load("house_price.pkl")


sample = X_test.iloc[[0]]
prediction = loaded_model.predict(sample)

print("Reloaded Prediction:", prediction)

locations = sorted(ti["location"].unique().tolist())

with open("locations.json", "w") as f:
    json.dump(locations, f)
print("Locations saved successfully.")