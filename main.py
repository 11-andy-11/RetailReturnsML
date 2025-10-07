#Importul bibliotecilor necesare pentru prelucrarea datelor, antrenarea modelului si generarea de grafice
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

#Citirea si pregatirea initiala a datelor 
#Acest encoding='ISO-8859-1' este folosit pentru a interpreta corect caracterele ă, î etc.
df = pd.read_csv("Brand_Sales_AdSpend_Data.csv", encoding='ISO-8859-1')  # Citim fisierul CSV cu datele
df['Country'] = df['Country'].replace('SA', 'South Africa')  # Inlocuim prescurtarea 'SA' cu numele complet South Africa

#Introducere zgomot artificial pe 'Net Sales' 
np.random.seed(42)  # Setam seed pentru a avea rezultate reproductibile
std_dev_net_sales = df['Net Sales'].std()  # Calculam deviatie standard a coloanei
noise = np.random.normal(loc=0, scale=0.05 * std_dev_net_sales, size=len(df))  # Generam zgomot gaussian (5%)
df['Net Sales'] += noise  # Adaugam zgomotul in coloana Net Sales

# Parametrul errors='coerce' asigură că valorile invalide devin NaT (Not a Time)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extragem anul, luna și ziua din coloana 'Date'
df['Year'] = df['Date'].dt.year   # Creează o coloană nouă cu anul
df['Month'] = df['Date'].dt.month # Creează o coloană nouă cu luna
df['Day'] = df['Date'].dt.day     # Creează o coloană nouă cu ziua

# Introducem artificial valori lipsă în coloana 'Order Count'
num_missing = int(0.05 * len(df))  # 5% din total
missing_indices = np.random.choice(df.index, num_missing, replace=False)
df.loc[missing_indices, 'Order Count'] = np.nan

# (Opțional) Ștergem coloana originală 'Date' dacă nu mai este necesară
df.drop(columns=['Date'], inplace=True)

# Afișăm primele 5 rânduri din noile coloane pentru a verifica extragerea
print("Primele 5 rânduri din coloanele extrase:")
print(df[['Year', 'Month', 'Day']].head())

#Selectam coloanele de intrare (X) si coloana tinta (y)
X = df[['Brand Name', 'Country', 'Gross Sales', 'Net Sales', 'Total Sales', 'Total Ad Spend', 'Order Count']]
y = df['Return Amount']  # Return Amount este coloana pe care vrem sa o prezicem

#EDA pe datele brute
#Impartim datele in set de antrenare si test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X.copy(), y, test_size=0.2, random_state=42)
train_data_raw = pd.concat([X_train_raw, y_train], axis=1)
test_data_raw = pd.concat([X_test_raw, y_test], axis=1)
train_df = train_data_raw.copy()
test_df = test_data_raw.copy()

#Selectam variabilele numerice si categorice din setul de antrenament
numerice = train_df.select_dtypes(include=["int64", "float64"])
categorice = train_df.select_dtypes(include=["object", "category"])

#Afisam informatii generale despre date si valori lipsa
print("Informatii generale (train):")
print(train_df.info())
missing = train_df.isnull().sum().to_frame("Missing Values")
missing["%"] = 100 * missing["Missing Values"] / len(train_df)
print("Valori lipsa (train):\n", missing)

#Statistici descriptive pentru variabile numerice
print("Statistici descriptive (train - numerice):")
print(train_df.describe())
print("Statistici descriptive (test - numerice):")
print(test_df.describe())

#Cream directorul si salvam histogramelor variabilelor numerice
eda_dir = "histograme"
os.makedirs(eda_dir, exist_ok=True)
for col in numerice.columns:
    plt.hist(train_df[col])
    plt.title(f"Histograma - {col}")
    plt.xlabel(col)
    plt.ylabel("Frecventa")
    plt.savefig(f"{eda_dir}/histograma_{col}.png")
    plt.close()

#Cream countplot pentru variabilele categorice
eda_dir = "countplot"
os.makedirs(eda_dir, exist_ok=True)
for col in categorice.columns:
    sns.countplot(data=train_df, x=col)
    plt.title(f"Countplot - {col}")
    plt.xticks(rotation=45)
    plt.savefig(f"{eda_dir}/countplot_{col}.png")
    plt.close()

#Detectam outlieri folosind metoda IQR (Interquartile Range)
print("Detectare outlieri folosind IQR")
for col in numerice.columns:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[col] < lower) | (train_df[col] > upper)]

    print(f"  Coloana: {col}")
    print(f"  Q1 (25%): {Q1:.2f}")
    print(f"  Q3 (75%): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Limita inferioară: {lower:.2f}")
    print(f"  Limita superioară: {upper:.2f}")
    print(f"  Numar outlieri: {len(outliers)} din {len(train_df)} ({100 * len(outliers) / len(train_df):.2f}%)\n")

#Cream heatmap pentru corelatii intre variabile numerice
heatmap_dir = "heatmap"
os.makedirs(heatmap_dir, exist_ok=True)
plt.figure(figsize=(12, 8))  # Setam dimensiunea graficului
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corelatii - Variabile numerice")
plt.tight_layout()
plt.savefig(f"{heatmap_dir}/heatmap_corelatii.png")
plt.close()

#Scatter plot intre fiecare variabila numerica si targetul Return Amount
rel_dir = "grafice_relatii_target"
os.makedirs(rel_dir, exist_ok=True)
variabile_numerice = ["Gross Sales", "Net Sales", "Total Sales", "Total Ad Spend", "Order Count"]
variabile_dummy = [col for col in train_df.columns if col.startswith("Country_")]
for col in variabile_numerice:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=train_df, x=col, y="Return Amount", alpha=0.6)
    plt.title(f"Scatter Plot - {col} vs Return Amount")
    plt.xlabel(col)
    plt.ylabel("Return Amount")
    plt.tight_layout()
    plt.savefig(f"{rel_dir}/scatter_{col}_vs_return.png")
    plt.close()

#Codificare Brand Name dupa EDA
brand_encoder = LabelEncoder()
df['Brand Name'] = brand_encoder.fit_transform(df['Brand Name'])  # Convertim brandurile in numere

#One-hot encoding (acum, dupa EDA)
X = df[['Brand Name', 'Country', 'Gross Sales', 'Net Sales', 'Total Sales', 'Total Ad Spend', 'Order Count']]
X = pd.get_dummies(X, columns=['Country'], drop_first=True)  # Codificam tarile in variabile binare

#Standardizare
scaler = StandardScaler()
X[['Gross Sales', 'Net Sales', 'Total Sales', 'Total Ad Spend', 'Order Count']] = scaler.fit_transform(
    X[['Gross Sales', 'Net Sales', 'Total Sales', 'Total Ad Spend', 'Order Count']])  # Standardizam pentru model

#Impartire dupa standardizare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Train/test split

#Salvarea in fisiere CSV 
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv("train.csv", index=False)  # Salvam setul de antrenare
test_data.to_csv("test.csv", index=False)    # Salvam setul de test

#Antrenarea modelului
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Initializam modelul Random Forest
model.fit(X_train, y_train)  # Antrenam modelul pe datele de antrenament

#Predictii si evaluare 
y_pred = model.predict(X_test)  # Facem predictii pe setul de test
mae = mean_absolute_error(y_test, y_pred)  # Calculam eroarea medie absoluta
print(f"MAE: {mae:.2f}")

#Grafic al erorilor reziduale 
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')  # Linie de referinta pe 0
plt.title("Grafic de Erori Reziduale")
plt.xlabel("Valori prezise")
plt.ylabel("Erori reziduale")
plt.savefig("erori_reziduale.png")
plt.close()

#Salvam modelul si transformarile 
joblib.dump(model, "model.pkl")               # Salvam modelul antrenat
joblib.dump(scaler, "scaler.pkl")             # Salvam standardizatorul
joblib.dump(brand_encoder, "brand_encoder.pkl")  # Salvam codificatorul de brand
joblib.dump(X.columns.tolist(), "model_features.pkl")  # Salvam numele caracteristicilor

