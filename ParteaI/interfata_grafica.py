import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Incarca modelul si transformarile salvate
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
brand_encoder = joblib.load("brand_encoder.pkl")
model_columns = joblib.load("model_features.pkl")

# Incarca datele de test pentru scatter plot
df_test = pd.read_csv("test.csv")
y_test = df_test["Return Amount"]
X_test = df_test.drop(columns=["Return Amount"])

def predict(brand_name, country, gross, net, total, ad_spend, orders):
    # Verificare: toate valorile numerice sunt 0
    if all(v == 0 for v in [gross, net, total, ad_spend, orders]):
        raise gr.Error("Toate valorile numerice sunt 0. Te rog introdu valori reale pentru predicție.")
    # Verificare: valorile introduse sunt prea mici pentru a fi relevante
    if gross < 5000 or net < 2500 or total < 1000 or ad_spend < 1000 or orders < 50:
        raise gr.Error("Ai introdus valori prea mici. Te rugăm să introduci cifre realiste, conform setului de date.")
    # Construim DataFrame-ul cu valorile introduse
    input_df = pd.DataFrame([{
        "Brand Name": brand_name,
        "Country": country,
        "Gross Sales": gross,
        "Net Sales": net,
        "Total Sales": total,
        "Total Ad Spend": ad_spend,
        "Order Count": orders
    }])

    # Codificare și preprocesare
    input_df["Brand Name"] = brand_encoder.transform([brand_name])
    input_df = pd.get_dummies(input_df, columns=["Country"], drop_first=True)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]
    numeric_cols = ['Gross Sales', 'Net Sales', 'Total Sales', 'Total Ad Spend', 'Order Count']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predicție
    pred = model.predict(input_df)[0]

    # Generare grafic
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, label="Date reale")
    ax.scatter(pred, pred, color='red', label="Predicție nouă", s=100)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valori reale")
    ax.set_ylabel("Valori prezise")
    ax.set_title("Valori reale vs. prezise")
    ax.legend()
    plt.tight_layout()

    return round(pred, 2), fig

# Listele pentru dropdown
brand_list = list(brand_encoder.classes_)
country_list = pd.read_csv("Brand_Sales_AdSpend_Data.csv", encoding="ISO-8859-1")["Country"].replace("SA", "South Africa").unique().tolist()

# Interfata grafica Gradio
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=brand_list, label="Brand Name"),
        gr.Dropdown(choices=country_list, label="Country"),
        gr.Number(label="Gross Sales"),
        gr.Number(label="Net Sales"),
        gr.Number(label="Total Sales"),
        gr.Number(label="Total Ad Spend"),
        gr.Number(label="Order Count"),
    ],
    outputs=[
        gr.Number(label="Return Amount prezis"),
        gr.Plot(label="Scatter Plot: Reale vs Prezise")
    ],
    title="Predicție Return Amount",
    description="Introdu valorile necesare pentru a estima Return Amount",
    submit_btn="Predicție"  # Aici setez textul butonului
)

# Lansam aplicatia
interface.launch()
