from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Muat model, nama fitur, dan nilai-nilai unik
dt_model = joblib.load("model.pkl")
feature_names = joblib.load("features.pkl")
unique_values = joblib.load("unique_values.pkl")  # Nilai unik untuk setiap fitur

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

feature_descriptions = {
    "cap-surface": "Permukaan Tudung Jamur",
    "bruises": "Memar Saat Disentuh",
    "gill-attachment": "Penyambungan Insang",
    "gill-spacing": "Jarak Insang",
    "gill-size": "Ukuran Insang",
    "gill-color": "Warna Insang",
    "stalk-shape": "Bentuk Tangkai",
    "stalk-root": "Akar Tangkai",
    "stalk-surface-above-ring": "Permukaan Tangkai (Atas Cincin)",
    "stalk-surface-below-ring": "Permukaan Tangkai (Bawah Cincin)",
    "stalk-color-above-ring": "Warna Tangkai (Atas Cincin)",
    "stalk-color-below-ring": "Warna Tangkai (Bawah Cincin)",
    "veil-color": "Warna Selaput",
    "ring-number": "Jumlah Cincin",
    "ring-type": "Jenis Cincin",
    "spore-print-color": "Warna Spora",
    "population": "Populasi",
    "habitat": "Habitat",
}

@app.route("/features", methods=["GET"])
def features():
    features = [
        {
            "name": feature,
            "label": feature_descriptions.get(feature, feature),
            "values": unique_values[feature]  # Pastikan nilai unik di file `unique_values.pkl`
        }
        for feature in feature_names
    ]
    return jsonify({"features": features})



@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()

    # Validasi input
    missing_features = [feature for feature in feature_names if feature not in input_data]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400

    # Konversi input ke DataFrame
    input_df = pd.DataFrame([input_data])

    # Lakukan prediksi
    prediction_numeric = int(dt_model.predict(input_df)[0])  # Ubah dari numpy.int32 ke int
    prediction_label = "Dapat Dimakan" if prediction_numeric == 0 else "Beracun"  # Tambahkan label

    return jsonify({"prediction": prediction_label})


if __name__ == "__main__":
    app.run(debug=True)
