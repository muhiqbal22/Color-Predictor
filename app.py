from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Muat model dan encoder
model = joblib.load('color_prediction_model.pkl')
encoder = joblib.load('encoder.pkl')

# Muat data warna untuk digunakan dalam pencarian nama warna
colors_df = pd.read_csv('colors.csv')

@app.route('/')
def index():
    color_names = colors_df['color_name'].tolist()
    return render_template('index.html', color_names=color_names)

@app.route('/predict', methods=['POST'])
def predict():
    color_name = request.form['color_name']
    print(f"Received color name: {color_name}")  # Logging
    try:
        # Encode nama warna
        if color_name not in encoder.classes_:
            raise ValueError("Color name not found in encoder classes.")
        
        # Dapatkan RGB dari DataFrame
        color_row = colors_df[colors_df['color_name'] == color_name]
        if color_row.empty:
            raise ValueError("Color name not found in the dataset.")
        
        r = int(color_row['r'].values[0])
        g = int(color_row['g'].values[0])
        b = int(color_row['b'].values[0])

        # Encode nama warna
        color_name_encoded = encoder.transform([color_name]).reshape(-1, 1)
        print(f"Encoded color name: {color_name_encoded}")  # Logging
        
        # Menggabungkan fitur untuk prediksi
        X_input = [[color_name_encoded[0][0], r, g, b]]
        print(f"Input to model: {X_input}")  # Logging
        
        # Prediksi kode warna
        color_code_pred = model.predict(X_input)[0]
        print(f"Predicted color code: {color_code_pred}")  # Logging
        
        return jsonify({'color_code': color_code_pred, 'rgb': {'r': r, 'g': g, 'b': b}})
    except Exception as e:
        print(f"Error: {e}")  # Logging
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
