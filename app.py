from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np

from pycaret import regression as reg

app = Flask(__name__)

# Load the trained model once at startup
model = reg.load_model('deployment_20231111')

# Ensure this list matches your training pipeline feature order/names
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/health', methods=["GET"])
def health():
    return "HEALTH OK"

def _coerce_numeric(df: pd.DataFrame, numeric_cols=('age', 'bmi', 'children')) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _extract_prediction_value(df_pred: pd.DataFrame):
    """
    Works across PyCaret versions:
    - v3: 'prediction_label'
    - older: 'Label'
    Falls back to first numeric column / first cell if needed.
    """
    if 'prediction_label' in df_pred.columns:
        return df_pred.loc[0, 'prediction_label']
    if 'Label' in df_pred.columns:
        return df_pred.loc[0, 'Label']
    for c in df_pred.columns:
        if np.issubdtype(df_pred[c].dtype, np.number):
            return df_pred.loc[0, c]
    return df_pred.iloc[0, 0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Values come from HTML form; order must match `cols`
        values = [v for v in request.form.values()]
        if len(values) != len(cols):
            return render_template('home.html', pred='Error: form fields do not match expected columns.'), 400

        row = pd.DataFrame([values], columns=cols)
        row = _coerce_numeric(row)

        pred_df = reg.predict_model(model, data=row, round=0)
        print(pred_df)

        pred_value = _extract_prediction_value(pred_df)
        # Format nicely; cast to int only if it makes sense
        try:
            pred_value = int(float(pred_value))
        except Exception:
            pass

        return render_template('home.html', pred=f'Expected Bill will be ${pred_value} annually')
    except Exception as e:
        return render_template('home.html', pred=f'Error during prediction: {e}'), 500

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)

        # Accept either a dict of {feature: value} or a raw list aligned to `cols`
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, (list, tuple)):
            if len(data) != len(cols):
                return jsonify({'error': 'List payload length does not match expected columns.'}), 400
            df = pd.DataFrame([data], columns=cols)
        else:
            return jsonify({'error': 'Unsupported JSON payload format.'}), 400

        # Ensure required columns exist; fill missing with NaN
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]  # order columns
        df = _coerce_numeric(df)

        pred_df = reg.predict_model(model, data=df)
        output = _extract_prediction_value(pred_df)

        # Always return JSON
        return jsonify({'prediction': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Note: debug=True enables the reloader (app loads twice). OK for dev.
    app.run(host='0.0.0.0', debug=True)