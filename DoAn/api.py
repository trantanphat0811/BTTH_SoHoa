import pickle
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Hàm tiền xử lý văn bản (Giữ giống `train_model.py`)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Xóa số
    text = re.sub(r'[^\w\s@.]', '', text)  # Giữ lại @ và .
    text = text.strip()
    return text


# Load mô hình và vectorizer **một lần duy nhất**
with open('spam_classifier.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('email', '')

    if not email_text:
        return jsonify({'error': 'No email provided'}), 400

    # Tiền xử lý văn bản trước khi dự đoán
    processed_text = preprocess_text(email_text)

    print(f" Input received: {email_text}")
    print(f" Processed text: {processed_text}")

    try:
        input_features = vectorizer.transform([processed_text])
        prediction = model.predict(input_features)[0]
        prediction_prob = model.predict_proba(input_features)[0]

        result = 'Ham mail' if prediction == 1 else 'Spam mail'
        confidence = max(prediction_prob) * 100  # Xác suất %

        print(f" Prediction: {result} (Confidence: {confidence:.2f}%)")

        return jsonify({'prediction': result, 'accuracy': confidence})

    except Exception as e:
        print(f"Lỗi dự đoán: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
