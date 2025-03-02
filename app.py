from flask import Flask, request, jsonify
from model import predict_fake_news
from etl import preprocess_text

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Predict fake or real
    result = predict_fake_news(processed_text)

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
