from flask import Flask, request, jsonify
from model import model_train, model_predict
from logger import update_train_log, update_predict_log

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data_dir = request.json.get('data_dir')
    test = request.json.get('test', False)
    model_train(data_dir, test)
    return jsonify({"status": "training started"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    country = request.json.get('country')
    year = request.json.get('year')
    month = request.json.get('month')
    day = request.json.get('day')
    test = request.json.get('test', False)
    result = model_predict(country, year, month, day, test=test)
    
    # Convert NumPy arrays to lists for JSON serialization
    if 'y_pred' in result:
        result['y_pred'] = result['y_pred'].tolist()
    if 'y_proba' in result and result['y_proba'] is not None:
        result['y_proba'] = result['y_proba'].tolist()
    
    return jsonify(result), 200

@app.route('/logs', methods=['GET'])
def logs():
    log_type = request.args.get('type', 'train')
    test = request.args.get('test', 'False').lower() == 'true'
    log_file = "logs/train.log" if log_type == 'train' else "logs/predict.log"
    if test:
        log_file = log_file.replace(".log", "-test.log")
    with open(log_file, 'r') as file:
        logs = file.readlines()
    return jsonify({"logs": logs}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)