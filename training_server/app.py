from flask import Flask, request, jsonify, send_from_directory
from lookup import lookup
import json
import base64
import io
import codecs
import os
app = Flask(__name__)


@app.route('/api/train', methods=['POST'])
def post():
    data = json.loads(request.get_data())
    for annotation in data['annotations']:
        img = base64.b64decode(annotation['document'][2:-1])
        annotation['document'] = io.BytesIO(img)
    result, dimensions = lookup(data['annotations'])
    return jsonify(
        result=result,
        dimensions=dimensions
    )


@app.route('/api/download-model', methods=['GET'])
def get():
    return send_from_directory(directory="trained_models/", path="model.pth", as_attachment=True)


@app.route('/')
def main():
    return 'Server to Train AKSHRA - Annotator Models.'

if __name__ == '__main__':
    app.run(debug=True)