from flask import Flask, render_template, redirect, request, jsonify
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/create')

@app.route('/create')
def create():
    return render_template('create.html')

@app.route('/register_network', methods=['POST'])
def register_network():
    data = request.json
    name = data['name']
    layers = data['layers']
    path = os.path.join(app.root_path, 'static/networks', name)
    if not os.path.exists(path):
        os.mkdir(path)
    fname = os.path.join(path, 'architecture.json')
    with open(fname, 'w') as f:
        json.dump(layers, f, indent=4)
    return jsonify({'result': '新しいネットワークを登録しました。'})

@app.route('/optimize')
def optimize():
    return render_template('optimize.html')

@app.route('/evaluate')
def eveluate():
    return render_template('evaluate.html')

if __name__ == '__main__':
    app.run()
