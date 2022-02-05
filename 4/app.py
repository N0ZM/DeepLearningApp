from flask import Flask, render_template, redirect


app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/create')

@app.route('/create')
def create():
    return render_template('create.html')

@app.route('/optimize')
def optimize():
    return render_template('optimize.html')

@app.route('/evaluate')
def eveluate():
    return render_template('evaluate.html')

if __name__ == '__main__':
    app.run()
