from flask import Flask,render_template,request
import logging
import sys
logging.basicConfig(level=logging.DEBUG)

app= Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    print('rendering template', file=sys.stderr)
    return render_template('index.html');

@app.route('/predict',methods=['POST'])
def predict():
    print(request.form, file=sys.stderr)
    q1=request.form['q1']
    q2=request.form['q2']
    print(q1,q2, file=sys.stderr)
    return 'hello'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
