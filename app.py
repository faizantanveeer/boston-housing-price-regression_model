from flask import Flask, request, jsonify, url_for, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('linear_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_api', methods=['POST'])    
def predict_api():
    data = request.json['data']
    print(data)
    new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify({'output': output[0]})


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    print(data)
    input=scalar.transform(np.array(data).reshape(1, -1))
    print(input)
    prediction = model.predict(input)[0]
    print(prediction)
    return render_template('index.html', prediction_text='Predicted Price is {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)


    