import pickle
from flask import Flask, jsonify, request, render_template
import numpy as np


app=Flask(__name__)

model=pickle.load(open('stackingModel.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))



@app.route('/')
def home():
     return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])



@app.route('/predict',methods=['POST'])
def predict():
     data=request.form
     data= [float(data[x]) for x in data]
     scale=scaler.transform(np.array(data).reshape(1,-1))
     output=model.predict(scale)[0]
     return render_template('home.html',prediction=f'Predicted magnitude {output:.2f}')





if __name__=='__main__':
     app.run(debug=True)