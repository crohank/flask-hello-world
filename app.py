from flask import Flask,request,jsonify
import numpy as np
import pickle
import sklearn

flask_app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))



@flask_app.route("/")
def index():
    return "Hello world"

@flask_app.route("/predict", methods = ["POST"])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    input_query = np.array([[cgpa,iq,profile_score]])

    result = model.predict(input_query)[0]

    return jsonify({'placement':str(result)})

if __name__ == "__main__":
    flask_app.run(debug=True)
