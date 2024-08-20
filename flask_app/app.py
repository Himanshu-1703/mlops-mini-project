from flask import Flask, render_template, request
import mlflow
import joblib
from pathlib import Path
import dagshub
import preprocess
from sklearn.pipeline import Pipeline
from yaml import safe_load


# root path
# root_path = Path(__file__).parent.parent

# load the transformers
label_encode = joblib.load("./models/transformers/label_encoder.joblib")
bow = joblib.load("./models/transformers/bow.joblib")

# initialize dagshub
dagshub.init(repo_owner='himanshu1703', repo_name='mlops-mini-project', mlflow=True)

# set the tracking uri
mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/mlops-mini-project.mlflow")

with open("params.yaml",'r') as params:
    model_version = safe_load(params)['model']['model_version']

# load the model
model_name = "Logistic_Regression_Emotion_Detection"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri) 

# make the model pipeline
model_pipe = Pipeline(steps=[
    ('bow',bow),
    ('model',model)
])

# make flask app 
app = Flask(__name__)

# route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# route for the predict
@app.route("/predict",methods=['POST'])
def predict():
    text = str(request.form.get("text"))
    
    preproccessed_text = preprocess.normalize_text(text)
    
    prediction = model_pipe.predict([preproccessed_text])
    
    label = label_encode.inverse_transform(prediction)[0]
    
    return render_template("index.html", response=label)

# run the server
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)