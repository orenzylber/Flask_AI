from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from csv import writer

api = Flask(__name__)

@api.route('/')
def hello():
    return render_template('index.html')

@api.route('/learn', methods=['get'])
def learn():
    return render_template('real.html')

@api.route('/predict', methods=['post'])
def get_data():

    model=joblib.load('our_pridction.joblib')
    
    # geting data from the user
    age=request.form.get('age')
    gender=request.form.get('gender')
    
    predictions=model.predict([[age, gender]])
    print(predictions)
    
    return render_template('predict.html', msg=predictions)

@api.route('/real', methods=['post'])
def real():
    age=request.form.get('age')
    gender=request.form.get('gender')
    genre=request.form.get('genre')
    print(age, gender, genre)

    with open('music.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([age, gender, genre])
        f_object.close()        

    music_dt  =pd.read_csv('music.csv')

    # #fit - training the model
    X=music_dt.drop(columns=['genre'])
    Y=music_dt['genre']
    model = DecisionTreeClassifier()
    model.fit(X,Y)

    # #save the model
    joblib.dump(model, 'our_pridction.joblib')
    
    return render_template('index.html')

if __name__ == '__main__':
    api.run(debug=True)
