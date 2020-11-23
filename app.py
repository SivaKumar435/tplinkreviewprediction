from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])



def predict():
        
                df= pd.read_csv("TPLINK_DATA1.csv", encoding="latin-1")
                from sklearn.model_selection import StratifiedShuffleSplit
                df["Rating"] = df["Rating"].astype(int)
                
                df["Sentiment"] = df["Rating"].apply(sentiments)
                df["Sentiment"] = df["Rating"].apply(sentiments)
                df['label'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0,'Neutral':2})
                df['label'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0,'Neutral':2})
                X_train = df["Review"]
                X_train_targetSentiment =  df['label']
                X_test = df["Review"]
                X_test_targetSentiment = df['label'] 
	
                if request.method == 'POST':
                        message = request.form['message']
                        data = [message]
                        from sklearn.feature_extraction.text import TfidfTransformer
                        tfidf_transformer = TfidfTransformer(use_idf=False)
                        from sklearn.pipeline import Pipeline
                        import numpy as np
                        from sklearn.ensemble import RandomForestClassifier
                        clf_randomForest_pipe = Pipeline([("vect", CountVectorizer()), 
                                  ("tfidf", TfidfTransformer()), 
                                  ("clf_randomForest", RandomForestClassifier())
                                 ])
                        clf_randomForest_pipe.fit(X_train, X_train_targetSentiment)

                        predictedRandomForest = clf_randomForest_pipe.predict(X_test)[0]
                        np.mean(predictedRandomForest == X_test_targetSentiment)
                        from flask import jsonify
                        output = {'predictedRandomForest': int(predictedRandomForest[0])}
                return render_template('result.html',prediction = predictedRandomForest)

def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"

if __name__ == '__main__':
	app.run(debug=True)
