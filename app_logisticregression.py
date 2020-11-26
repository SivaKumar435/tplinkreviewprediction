from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.metrics import classification_report

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])



def predict():
        
                df= pd.read_csv("TPLINK_DATA1.csv", encoding="latin-1")
                from sklearn.model_selection import StratifiedShuffleSplit
                df["Rating"] = df["Rating"].astype(int)
                from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
                df["Sentiment"] = df["Rating"].apply(sentiments)
                X_train = df["Review"]
                X_train_targetSentiment =  df['Sentiment']
                X_test = df["Review"]
                X_test_targetSentiment = df['Sentiment'] 
               
                if request.method == 'POST':
                        message = request.form['message']
                        data = [message]
                        from sklearn.feature_extraction.text import TfidfTransformer
                        tfidf_transformer = TfidfTransformer(use_idf=False)
                     
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.pipeline import Pipeline
                        clf_logReg_pipe = Pipeline([("vect", CountVectorizer()), 
                            ("tfidf", TfidfTransformer()), 
                            ("clf_logReg", LogisticRegression())])
                        clf_logReg_pipe.fit(X_train, X_train_targetSentiment)
                        predictedLogReg = clf_logReg_pipe.predict(data)
                        if ('Positive' in predictedLogReg):
                                output=1
                        elif ('Negative' in predictedLogReg ):
                                output=0
                        else:
                                output=2
                X = np.array(df['Review'].values.tolist())
                y = np.array(df['Sentiment'].values.tolist())
                from sklearn.model_selection import train_test_split
                cv = CountVectorizer()
                X = cv.fit_transform(X) # Fit the Data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
              

               
                clf = LogisticRegression()
                clf.fit(X_train,y_train)
                clf.score(X_test,y_test)
                y_pred = clf.predict(X_test)
                print(classification_report(y_test, y_pred))
                print(confusion_matrix(y_test,y_pred))
                acc=accuracy_score(y_test, y_pred)*100
                acc=round(acc,2)
                print(acc)

                 


                return render_template('result.html',prediction=output,accuracy=acc)

def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"

if __name__ == '__main__':
	app.run(debug=True,host="localhost",port=5000)
