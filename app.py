# Import libraries
from flask import *
import pickle
import pandas as pd

# Load Data from 1st csv
cleaned = pd.read_csv("data/cleaned.csv").set_index(['Unnamed: 0'])
cleaned.index.name = None
# Load Data from 2nd csv
cleaned_df = pd.read_csv("data/cleaned_df.csv").set_index(['Unnamed: 0'])
cleaned_df.index.name = None

# Flask Setup
app = Flask(__name__)

# Load Models 
with open('models/svm_model.pkl', 'rb') as f:
    svm_model_pickle = pickle.load(f)
with open('models/svm_fasttext_model.pkl', 'rb') as f:
    svm_fasttext_model = pickle.load(f)

# SVM
def svm_model(x):
    predicted_list = []
    for i in x:
        predicted= svm_model_pickle.predict([i])
        s = cleaned[cleaned["Target"] == predicted[0]]['Line item'].to_string()
        output = ''.join([i for i in s if not i.isdigit()]).strip()
        predicted_list.append(output)        
    return predicted_list

# SVM Fasttext   
def svm_fasttext_prediction(y):
    predicted_list_svmfast =[]  
    for i in y:
        if i in list(cleaned_df['value']):
            predicted= svm_model_pickle.predict([i])
            s = cleaned[cleaned["Target"] == predicted[0]]['Line item'].to_string()
            output = ''.join([i for i in s if not i.isdigit()]).strip()
        else:
            n = 0
            semantically_similar_words = {words: [item[0] for item in svm_fasttext_model.wv.most_similar([words], topn=5)]
                      for words in [i]}
            s = cleaned_df[cleaned_df['value'] == semantically_similar_words[i][n]]["Line item"].drop_duplicates().to_string()
            while s == "Series([], )":
                n+=1
                s = cleaned_df[cleaned_df['value'] == semantically_similar_words[i][n]]["Line item"].drop_duplicates().to_string()
            output = ''.join([i for i in s if not i.isdigit()]).strip()

        predicted_list_svmfast.append(output)
    return(predicted_list_svmfast)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        ## Get user form input and process it
        user_input = request.form['Input']
        words_list = user_input.split(",")
        words_list = [x.strip(' ') for x in words_list]

        ## SVM
        svm_predicted = svm_model(words_list)
        ## SVM + Fast Text
        svm_fasttext_predicted = svm_fasttext_prediction(words_list)

        results_table = pd.DataFrame(words_list,columns =['Input'])
        results_table["Model 1 - SVM Prediction"] = svm_predicted
        results_table["Model 2 - Fasttext Prediction"] = svm_fasttext_predicted
        
        return render_template('index.html', tables=[results_table.to_html(classes='data',header="true")])

if __name__ == '__main__':
    app.run(debug=True)