# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 08:46:23 2019

@author: Anuj Pareek
"""
#importing libraries
from flask import Flask,render_template,request
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import re
import nltk
import random
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

"""
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
"""
buffer={}
dataset=pd.read_csv('input_data.csv')

dataset=dataset.iloc[:,[1,2,3,4]]
data={}

# load the model from disk
cv = pickle.load(open('cv_model.sav', 'rb'))
classifier = pickle.load(open('classifier_model.sav', 'rb'))

# load json and create model
from keras.models import model_from_json

json_file = open('image_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
image_model = model_from_json(loaded_model_json)

# load weights into new model
image_model.load_weights("image_model.h5")
#print("Loaded model from disk")

#make the predict function
image_model._make_predict_function()

app=Flask(__name__)

result={}


def analyseText(text):
    phrase = re.sub('[^a-zA-Z]', ' ', text)
    phrase = phrase.lower()
    phrase = phrase.split()
    
    phrase = [word for word in phrase 
          if not word 
          in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    phrase = [ps.stem(word) for word in phrase]
    #phrase = [lem.lemmatize(word) for word in phrase if not word in set(stopwords.words('english'))]
    phrase = ' '.join(phrase)
    corpus=[]
    corpus.append(phrase)
    inputPred = cv.transform(corpus).toarray()
    
    n=classifier.predict(inputPred)
    #print("Analyzed:::::::-----  ",n)
    if(n):
        return ("SAFE")
    else:
        return ("UNSAFE")

    


@app.route('/')
def home(): 

    return render_template('home.html')

@app.route('/text')
def textpage():
    x = 0
    global buffer
    for i in random.sample(range(0,len(dataset.text)),10) :
        data[x]={}
        data[x]['text']=dataset['text'][i]
        data[x]['Username']=dataset['Username'][i]
        data[x]['Timestamp']=dataset['Timestamp'][i]
        data[x]['location']=dataset['location'][i]
        x = x+1
    buffer=data
    return render_template("text.html",result=data)
@app.route('/text/analyze',methods=['POST'])
def showAnalysis():
    x = 0
    for i in random.sample(range(0,len(dataset.text)),5) :
        data[x]={}
        data[x]['text']=dataset['text'][i]
        data[x]['Username']=dataset['Username'][i]
        data[x]['Timestamp']=dataset['Timestamp'][i]
        data[x]['location']=dataset['location'][i]
        x = x+1
    
    text = (request.form['textbox'])
    # '''
    # FOR TESTING THE RECEIVED DATA, WE STORED IT IN A FILE'''
    # file = open('new.txt', mode='wt')
    # file.write(str(text))
    # file.close()
    phrase = re.sub('[^a-zA-Z]', ' ', text)
    phrase = phrase.lower()
    phrase = phrase.split()
    
    phrase = [word for word in phrase 
          if not word 
          in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    phrase = [ps.stem(word) for word in phrase]
    #phrase = [lem.lemmatize(word) for word in phrase if not word in set(stopwords.words('english'))]
    phrase = ' '.join(phrase)
    corpus=[]
    corpus.append(phrase)
    inputPred = cv.transform(corpus).toarray()
    
    n=classifier.predict(inputPred)
    #print("Analyzed:::::::-----  ",n)
    if(n):
        return render_template('text.html',display_text="""SAFE""",result=data)
    else:
        return render_template('text.html',display_text="""UNSAFE""",result=data)
@app.route('/text/analyzeTable')
def analyzeTable():
    for i in range(0,10):
        buffer[i]['Status']=analyseText(buffer[i]['text'])
        
    return render_template('tableOutput.html',result=buffer)
        
        

@app.route('/images',methods=['POST','GET'])
def showImageAnalyzer():
    return render_template('images.html')
    
    
@app.route('/images/analyzeImage',methods=['POST'])
def showImageAnalysis():
        try:
            if (request.method=='POST'):
                f=request.files['file']
                if((f)==None):
                    return render_template('images.html',display_text="""SELECT AN IMAGE""")    
                f=request.files['file']
                #f.save('C:\\Users\\Windows10\\Desktop\\FYP\\Running Environment 2 (with twitter text data)\\',f.filename)
                f.save(f.filename)
                #Catching the data received
                '''
                data=(request.form)
                #Slicing out only the required portion of the data (BASE64 string)
                #converting the clipped string into byte format as it is the supported format by the method decodebytes()
                data=base64.decodebytes(bytes((data['value'][22:]),'utf-8'))
                '''
                
                image = Image.open(str(f.filename))
                #converting the image to greyscale
                #image=image.convert('L')
                #resizing with smoothing (ANTIALIAS)
                '''left = 500
                top = 500
                right =500
                bottom = 500
                image = image.crop((left, top, right, bottom))'''
                image=image.resize((64,64),Image.ANTIALIAS)
                image.save(f.filename)
                #converting the image to array
                image = np.asarray(image)
                #dividing each pixel intensity by 255 to apply MINMAX scaling
                image=image.astype('float32')/255
                #converting the image shape to that of the training data as it is what the model accepts
                image=image.reshape(1,64,64,3)
                #storing the index of the output array which has the greatest probabilistic value
                number=np.argmax(image_model.predict(image))
                os.remove(f.filename)
        except:
            os.remove(f.filename)
            return render_template('images.html',display_text="""UNABLE TO PREDICT""")

        #returning predicted number as a response
        if((type(number)!=None)):
            if(number==0):
                return render_template('images.html',display_text="""UNSAFE""")
            else:
                return render_template('images.html',display_text="""SAFE""")
            
        else:
            return render_template('images.html',display_text="UNABLE TO PREDICT")
            
if __name__=='__main__':
    app.run(debug=True)
