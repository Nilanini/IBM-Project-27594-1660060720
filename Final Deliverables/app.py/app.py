import os
import numpy as np #used for numerical analysis
from flask import Flask,request,render_template
# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
#render_template- used for rendering the html pages
import  tensorflow as tf
from tensorflow.keras.models import load_model #to load our trained model
from tensorflow.keras.preprocessing import image
from PIL import Image

app=Flask(__name__)#our flask app
model=load_model('mnistCNN.h5')#loading the model
@app.route("/") #default route
def about():
    return render_template("main.html")#rendering html page

@app.route("/home") #default route
def home():
    return render_template("main.html")#rendering html page
@app.route("/upload") #default route
def test():
    return render_template("index6.html")#rendering html page


@app.route("/predict",methods=["GET","POST"]) #route for our prediction
def upload_image_file():
   if request.method == 'POST':
      img = Image.open(request.files['file'].stream).convert("L")
      img = img.resize((28,28))
      im2arr = np.array(img)
      im2arr = im2arr.reshape(1,28,28,1)
      y_pred = model.predict(im2arr)
      predict = np.argmax(y_pred)
      print(predict)

      #return "Predicted Number: + str(pred) returning our output


      if(predict == 0):
          return render_template("0.html")
      elif(predict == 1):
          return render_template("1.html")
      elif(predict == 2):
          return render_template("2.html")
      elif(predict == 3):
          return render_template("3.html")
      elif(predict == 4):
          return render_template("4.html")
      elif(predict == 5):
          return render_template("5.html")
      elif(predict == 6):
          return render_template("6.html")
      elif(predict == 7):
          return render_template("7.html")
      elif(predict == 8):
          return render_template("8.html")
      elif(predict == 9):
          return render_template("9.html")  
      else:
          return None     
if __name__=="__main__":
    #app.run(debug=False)#running our app
    app.run(host='0.0.0.0', port=8000)
            
            
