# MHACKS2018
disease detective program

Our official MHACKS 2018 project. Our program is a text chatbot that can receive a text, scan it for possible symptoms, and
input those symptoms into a tensorflow machine learning program. The basic classifier model will have already trained a
classification model based on the disease and symptoms csv files. Based off the given text, our model will evaluate the 
probability of each possible disease and output a disease with the highest probability. It will then send a text--via Twilio
API--that tells the user the most probable disease and how to treat such disease. Our model went up to around 50% accuracy and
took in data as a numpy array of 1's and 0's for containing symptom or symptom free. The run.py file is to run the Twilio 
chatbot API which then calls on the machine learning file to classify the disease. Our project was focused around food borne
illnesses.
