# Jerry Liu MHACKS 2018
# IMPORT LIBRARIES ---------------------------------------------------------------
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# IMPORT DATA ---------------------------------------------------------------------
TRAIN_URL = "disease_train.csv"
TEST_URL = "disease_test.csv"

CSV_COLUMN_NAMES = ['cramps', 'diarrhea', 'nausea', 'fever', 'vomitting', 'vision', 'urine', 'illness']
SPECIES = ['b. cereus food poisoning', 'campylobacteriosis', 'botulism', 'e. coli', 'hepatitis']
SYMPTOMS = ['cramps', 'diarrhea', 'nausea', 'fever', 'vomitting', 'vision', 'urine']

def load_data(y_name='illness'):
   """Returns the disease dataset as (train_x, train_y), (test_x, test_y)."""

   train = pd.read_csv(TRAIN_URL, names=CSV_COLUMN_NAMES, header=0)
   train_x, train_y = train, train.pop(y_name)

   test = pd.read_csv(TEST_URL, names=CSV_COLUMN_NAMES, header=0)
   test_x, test_y = test, test.pop(y_name)

   return (train_x, train_y), (test_x, test_y)


(train_symptoms, train_illness), (test_symptoms, test_illness) = load_data()

# prep data
train_symptoms_np = np.array(train_symptoms)
train_illness_np = np.array(train_illness)
test_symptoms_np = np.array(test_symptoms)
test_illness_np = np.array(test_illness)


# explore data
# print(train_symptoms.shape)
# print(train_illness.shape)
# print(test_symptoms.shape)
# print(test_illness.shape)

# TRAIN MODEL --------------------------------------------------------------
# build model
# attempt 1: keras
model = keras.Sequential()
model.add(keras.layers.Dense(5, activation=tf.nn.relu))
model.add(keras.layers.Dense(4, activation=tf.nn.softmax))

# compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# train model
model.fit(train_symptoms_np, train_illness_np, epochs=5)


# TEST MODEL --------------------------------------------------------------

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_symptoms, test_illness)
print('Test accuracy:', test_acc)

# CONVERT TEXT -----------------------------------------------------------------

def transferData(textmsg):
    words = textmsg.split()
    uniqueWords = []
    for word in words:
        uniqueWords.append(word.strip('.,'))

    print(uniqueWords)
    print(SYMPTOMS)
    inputData = [0, 0, 0, 0, 0, 0, 0]
    index = 0
    for symptom in SYMPTOMS:
        for word in uniqueWords:
            if symptom == word:
                print(symptom)
                inputData[index] = 1
        index += 1
    print(inputData)
    return inputData

# PREDICT ------------------------------------------------------------------

# define a function to predict a disease based on symptoms

def predictDisease(inputText):
    inData = transferData(inputText)
    print(inData)
    inData = (np.expand_dims(inData,0))
    prediction = model.predict(inData)
    print(prediction)
    sendText = ""

    if np.argmax(prediction) == 0:
        # print("You have cancer0!")
        sendText = "You have salmonella!"
    elif np.argmax(prediction) == 1:
        # print("You have cancer1!")
        sendText = "You have hepatitis!"
    elif np.argmax(prediction) == 2:
        # print("You have cancer2!")
        sendText = "You have e. coli!"
    elif np.argmax(prediction) == 3:
        # print("You have cancer3!")
        sendText = "You have bacillus cereus!"
    return sendText

# TEST PREDICT ---------------------------------------------------------------

# print(predictDisease("I, Catherine B. Zhang, have diarrhea."))

# print("diarrhea" == "diarrhea")
# print(transferData("I, Catherine B. Zhang, have diarrhea."))
# var = transferData("I, Catherine B. Zhang, have diarrhea.")
print(predictDisease("I, Catherine B. Zhang, have diarrhea."))




