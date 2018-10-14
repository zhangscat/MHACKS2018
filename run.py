# IMPORT LIBRARIES 
# ------------------------------------------------------------
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import data

# SEND TEXT ------------------------------------------------------------------

# Your Account Sid and Auth Token from twilio.com/console
account_sid = 'AC9a2390a48b67fd3b749b74bd5b86ed28'
auth_token = 'c9196a9d80b28804445ca70276d507c4'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="Hello, how may I help?",
                     from_='+16507536403',
                     to='+16508673228'
                 )

print(message.sid)

# RECEIVE TEXT SYMPTOMS ---------------------------------------------------------

app = Flask(__name__)

@app.route("/sms", methods=['GET', 'POST'])
def incoming_sms():
    """Send a dynamic reply to an incoming text message"""
    # Get the message the user sent our Twilio number
    body = request.values.get('Body', None)

    # CALL PREDICTION FUNCTION FROM DATA ***************************************
    print(body)
    sendText = data.predictDisease(body)

    # Start our TwiML response
    resp = MessagingResponse()

    # SET RESPONSE MESSAGE BASED ON PREDICTION FUNCTION ************************
    print(sendText)
    resp.message(sendText)

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)