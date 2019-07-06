import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pickle
import random
import json
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse, Message
from twilio.rest import Client
import urllib



def clean_up_sentence(sentence):
	# tokenize the pattern - split words into array
	sentence_words = nltk.word_tokenize(sentence)
	# stem each word - create short form for word
	sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
	return sentence_words

def bow(sentence, words, show_details=True):
	# tokenize the pattern
	sentence_words = clean_up_sentence(sentence)
	# bag of words - matrix of N words, vocabulary matrix
	bag = [0]*len(words)  
	for s in sentence_words:
		for i,w in enumerate(words):
			if w == s: 
				bag[i] = 1
	return(np.array(bag))

client = Client('ACd6999e72ac712cfec4833726199e1129', '75741f5eb5f52ec79df75c4832d99fa4')
app = Flask(__name__)
def init():
	global model,graph
	# load the pre-trained Keras model
	model = load_model('models/RBTNewPrediction.h5')
	graph = tf.get_default_graph()

def getSentence():
	parameters = []
	parameters.append(flask.request.args.get('sentence'))
	return parameters

def sendResponse(responseObj):
	response = flask.jsonify(responseObj)
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Access-Control-Allow-Methods', 'GET')
	response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
	response.headers.add('Access-Control-Allow-Credentials', True)
	return response

@app.route("/prediction", methods=["GET"])
def prediction():
	sentence = getSentence()
	s = sentence[0]
	p = bow(s, words)
	with graph.as_default():
		results = model.predict( np.array( [p,] )  )
		arr = np.where(results == np.amax(results))
		guess = classes[arr[1][0]]
		return sendResponse({"prediction": guess})

@app.route('/sms', methods=['POST'])
def inbound_sms():
	resp = MessagingResponse()
	sentence = request.values.get('Body', None)
	p = bow(sentence, words)
	with graph.as_default():
		results = model.predict( np.array( [p,] )  )
		arr = np.where(results == np.amax(results))
		guess = classes[arr[1][0]]
		resp.message("I'm gonna guess that was a: " + guess)

	return str(resp)

if __name__ == "__main__":	
	with open('intents.json') as data:
		intents = json.load(data)
	
	words = []
	classes = []
	documents = []
	ignore_words = ['?']

	for intent in intents['intents']:
		for pattern in intent['patterns']:
			w = nltk.word_tokenize(pattern)
			words.extend(w)
			documents.append((w, intent['tag']))
			if intent['tag'] not in classes:
				classes.append(intent['tag'])

	words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
	words = sorted(list(set(words)))
	classes = sorted(list(set(classes)))

	init()
	app.run(threaded=True)






