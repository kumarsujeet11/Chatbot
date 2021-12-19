import json
import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lematizer=WordNetLemmatizer()

intents=json.loads(open('intents.json').read())

#Data Preprocessing

all_words=[]
tags=[]
documents=[]

ignore_words=["?",".",",","!"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        all_words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

all_words=[lematizer.lemmatize(word.lower()) for word in all_words if word not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))

pickle.dump(all_words,open('words.pkl','wb'))
pickle.dump(tags,open('classes.pkl','wb'))

training=[]
output_empty=np.zeros(len(tags))

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lematizer.lemmatize(word.lower()) for word in word_patterns]
    for word in all_words:
        bag.append(1) if word in word_patterns else bag.append(0)
    #or can be written (for idx,w in enumerate(all_words_new):
            #if w in word_patterns_new:
                #bag_new[idx]=1.0)
    output_row=list(output_empty)
    output_row[tags.index(document[1])]=1
    training.append([bag,output_row])

random.shuffle(training)
training=np.array(training)

train_X=list(training[:,0])
train_y=list(training[:,1])

#Training the model

model=Sequential()
model.add(Dense(128,input_shape=(len(train_X[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

#Optimizer

sgd=SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(np.array(train_X),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist)
print("Done")
