#!/usr/bin/env python
# coding: utf-8

# In[129]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

from tkinter import *
import tkinter as tk
import io
from PIL import ImageGrab, Image,ImageDraw


# In[130]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
len(X_train)


# In[131]:


len(X_test)


# In[132]:


X_train[0].shape


# In[133]:


X_train[0]


# In[134]:


plt.matshow(X_train[0])


# In[135]:


y_train[0]


# In[136]:


X_train = X_train / 255
X_test = X_test / 255


# In[137]:


X_train[0]


# In[138]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[139]:


X_train_flattened.shape


# In[140]:


X_train_flattened[0]


# In[141]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[142]:


model.evaluate(X_test_flattened, y_test)


# In[143]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[144]:


plt.matshow(X_test[0])


# In[145]:


np.argmax(y_predicted[0])


# In[146]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[147]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[148]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[149]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[150]:


model.evaluate(X_test_flattened,y_test)


# In[151]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[163]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1)


# In[164]:


model.evaluate(X_test,y_test)


# In[165]:


drawing = False
last_x = 0
last_y = 0

def start_drawing(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y

def draw(event):
    global drawing, last_x, last_y, canvas
    if drawing:
        x, y = event.x, event.y
        canvas.create_line((last_x, last_y, x, y), width=10, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
        last_x, last_y = x, y

def stop_drawing(event):
    global drawing
    drawing = False

def predict_digit():
    # Get the drawn image from the canvas
    img = canvas.postscript(colormode='gray')
    img = Image.open(io.BytesIO(img.encode('utf-8')))
    img = img.resize((28, 28))
    img_data = np.array(img)[:, :, 0] / 255.0
    img_data = np.reshape(img_data, (1, 28, 28))
    
    # Predict the digit using the model
    prediction = model.predict(img_data)
    predicted_digit = np.argmax(prediction)
    
    result_label.config(text=f"Predicted Digit: {predicted_digit}")




# In[166]:


# Create the main application window
app = tk.Tk()
app.title("Digit Recognition")

# Create a canvas for drawing
canvas = tk.Canvas(app, width=280, height=280, bg="white")
canvas.pack()

# Bind canvas events
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

# Create a button to predict the digit
predict_button = tk.Button(app, text="Predict Digit", command=predict_digit)
predict_button.pack()

# Create a label to display the predicted digit
result_label = tk.Label(app, text="")
result_label.pack()

app.mainloop()


# In[ ]:




