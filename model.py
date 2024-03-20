import http.client
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

# Fetch data from Alpha Vantage API
conn = http.client.HTTPSConnection("alpha-vantage.p.rapidapi.com")
headers = {
    'X-RapidAPI-Key': "ca0857ab6amsh5a61f9cefe65300p1f7020jsn6fe51adb2fbb",
    'X-RapidAPI-Host': "alpha-vantage.p.rapidapi.com"
}
conn.request("GET", "/query?interval=15min&function=TIME_SERIES_INTRADAY&symbol=MSFT&datatype=json&output_size=compact", headers=headers)
res = conn.getresponse()
data = res.read().decode("utf-8")

# Convert data to DataFrame
data_json = json.loads(data)
df = pd.DataFrame(data_json['Time Series (15min)']).T
df.columns = [col.split()[-1] for col in df.columns]  # Clean column names

# Scale the data
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)

# Define the number of past time steps you want to use for each sequence
sequence_length = 15

# Initialize lists to store sequences and corresponding target values
sequences = []
targets = []

# Create sequences and targets
for i in range(len(scaled_df) - sequence_length):
    sequences.append(scaled_df[i:i + sequence_length])
    targets.append(scaled_df[i + sequence_length])

# Convert sequences and targets to numpy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, shuffle=False)

# Define the features you want to include
selected_features = ['open', 'high', 'low', 'close', 'volume']

# Select the relevant columns from your data for training and testing sets
X_train_selected = X_train[:, :, [df.columns.get_loc(col) for col in selected_features]]
X_test_selected = X_test[:, :, [df.columns.get_loc(col) for col in selected_features]]

# Define the shape of input sequences
input_shape = X_train_selected.shape[1:]

# Create a sequential model
model = Sequential()

# Add Bi-Directional LSTM layers
model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=input_shape))
model.add(Dropout(0.2))  

model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(Dropout(0.2))

# Add a Dense layer for output
model.add(Dense(units=len(selected_features)))  

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data into training and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_selected, y_train, test_size=0.2, random_state=42)

# Define a loss function suitable for your trading task
loss_function = 'mean_squared_error'

# Choose an optimizer
optimizer = tf.keras.optimizers.Adam()

# Compile the model with the chosen loss function and optimizer
model.compile(optimizer=optimizer, loss=loss_function)

# Define the number of epochs for training
num_epochs = 5

# Train the model using the training data
history = model.fit(X_train_final, y_train_final, epochs=num_epochs, batch_size=1, validation_data=(X_val, y_val))

# Plot the training and validation loss over epochs
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

