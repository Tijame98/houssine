import datetime
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, SimpleRNN, LSTM

#import test

n_lags = 3

df = pd.read_csv('adapted_weather_dataset.csv')
df1 = pd.read_csv('updated_weather_dataset_with_energy.csv')

if st.checkbox('Show dataframe Head'):
    chart_data = df.head()
    x = df1.head(11)

    chart_data, x

if st.checkbox('Show dataframe Columns'):
    chart_data = df.columns
    x = df1.columns

    chart_data, x

if st.checkbox('Show dataframe info'):
    chart_data = df.describe()
    x = df1.describe()

    chart_data, x

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Wish variable would like to select?',
    ('Temperature', 'IndoorTemp', 'Radiation')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

option = st.selectbox(
    'Which number do you like best?',
     df[['Temperature', 'Humidity', 'Radiation', 'Time', 'DayOfWeek', 'Month', 'WindDirection(Degrees)']])

'You selected: ', option

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Taux de charge d'occupance", "Indoor Temp", "Outdoor Temp"))
    st.write(f"You are in {chosen} variable!")

#---------------------------------------------ANN MODEL--------------------------------------------------------

# Define relevant features
features = ['Temperature', 'Humidity', 'Radiation', 'Hour', 'DayOfWeek', 'Month']
target = 'IndoorTemp'

# Define input and output for the normal ANN
X1 = df[features]
y1 = df[target]

@st.cache_resource
def train_ann_model(X, y):
    # Normalize the input features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Define and train the normal ANN model
    ann_model = MLPRegressor(hidden_layer_sizes=(40,80,100,130), max_iter=500, random_state=42)
    ann_model.fit(X_train, y_train)
    
    # Predict and evaluate the normal ANN model
    y_pred = ann_model.predict(X_test)
    mse_ann = mean_squared_error(y_test, y_pred)
    r2_ann = r2_score(y_test, y_pred)
    
    return ann_model, X_test, y_test, y_pred, mse_ann, r2_ann

# Train the model
model, X_test, y_test, y_pred, mse, r2 = train_ann_model(X1, y1)

# Display the metrics
st.write(f"MSE of the normal ANN: {mse}")
st.write(f"R2 Score of the normal ANN: {r2}")

# Plotting actual vs predicted values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
ax.grid(True)

st.pyplot(fig)

# Plotting the difference between actual and predicted values
fig, ax = plt.subplots()
ax.plot(y_test.values, label='Actual', color='b')
ax.plot(y_pred, label='Predicted', color='r')
ax.set_xlabel('Sample index')
ax.set_ylabel('Indoor Temperature')
ax.set_title('Actual and Predicted Indoor Temperature')
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Calculating and plotting the error between actual and predicted values
error_ann = y_test.values - y_pred

fig3, ax3 = plt.subplots()
ax3.plot(error_ann, label='Error', color='g')
ax3.set_xlabel('Sample index')
ax3.set_ylabel('Error')
ax3.set_title('Error between Actual and Predicted Values')
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

#--------------------------------------------------Auto-reg ANN-------------------------------------------------

# Define input and output for the autoregressive ANN
#lagged_features = features + [f'Temperature_lag{i}' for i in range(1, n_lags + 1)] + [f'IndoorTemp_lag{i}' for i in range(1, n_lags + 1)]
lagged_features = features + [f'Temperature_lag{i}' for i in range(1, n_lags + 1)] + [f'IndoorTemp_lag{i}' for i in range(1, n_lags + 1)] 
X_ar = df[lagged_features]
y_ar = df[target]

@st.cache_resource
def train_autoreg_ann(X_ar, y_ar):
    # Normalize the input features
    scaler = MinMaxScaler()
    X_ar_scaled = scaler.fit_transform(X_ar)
    
    # Split the dataset into training and testing sets
    X_train_ar, X_test_ar, y_train_ar, y_test_ar = train_test_split(X_ar_scaled, y_ar, test_size=0.2, random_state=42)

    # Define and train the autoregressive ANN model
    ar_ann_model = MLPRegressor(hidden_layer_sizes=(40,80,100,130), max_iter=500, random_state=42)
    ar_ann_model.fit(X_train_ar, y_train_ar)
    
    # Predict and evaluate the autoregressive ANN model
    y_pred_ar = ar_ann_model.predict(X_test_ar)
    mse_ar_ann = mean_squared_error(y_test_ar, y_pred_ar)
    r2_ar_ann = r2_score(y_test_ar, y_pred_ar)

    return ar_ann_model, X_test_ar, y_test_ar, y_pred_ar, mse_ar_ann, r2_ar_ann

model_ar, X_test_ar, y_test_ar, y_pred_ar, mse_ar, r2_ar = train_autoreg_ann(X_ar, y_ar)

# Display the metrics
st.write(f"MSE of the autoreg ANN: {mse_ar}")
st.write(f"R2 Score of the autoreg ANN: {r2_ar}")

# Plotting actual vs predicted values
fig, ax = plt.subplots()
ax.scatter(y_test_ar, y_pred_ar, edgecolors=(0, 0, 0))
ax.plot([y_test_ar.min(), y_test_ar.max()], [y_test_ar.min(), y_test_ar.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
ax.grid(True)

st.pyplot(fig)

# Calculating and plotting the error between actual and predicted values
error_ar = y_test_ar.values - y_pred_ar

fig3, ax3 = plt.subplots()
ax3.plot(error_ar, label='Error', color='g')
ax3.set_xlabel('Sample index')
ax3.set_ylabel('Error')
ax3.set_title('Error between Actual and Predicted Values')
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

#----------------------------------RNN MODEL--------------------------------------------

@st.cache_resource
def rnn_model():
    scaler = MinMaxScaler()
    X_ar_scaled = scaler.fit_transform(X_ar)
    # Reshape input data to be 3D [samples, time steps, features] for RNN and LSTM
    X_ar_reshaped = X_ar_scaled.reshape((X_ar_scaled.shape[0], 1, X_ar_scaled.shape[1]))
    
    # Split the dataset into training and testing sets
    X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_ar_reshaped, y_ar, test_size=0.2, random_state=42)
    
    # Define the RNN model
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(50, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
    rnn_model.add(Dense(1))
    rnn_model.compile(optimizer='adam', loss='mse')
    
    # Train the RNN model
    rnn_model.fit(X_train_rnn, y_train_rnn, epochs=50, batch_size=32, verbose=1)
    
    # Predict and evaluate the RNN model
    y_pred_rnn = rnn_model.predict(X_test_rnn)
    mse_rnn = mean_squared_error(y_test_rnn, y_pred_rnn)
    r2_rnn = r2_score(y_test_rnn, y_pred_rnn)

    return rnn_model, X_test_rnn, y_test_rnn, y_pred_rnn, mse_rnn, r2_rnn

model_rnn, X_test_rnn, y_test_rnn, y_pred_rnn, mse_rnn, r2_rnn = rnn_model()

# Display the metrics
st.write(f"MSE of the autoreg ANN: {mse_rnn}")
st.write(f"R2 Score of the autoreg ANN: {r2_rnn}")

# Plotting actual vs predicted values
fig1, ax1 = plt.subplots()
ax1.scatter(y_test_rnn, y_pred_rnn, edgecolors=(0, 0, 0), alpha=0.7, label='Predicted vs Actual')
ax1.plot([y_test_rnn.min(), y_test_rnn.max()], [y_test_rnn.min(), y_test_rnn.max()], 'k--', lw=2, label='Ideal Fit')
ax1.set_xlabel('Actual Values')
ax1.set_ylabel('Predicted Values')
ax1.set_title('Actual vs Predicted Values (RNN)')
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# Convert y_test_rnn and y_pred_rnn to NumPy arrays and calculate the error
error = y_test_rnn.ravel() - y_pred_rnn.ravel()

fig2, ax2 = plt.subplots()
ax2.plot(error, color='red', alpha=0.6, label='Error')
ax2.axhline(y=0, color='black', linestyle='--', lw=2)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Error')
ax2.set_title('Error between Actual and Predicted Values')
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)
