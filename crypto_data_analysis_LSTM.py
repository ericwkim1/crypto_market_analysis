import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import math
import mysql.connector
from mysql.connector import errorcode
import matplotlib.dates as mdates

db_config = {
    'user': 'your_database_user',
    'password': 'your_database_password',
    'host': 'your_database_host',
    'database': 'your_database_name'
}
def fetch_historical (symbol):

    try:
        # conn = mysql.connector.connect(user='root', password='testpass', host='127.0.0.1', 
        #                         database='jobfinder')
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        #sql = "SELECT * FROM historical_klines WHERE symbol = %s"

        sql = "SELECT symbol, close, close_time FROM intraday_price WHERE TIME = (SELECT distinct TIME FROM intraday_price ORDER BY time DESC LIMIT 1) AND symbol = %s"

        cursor.execute(sql, (symbol,))
        rs = cursor.fetchall()
        df = pd.DataFrame(rs)
        df.columns = [column[0] for column in cursor.description]
        df['Date'] = pd.to_datetime(df['close_time'])
        df = df.set_index('Date')

        return df

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
          print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
          print("Database does not exist")
        else:
          print(err)

    return False

# Make sure Tensorflow is using GPU
#tf.debugging.set_log_device_placement(True)

# Load the data into a pandas DataFrame
coin_id = 'bitcoin'
df = fetch_historical(coin_id)

# Preprocess the data
df['close'] = np.log(df['close'])
dataset = df[['close']].values

# Split the data into training and testing sets
training_data_len = math.ceil(len(dataset) * 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]

# Reshape the data into a 3D array to feed into the LSTM model
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 60
X_train, y_train = create_dataset(train_data, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Make predictions using the last 60 days of the training set
inputs = np.array(scaled_data[training_data_len - look_back:])
inputs = np.reshape(inputs, (1, inputs.shape[0], 1))
predictions = []
future_days = 100
for i in range(future_days):
    prediction = model.predict(inputs)
    predictions.append(prediction[0][0])
    inputs = np.append(inputs[0][1:], prediction[0][0])
    inputs = np.reshape(inputs, (1, inputs.shape[0], 1))

# Inverse transform the predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
predictions = np.exp(predictions)

# Calculate the RMSE

y_test = dataset[training_data_len:training_data_len+100, :]
predictions_rmse = predictions[:y_test.shape[0]]
rmse = np.sqrt(np.mean((predictions_rmse - y_test)**2))
print(predictions)
print("RMSE:", rmse)



# Plot the historical data
plt.plot(df.index[:training_data_len], np.exp(df['close'][:training_data_len]), label="Training Data")
plt.plot(df.index[training_data_len:], np.exp(df['close'][training_data_len:]), label="Historical Data")

# Plot the predictions
prediction_index = [df.index[-1] + pd.Timedelta(days=i+1) for i in range(100)]
plt.plot(prediction_index, predictions, label="Predictions")

# Set the x-axis to display as dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))

plt.xlabel("Days")
plt.ylabel("Close Price")
plt.title(str(coin_id) + " - LSTM Model Predictions, RMSE: " + str(float(rmse)))
plt.legend()
plt.xticks(rotation=45)
plt.show()