import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
import keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score

DATA_PATH = 'datasets/'
df_train = pd.read_csv(DATA_PATH + 'PM_train.csv')
df_test = pd.read_csv(DATA_PATH + 'PM_test.csv')
df_truth = pd.read_csv(DATA_PATH + 'PM_truth.csv')

df_truth.head()

for i in range(1, 101):
    max_rul = df_train[df_train['id'] == i]['cycle'].max()
    df_train.loc[df_train['id'] == i, 'RUL'] = df_train[df_train['id'] == i]['cycle'].apply(lambda x: max_rul - x)


# Define window values
w0, w1 = 15, 30

# Create label1 for training data
df_train['label1'] = np.where(df_train['RUL'] <= w1, 1, 0)

# Create label2 for training data
df_train['label2'] = np.where(df_train['RUL'] > w1,
                              0,
                              np.where((df_train['RUL'] <= w1) & (df_train['RUL'] > w0),
                                      1, 2))


def normalize_data(df, col_not_to_norm):
    columns_to_normalize = df.columns.difference(col_not_to_norm) 
    
    # Separate the columns
    df_to_normalize = df[columns_to_normalize]
    df_not_to_normalize = df[col_not_to_norm]
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    normalized_df = pd.DataFrame(scaler.fit_transform(df_to_normalize), columns=columns_to_normalize)

    return pd.concat([df_not_to_normalize, normalized_df], axis=1)

columns_not_to_normalize = ['id', 'cycle', 'RUL', 'label1', 'label2']
df_train = normalize_data(df_train, columns_not_to_normalize)
df_train.head()

# Combine test and truth data
df_test = pd.merge(df_test, df_truth.rename(columns={'cycle': 'RUL'}), on=['id'], how='left')

# create RUL column
df_test['RUL'] = df_test['RUL'] - df_test['cycle']

# Create label1 for training data
df_test['label1'] = np.where(df_test['RUL'] <= w1, 1, 0)

# Create label2 for training data
df_test['label2'] = np.where(df_test['RUL'] > w1,
                              0,
                              np.where((df_test['RUL'] <= w1) & (df_test['RUL'] > w0),
                                      1, 2))

# Normalize data
df_test = normalize_data(df_test, columns_not_to_normalize)
df_test.head()

# Select the engine ID
engine_id = 1

# Filter the dataframe for the selected engine
engine_data = df_train[df_train['id'] == engine_id]

# List of sensors to plot
sensors = [f's{i}' for i in range(1, 22)]

# Plot each sensor reading over cycles in separate graphs
for sensor in sensors:
    plt.figure(figsize=(10, 2))
    plt.plot(engine_data['cycle'], engine_data[sensor], label=sensor)
    plt.xlabel('Cycle')
    plt.ylabel('Sensor Reading')
    plt.title(f'Sensor {sensor} Readings Over Cycles for Engine 1')
    plt.legend()
    plt.show()

# Calculate the correlation matrix
correlation_matrix = df_train.corr()

# Set up the matplotlib figure
plt.figure(figsize=(20, 14))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)

# Add titles and labels
plt.title('Correlation Heatmap of Sensor Readings and Settings')
plt.xlabel('Sensors and Settings')
plt.ylabel('Sensors and Settings')

# Show the heatmap
plt.show()

# pick a large window size of 50 cycles
sequence_length = 50

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


# pick the feature columns 
sequence_cols = ['setting1', 'setting2', 'setting3']
key_cols = ['id', 'cycle']
label_cols = ['label1', 'label2', 'RUL']

input_features = df_test.columns.values.tolist()
sensor_cols = [x for x in input_features if x not in set(key_cols)]
sensor_cols = [x for x in sensor_cols if x not in set(label_cols)]
sensor_cols = [x for x in sensor_cols if x not in set(sequence_cols)]

# The time is sequenced along
# This may be a silly way to get these column names, but it's relatively clear
sequence_cols.extend(sensor_cols)

print(sequence_cols)

# generator for the sequences
seq_gen = (list(gen_sequence(df_train[df_train['id']==id], sequence_length, sequence_cols)) 
           for id in df_train['id'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape)

# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

# generate labels
label_gen = [gen_labels(df_train[df_train['id']==id], sequence_length, ['label1']) 
             for id in df_train['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)

# build the network
# Feature weights
from keras.layers import LSTM

nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

# GRU model
model = Sequential()

# The first layer
model.add(GRU(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))

# Plus a 20% dropout rate
model.add(Dropout(0.2))

# The second layer
model.add(GRU(
          units=50,
          return_sequences=False))

# Plus a 20% dropout rate
model.add(Dropout(0.2))

# Dense sigmoid layer
model.add(Dense(units=nb_out, activation='sigmoid'))

# With adam optimizer and a binary crossentropy loss. We will optimize for model accuracy.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Verify the architecture 
print(model.summary())

# Fit the network
model.fit(seq_array, # Training features
          label_array, # Training labels
          epochs=10,   # We'll stop after 10 epochs
          batch_size=200, # 
          validation_split=0.10, # Use 10% of data to evaluate the loss. (val_loss)
          verbose=1, #
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                   min_delta=0,    # until it doesn't change (or gets worse)
                                                   patience=5,  # patience > 1 so it continues if it is not consistently improving
                                                   verbose=0, 
                                                   mode='auto')]) 

# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Training Accurracy: {}'.format(scores[1]))


# Make predictions
y_pred = (model.predict(seq_array) > 0.5).astype("int32")
y_true = label_array

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Confusion Matrix')
plt.show()

# Compute precision, recall, and F1 score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)

print('Training Precision: ', precision)
print('Training Recall: ', recall)
print('Training F1 Score:', f1)

# Select the last sequence for each engine ID in the test data
seq_array_test_last = [
    df_test[df_test['id'] == id][sequence_cols].values[-sequence_length:]
    for id in df_test['id'].unique() if len(df_test[df_test['id'] == id]) >= sequence_length
]

# Convert the list to a numpy array and ensure the data type is float32
seq_array_test_last = np.array(seq_array_test_last, dtype=np.float32)
seq_array_test_last.shape


# Create a mask to filter engine IDs with enough data points
y_mask = [len(df_test[df_test['id'] == id]) >= sequence_length for id in df_test['id'].unique()]

# Extract the last label for each engine ID that meets the sequence length requirement
label_array_test_last = df_test.groupby('id')['label1'].nth(-1)[y_mask].values

# Reshape and convert to float32
label_array_test_last = label_array_test_last.reshape(-1, 1).astype(np.float32)

# Display shapes of the test sequences and labels
print(seq_array_test_last.shape)
print(label_array_test_last.shape)


# Evaluate the model on the test data
test_scores = model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
test_accuracy = test_scores[1]

# Print and log the test accuracy
print(f'Test Accuracy: {test_accuracy}')

# Make predictions
y_test_pred = (model.predict(seq_array_test_last) > 0.5).astype("int32")
y_test_true = label_array_test_last

# Compute confusion matrix
cm = confusion_matrix(y_test_true, y_test_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Confusion Matrix')
plt.show()

# Compute precision, recall, and F1 score
precision = precision_score(y_test_true, y_test_pred)
recall = recall_score(y_test_true, y_test_pred)
f1 = 2 * (precision * recall) / (precision + recall)

print('Training Precision: ', precision)
print('Training Recall: ', recall)
print('Training F1 Score:', f1)