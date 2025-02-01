import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

import numpy as np
import pandas as pd
from keras.models import load_model

# Load the model
model = load_model("model.keras")

# Generate random data as a DataFrame
sequence_length = 50
num_features = 24
random_df = pd.DataFrame(np.random.rand(sequence_length, num_features), columns=[f"feature_{i}" for i in range(num_features)])

# Convert to NumPy array and reshape
random_data = random_df.to_numpy().reshape(1, sequence_length, num_features)

# Make a prediction
prediction = model.predict(random_data)

# Print DataFrame and Prediction
print("Input DataFrame:\n", random_df.head())  # Show first few rows
print("\nModel Prediction:", prediction)

random_df.to_csv('random.csv', index=False)