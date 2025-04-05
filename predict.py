import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle

# Sports and Weights
sports = ['Football', 'Archery', 'Swimming', 'Running', 'Basketball']
genders = ['Male', 'Female', 'Other']
sport_weights = {
    'Football': {'strength': 0.3, 'stamina': 0.4, 'endurance': 0.2, 'eyesight': 0.1},
    'Archery': {'strength': 0.1, 'stamina': 0.2, 'endurance': 0.2, 'eyesight': 0.5},
    'Swimming': {'strength': 0.25, 'stamina': 0.35, 'endurance': 0.3, 'eyesight': 0.1},
    'Running': {'strength': 0.2, 'stamina': 0.4, 'endurance': 0.3, 'eyesight': 0.1},
    'Basketball': {'strength': 0.3, 'stamina': 0.3, 'endurance': 0.2, 'eyesight': 0.2}
}

# Generate Data
def generate_custom_data(n=1000):
    data = []
    for _ in range(n):
        sport = np.random.choice(sports)
        gender = np.random.choice(genders)
        strength = np.random.uniform(1, 10)
        stamina = np.random.uniform(1, 10)
        endurance = np.random.uniform(1, 10)
        eyesight = np.random.uniform(5, 10) if sport == 'Archery' else np.random.uniform(1, 10)

        weights = sport_weights[sport]
        fitness_score = (
            weights['strength'] * strength +
            weights['stamina'] * stamina +
            weights['endurance'] * endurance +
            weights['eyesight'] * eyesight
        ) * 10
        fitness_score = min(fitness_score, 100)
        risk_factor = max(0, 100 - fitness_score + np.random.normal(0, 5))

        data.append([strength, stamina, endurance, eyesight, gender, sport, fitness_score, risk_factor])
    return pd.DataFrame(data, columns=['strength', 'stamina', 'endurance', 'eyesight', 'gender', 'sport', 'fitness_score', 'risk_factor'])

# Load data
df = generate_custom_data()

# Encode gender and sport
gender_encoder = LabelEncoder()
sport_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])
df['sport'] = sport_encoder.fit_transform(df['sport'])

# Features & Labels
X = df[['strength', 'stamina', 'endurance', 'eyesight', 'gender', 'sport']]
y = df[['fitness_score', 'risk_factor']]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y.values, test_size=0.2, random_state=42)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(1, X.shape[1])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=30, validation_split=0.1, verbose=1)

# ✅ Save model
model.save("lstm_model.keras")

# ✅ Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Save encoders
with open("encoder.pkl", "wb") as f:
    pickle.dump({"gender": gender_encoder, "sport": sport_encoder}, f)

print("✅ LSTM model, scaler, and encoders saved!")
