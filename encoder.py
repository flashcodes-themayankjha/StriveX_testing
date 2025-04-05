import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save encoders (if you used them for gender/sport)
with open("encoder.pkl", "wb") as f:
    pickle.dump({"gender": gender_encoder, "sport": sport_encoder}, f)
