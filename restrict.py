import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Sample dataset (add more rows for better model)
data = pd.DataFrame([
    {"Strength": 80, "Stamina": 70, "Endurance": 75, "Eye_Sight": 1.0, "Gender": "male", "Sport": "football", "Fitness_Score": 85, "Risk_Factor": 20},
    {"Strength": 60, "Stamina": 50, "Endurance": 55, "Eye_Sight": 0.9, "Gender": "female", "Sport": "tennis", "Fitness_Score": 70, "Risk_Factor": 35},
    # Add more realistic rows here...
])

X = data.drop(["Fitness_Score", "Risk_Factor"], axis=1)
y_fitness = data["Fitness_Score"]
y_risk = data["Risk_Factor"]

categorical_features = ['Gender', 'Sport']
numerical_features = ['Strength', 'Stamina', 'Endurance', 'Eye_Sight']

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

# Fitness model pipeline
fitness_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# Risk model pipeline
risk_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

fitness_pipeline.fit(X, y_fitness)
risk_pipeline.fit(X, y_risk)

# Save models using joblib
joblib.dump(fitness_pipeline, "fitness_model.joblib")
joblib.dump(risk_pipeline, "risk_model.joblib")

print("✅ Models saved successfully with compatible version.")
