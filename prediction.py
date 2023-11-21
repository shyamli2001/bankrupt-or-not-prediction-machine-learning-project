# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from pickle import dump, load

# Function to train and save selected models
def train_selected_models(X_train, y_train, selected_models):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0),
        'Decision Tree': DecisionTreeClassifier()
    }

    selected_models = [model.strip() for model in selected_models]

    for model_name, model in models.items():
        if model_name in selected_models:
            model.fit(X_train, y_train)
            # Save the trained model
            with open(f'{model_name.lower().replace(" ", "_")}_allmodels.save', 'wb') as file:
                dump(model, file)


# For demonstration purposes
data = {
    'Industrial Risk': [0.2, 0.5, 0.8],
    'Management Risk': [0.6, 0.3, 0.7],
    'Financial Flexibility': [0.1, 0.9, 0.4],
    'Credibility': [0.8, 0.2, 0.5],
    'Competitiveness': [0.3, 0.7, 0.1],
    'Operating Risk': [0.9, 0.4, 0.6],
    'Target': [0, 1, 0]
}

X, y = pd.DataFrame(data), data['Target']
X_train, _, y_train, _ = train_test_split(X.drop('Target', axis=1), y, test_size=0.2, random_state=42)

# Train selected models
selected_models = st.multiselect('Select Models to Train', ['Naive Bayes', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree'])
train_selected_models(X_train, y_train, selected_models)

# Function to get user input features
def user_input_features():
    user_inputs = {}
    for feature in X.columns[:-1]:  # Exclude the target column
        user_input = st.slider(f'{feature}', 0.0, 1.0, step=0.1)
        user_inputs[feature] = user_input
    return pd.DataFrame(user_inputs, index=[0])

# Streamlit app
st.title('Bankruptcy Predictions')

# Load selected models
loaded_models = {}
for model_name in selected_models:
    try:
        loaded_models[model_name] = load(open(f'{model_name.lower().replace(" ", "_")}_allmodels.save', 'rb'))
    except FileNotFoundError:
        st.error(f'Model for {model_name} not found. Please run the training script.')

# Getting data from the user
df_user_input = user_input_features()

# Creating a button for prediction
if st.button('Predict'):
    st.subheader('Predicted Result')
    for model_name, model in loaded_models.items():
        prediction_proba = model.predict_proba(df_user_input)
        result_message = f'{model_name}: Success (Non-Bankruptcy)' if prediction_proba[0][1] > prediction_proba[0][0] else f'{model_name}: Failure (Bankruptcy)'
        st.success(result_message)

