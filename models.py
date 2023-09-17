import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

data = pd.read_csv("flight_data.csv")
data = data.iloc[:, 1:]

st.title("Airline Flight Data EDA and Price Prediction")

st.write(
    "Welcome to the Airline Flight Data EDA and Price Prediction App!\n\n"
    "This app allows you to explore and analyze airline flight data and make predictions on flight prices.\n\n"
    "You can filter the data by source city, destination city, and airline using the sidebar on the left."
)

st.sidebar.header("Filter Data")
source_city = st.sidebar.selectbox("Source City", data["source_city"].unique())
destination_city = st.sidebar.selectbox("Destination City", data["destination_city"].unique())
selected_airline = st.sidebar.selectbox("Airline", ["All"] + list(data["airline"].unique()))

if selected_airline == "All":
    filtered_data = data[(data["source_city"] == source_city) &
                         (data["destination_city"] == destination_city)]
else:
    filtered_data = data[(data["source_city"] == source_city) &
                         (data["destination_city"] == destination_city) &
                         (data["airline"] == selected_airline)]

st.subheader("Exploratory Data Analysis")

# Countplot of Airlines
st.write("Countplot of Airlines")
airline_counts = filtered_data["airline"].value_counts()
st.bar_chart(airline_counts)

# Duration Histogram
st.write("Duration Histogram")
hist_values, bins = np.histogram(filtered_data["duration"], bins=24, range=(0, 24))
st.bar_chart(hist_values)

# Price vs Duration Scatterplot
st.write("Price vs. Duration Scatterplot (Altair)")
scatter_chart = alt.Chart(filtered_data).mark_circle().encode(
    x=alt.X('duration', title='Duration (hours)'),
    y=alt.Y('price', title='Price'),
    color=alt.Color('class:N', title='Class'),
    tooltip=['duration', 'price', 'class']
).properties(width=600, height=400)
st.altair_chart(scatter_chart)


col1, col2 = st.columns(2)
# Arrival Time Distribution
with col1:
    st.write("Arrival Time Distribution")
    arrival_time_counts = filtered_data["arrival_time"].value_counts()
    st.bar_chart(arrival_time_counts)
# Departure Time Distribution
with col2:
    st.write("Departure Time Distribution")
    departure_time_counts = filtered_data["departure_time"].value_counts()
    st.bar_chart(departure_time_counts)

# Correlation matrix
numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])
st.write("Correlation Heatmap")
correlation = numeric_data.corr()
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(correlation, cmap="Blues")  # Change the colormap to "viridis"
fig.colorbar(cax)
plt.xticks(range(len(numeric_data.columns)), numeric_data.columns, rotation=45)
plt.yticks(range(len(numeric_data.columns)), numeric_data.columns)
st.pyplot(fig)


# Data Preprocessing
X = filtered_data.drop(columns=["price"])
y = filtered_data["price"]

categorical_cols = ["airline", "flight", "source_city", "departure_time", "stops", "destination_city", "class"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
X_numeric = X.select_dtypes(include=["int64", "float64"])
X_encoded.reset_index(drop=True, inplace=True)
X_numeric.reset_index(drop=True, inplace=True)
X_preprocessed = pd.concat([X_encoded, X_numeric], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

models = {
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBoost Regressor": XGBRegressor(),
    "CatBoost Regressor": CatBoostRegressor(learning_rate=0.1, iterations=500)
}

st.subheader("Model Comparison")

@st.cache_data
def train_and_evaluate_models(selected_model):
    model = models[selected_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return y_test, y_pred, mse, r2

model_comparison_results = []

for selected_model in models:
    data_load_state = st.text('Loading data...')
    y_test, y_pred, mse, r2 = train_and_evaluate_models(selected_model)
    data_load_state.text("")
    model_comparison_results.append({
        "Model": selected_model,
        "Mean Squared Error": mse,
        "R-squared": r2
    })

model_comparison_df = pd.DataFrame(model_comparison_results)

st.write(model_comparison_df)

selected_model = st.selectbox("Select Model for Visualization", list(models.keys()))

st.write(f"Scatter Plot for {selected_model}")
scatter_data = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': train_and_evaluate_models(selected_model)[1]})
scatter_chart = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.5).encode(
    x='Actual Prices',
    y='Predicted Prices',
    tooltip=['Actual Prices', 'Predicted Prices']
).properties(
    width=700,
    height=400
).interactive()
st.altair_chart(scatter_chart)


st.title("Flight Price Prediction")


st.sidebar.header("Input Parameters")
airline = st.sidebar.selectbox("Airline", X["airline"].unique())
flight = st.sidebar.text_input("Flight Number", "SG-8709")
source_city = st.sidebar.selectbox("Source City", X["source_city"].unique())
departure_time = st.sidebar.selectbox("Departure Time", X["departure_time"].unique())
stops = st.sidebar.selectbox("Stops", X["stops"].unique())
destination_city = st.sidebar.selectbox("Destination City", X["destination_city"].unique())
class_ = st.sidebar.selectbox("Class", X["class"].unique())
duration = st.sidebar.number_input("Duration (hours)", min_value=0.0, max_value=10.0, value=2.0)
days_left = st.sidebar.number_input("Daysleft", min_value=0.0, max_value=100.0, value=2.0)

user_input = pd.DataFrame({
    "airline": [airline],
    "flight": [flight],
    "source_city": [source_city],
    "departure_time": [departure_time],
    "stops": [stops],
    "destination_city": [destination_city],
    "class": [class_],
    "duration": [duration],
    "days_left": [days_left]
})

user_input_encoded = pd.DataFrame(encoder.transform(user_input[categorical_cols]))
user_input_encoded.columns = encoder.get_feature_names_out(categorical_cols)

user_input_encoded["duration"] = duration
user_input_encoded["days_left"] = days_left 

selected_model = st.sidebar.selectbox("Select Model", [model["Model"] for model in model_comparison_results])
selected_model_obj = models[selected_model]

if st.sidebar.button("Predict"):
    if not hasattr(selected_model_obj, "is_fitted_") or not selected_model_obj.is_fitted_:
        selected_model_obj.fit(X_train, y_train)
    
    y_pred = selected_model_obj.predict(user_input_encoded)
    st.subheader("Predicted Price")
    st.write(f"The predicted price is: ${y_pred[0]:.2f}")

st.subheader("Model Comparison Results")
model_comparison_df = pd.DataFrame(model_comparison_results)
st.write(model_comparison_df)

if st.checkbox("Show Raw Data"):
    st.write(filtered_data)