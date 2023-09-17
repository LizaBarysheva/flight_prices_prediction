# Airline Flight Data EDA and Price Prediction App

This is a Streamlit web application that allows you to explore and analyze airline flight data and make predictions on flight prices.

## Getting Started

To get started, follow the instructions below to clone the repository and run the Streamlit app using Docker.

### Prerequisites

You need to have Docker installed on your system. If you don't have Docker, you can download and install it from [Docker's official website](https://docs.docker.com/get-docker/).

### Clone the Repository

```
git clone https://github.com/LizaBarysheva/flight_prices_prediction.git
```

### Build the Docker Image

Navigate to the project directory:

```
cd your-repo-name
```

Build the Docker image using the provided Dockerfile:

```
docker build -t flight-price-prediction-app .
```

### Run the Docker Container

Once the Docker image is built, you can run the Streamlit app in a Docker container:

```
docker run -p 8501:8501 flight-price-prediction-app
```

### Access the App

Open your web browser and visit http://localhost:8501 to access the Airline Flight Data EDA and Price Prediction App.

### Usage

Follow the on-screen instructions to filter data, explore visualizations, and make predictions on flight prices using the app.
