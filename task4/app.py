import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Function to load data for Task 1
def load_data(file_name):
    try:
        # Load data from the provided file
        data = pd.read_excel(file_name)
        return data
    except FileNotFoundError:
        st.error(f"File '{file_name}' not found. Please check the file path and name.")
        return None


# Function to perform clustering
def perform_clustering(data, n_clusters):
    # Handle missing values
    data = data.dropna()

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Fit K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)

    # Predict clusters
    clusters = kmeans.predict(data_scaled)

    # Add cluster labels to the original data
    data_clustered = data.copy()
    data_clustered["Cluster"] = clusters

    # Plot clusters if the data is 2D
    if data.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=data.iloc[:, 0], y=data.iloc[:, 1], hue=clusters, palette="viridis"
        )
        plt.title("K-means Clustering")
        st.pyplot()

    return data_clustered


# Function to display Task 1
def display_task1():
    st.header("Task 1: Clustering Results")

    # Specify the file names
    train_file = "train.xlsx"  # Replace with your train data file name

    # Load or generate data for demonstration
    data = load_data(train_file)

    if data is None:
        return

    # Perform clustering
    n_clusters = st.slider("Select number of clusters:", 2, 10, 3)
    clustered_data = perform_clustering(data, n_clusters)

    st.subheader("Clustered Data:")
    st.dataframe(clustered_data)

    # Show explanation for a data point
    if st.checkbox("Explain Cluster for a Data Point"):
        data_point = st.selectbox("Select a Data Point:", data.index)
        explanation = f"Data point {data_point} belongs to cluster {clustered_data.loc[data_point, 'Cluster']}."
        st.write(explanation)


# Function to perform classification
def perform_classification(X_train, y_train, X_test):
    # Scale the data using the StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize classifiers
    classifiers = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # Train and predict with each classifier
    predictions = {}
    train_accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        predictions[name] = y_pred
        train_accuracies[name] = accuracy_score(y_train, clf.predict(X_train_scaled))

    return predictions, train_accuracies


# Function to display Task 2
def display_task2():
    st.header("Task 2: Classification Results")

    # Specify the file names
    train_file = "train.xlsx"  # Replace with your train data file name
    test_file = "test.xlsx"  # Replace with your test data file name

    # Load or generate data for demonstration
    X_train, y_train, X_test = load_classification_data(train_file, test_file)

    # Perform classification
    predictions, train_accuracies = perform_classification(X_train, y_train, X_test)

    # Display results
    st.subheader("Train Accuracy:")
    st.write(train_accuracies)

    st.subheader("Predictions:")
    st.write(predictions)


# Function to load classification data
def load_classification_data(train_file, test_file):
    # Load training and test data
    train_data = pd.read_excel(train_file)
    test_data = pd.read_excel(test_file)

    # Separate features (X) and target variable (y)
    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]
    X_test = test_data

    return X_train, y_train, X_test


# Function to perform raw data analysis
def perform_raw_data_analysis(raw_data):
    # 1. Datewise total duration for each inside and outside
    raw_data["time"] = pd.to_datetime(raw_data["time"])
    datewise_duration = (
        raw_data.groupby(["date", "position"])["time"]
        .sum()
        .unstack(fill_value=pd.Timedelta(seconds=0))
    )

    # 2. Datewise number of picking and placing activities
    pick_activities = raw_data[raw_data["activity"] == "picked"].groupby("date").size()
    place_activities = raw_data[raw_data["activity"] == "placed"].groupby("date").size()

    return datewise_duration, pick_activities, place_activities


# Function to display Task 3
def display_task3():
    st.header("Task 3: Raw Data Analysis Results")

    # Specify the file name
    rawdata_file = "rawdata.xlsx"  # Replace with your rawdata file name

    # Load raw data
    raw_data = load_data(rawdata_file)

    if raw_data is None:
        return

    # Perform raw data analysis
    datewise_duration, pick_activities, place_activities = perform_raw_data_analysis(
        raw_data
    )

    # Display results
    st.subheader("Datewise Total Duration for Each Inside and Outside:")
    st.dataframe(datewise_duration)

    st.subheader("Datewise Number of Picking Activities:")
    st.write(pick_activities)

    st.subheader("Datewise Number of Placing Activities:")
    st.write(place_activities)


# Main function to run the Streamlit app
def main():
    st.title("Data Analysis Tasks")

    # Sidebar menu
    menu = ["Task 1: Clustering", "Task 2: Classification", "Task 3: Raw Data Analysis"]
    choice = st.sidebar.selectbox("Select Task", menu)

    if choice == "Task 1: Clustering":
        display_task1()
    elif choice == "Task 2: Classification":
        display_task2()
    elif choice == "Task 3: Raw Data Analysis":
        display_task3()
    else:
        st.error("Invalid Choice")


if __name__ == "__main__":
    main()
