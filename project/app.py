import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# Function to perform clustering
def perform_clustering(data, n_clusters):
    numeric_cols = data.select_dtypes(include=["number"]).columns
    data_numeric = data[
        numeric_cols
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    data_numeric.fillna(data_numeric.mean(), inplace=True)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(data_scaled)
    clusters = kmeans.predict(data_scaled)

    data_clustered = data.copy()
    data_clustered["Cluster"] = clusters

    if data_numeric.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=data_numeric.iloc[:, 0],
            y=data_numeric.iloc[:, 1],
            hue=clusters,
            palette="viridis",
        )
        plt.title("K-means Clustering")
        st.pyplot()

    return data_clustered


def display_task1():
    st.header("Task 1: Clustering Results")
    data = load_data("train.xlsx")  # Replace with your actual file name
    n_clusters = st.slider("Select number of clusters:", 2, 10, 3)
    clustered_data = perform_clustering(data, n_clusters)
    st.subheader("Clustered Data:")
    st.dataframe(clustered_data)

    if st.checkbox("Explain Cluster for a Data Point"):
        data_point = st.selectbox("Select a Data Point:", data.index)
        explanation = f"Data point {data_point} belongs to cluster {clustered_data.loc[data_point, 'Cluster']}."
        st.write(explanation)


def display_task2():
    st.header("Task 2: Classification Results")
    X_train, y_train, X_test = load_classification_data("train.xlsx", "test.xlsx")
    predictions, train_accuracies = perform_classification(X_train, y_train, X_test)

    st.subheader("Train Accuracy:")
    st.write(train_accuracies)

    st.subheader("Predictions:")
    st.write(predictions)


def perform_classification(X_train, y_train, X_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    predictions = {}
    train_accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        predictions[name] = y_pred
        train_accuracies[name] = accuracy_score(y_train, clf.predict(X_train_scaled))

    return predictions, train_accuracies


def display_task3():
    st.header("Task 3: Raw Data Analysis Results")
    raw_data = load_raw_data("rawdata.xlsx")
    datewise_duration, pick_activities, place_activities = perform_raw_data_analysis(
        raw_data
    )

    st.subheader("Datewise Total Duration for Each Inside and Outside:")
    st.dataframe(datewise_duration)

    st.subheader("Datewise Number of Picking Activities:")
    st.write(pick_activities)

    st.subheader("Datewise Number of Placing Activities:")
    st.write(place_activities)


def perform_raw_data_analysis(raw_data):
    raw_data["date"] = pd.to_datetime(raw_data["date"])

    if (
        raw_data["time"].dtype == "O"
        or raw_data["time"].dt.to_pydatetime()[0].__class__ == datetime.time
    ):
        raw_data["datetime"] = pd.to_datetime(
            raw_data["date"].astype(str) + " " + raw_data["time"].astype(str)
        )
    else:
        raw_data["datetime"] = pd.to_datetime(raw_data["time"])

    raw_data["duration"] = raw_data["datetime"].diff().fillna(pd.Timedelta(seconds=0))
    raw_data = raw_data[raw_data["duration"] >= pd.Timedelta(seconds=0)]

    datewise_duration = (
        raw_data.groupby(["date", "position"])["duration"]
        .sum()
        .unstack(fill_value=pd.Timedelta(seconds=0))
    )

    pick_activities = raw_data[raw_data["activity"] == "picked"].groupby("date").size()
    place_activities = raw_data[raw_data["activity"] == "placed"].groupby("date").size()

    return datewise_duration, pick_activities, place_activities


def load_data(file_name):
    df = pd.read_excel(file_name)
    return df


def load_classification_data(train_file, test_file):
    X_train = pd.read_excel(train_file)
    X_test = pd.read_excel(test_file)
    y_train = X_train.pop("target")
    return X_train, y_train, X_test


def load_raw_data(file_name):
    raw_data = pd.read_excel(file_name)
    return raw_data


def main():
    st.title("Task 4: Integrated Data Analysis")
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
