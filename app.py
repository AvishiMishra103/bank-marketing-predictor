# app.py

# ------------------ IMPORT LIBRARIES ------------------
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIGURATION ------------------
# Sets the title, icon, and layout of the Streamlit app
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="💳",
    layout="wide"
)

# ------------------ SIDEBAR NAVIGATION ------------------
# Sidebar radio button for navigation between different sections
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dataset", "⚙️ Model Training", "📈 Results", "🧪 Try Prediction"])

# Sidebar for hyperparameter tuning of Decision Tree
st.sidebar.title("⚙️ Settings")
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)  # Controls how deep the decision tree can go
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])  # Splitting criteria

# ------------------ DATA LOADING FUNCTION ------------------
@st.cache_data  # Cache so the dataset isn’t reloaded every time the app refreshes
def load_and_merge():
    # Load all four datasets provided in UCI Bank Marketing repo
    df1 = pd.read_csv("bank.csv", sep=";")
    df2 = pd.read_csv("bank-full.csv", sep=";")
    df3 = pd.read_csv("bank-additional.csv", sep=";")
    df4 = pd.read_csv("bank-additional-full.csv", sep=";")
    
    # Merge them into a single dataset (stack vertically)
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    # Remove duplicate rows that appear in multiple datasets
    df = df.drop_duplicates()
    return df



# Load merged dataset
df = load_and_merge()

# ------------------ DATA PREPROCESSING ------------------
# Make a copy of dataset for encoding categorical features
df_encoded = df.copy()

# Encode categorical columns into numeric values
# DecisionTree in sklearn only works with numerical data
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":  # If column is categorical
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])



# Define features (X) and target variable (y)
X = df_encoded.drop("y", axis=1)  # Features are all columns except 'y'
y = df_encoded["y"]  # Target variable is 'y' (yes/no for subscription)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ MODEL TRAINING ------------------

# Create Decision Tree model with selected hyperparameters
clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

# Train the model on training data
clf.fit(X_train, y_train)

# Predict results on test set
y_pred = clf.predict(X_test)

# ------------------ PAGE LOGIC ------------------

# Section 1: Dataset Preview
if page == "📊 Dataset":
    st.title("📊 Bank Marketing Dataset")
    st.write(
        "This dataset comes from the **UCI Machine Learning Repository**. "
        "It contains demographic and behavioral data of customers, "
        "used to predict whether they will subscribe to a product/service."
    )

    # Show first 20 rows of dataset
    st.subheader("Preview")
    st.dataframe(df.head(20))

    # Show dataset shape
    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Section 2: Model Training Information
elif page == "⚙️ Model Training":
    st.title("⚙️ Decision Tree Training")
    st.write("We use a **Decision Tree Classifier** to model customer purchase predictions.")

    # Display how many samples are in training vs testing sets
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Testing Samples", len(X_test))

    # Show hyperparameters selected
    st.subheader("Hyperparameters")
    st.write(f"- Max Depth: **{max_depth}**")
    st.write(f"- Criterion: **{criterion}**")

# Section 3: Model Results
elif page == "📈 Results":
    st.title("📈 Model Performance")

    # Calculate and show accuracy
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Accuracy")
    st.success(f"Model Accuracy: {acc:.2f}")

    # Show confusion matrix as heatmap
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )
    st.pyplot(fig)

    # Show precision, recall, f1-score
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Show feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
    importance_df = importance_df.sort_values("Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

# Section 4: Try Prediction with User Input
elif page == "🧪 Try Prediction":
    st.title("🧪 Try Your Own Prediction")
    st.write("Enter customer details to predict if they will subscribe to the product/service.")

    input_data = {}
    col1, col2 = st.columns(2)  # Split form into two columns for cleaner layout

    # Loop through features and create input fields
    for i, col in enumerate(X.columns):
        if i % 2 == 0:  # Left column
            val = col1.number_input(
                f"{col}",
                float(df_encoded[col].min()),
                float(df_encoded[col].max()),
                float(df_encoded[col].mean())
            )
        else:  # Right column
            val = col2.number_input(
                f"{col}",
                float(df_encoded[col].min()),
                float(df_encoded[col].max()),
                float(df_encoded[col].mean())
            )
        input_data[col] = val

    # When button is clicked, predict
    if st.button("Predict Now"):
        user_df = pd.DataFrame([input_data])  # Convert user input to DataFrame
        pred = clf.predict(user_df)[0]  # Predict using trained model
        result = "Yes ✅ (Will Subscribe)" if pred == 1 else "No ❌ (Will Not Subscribe)"
        st.success(result)
