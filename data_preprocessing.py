import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset from the specified file path.

    The function performs the following steps:
    1. Loads the dataset from a CSV file.
    2. Drops unnecessary columns.
    3. Handles missing values by filling them with "Unknown" and dropping columns that are completely empty.
    4. Encodes categorical variables using Label Encoding.
    5. Vectorizes text data using TF-IDF vectorization.
    6. Splits the data into training and test sets.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: A tuple containing:
            - X_train, X_test, y_train_type2, y_test_type2, y_train_type3, y_test_type3, y_train_type4, y_test_type4:
              The training and test sets for features and labels.
            - label_encoders (dict): A dictionary containing the label encoders for categorical columns.
    """

    # Load the dataset from the specified CSV file
    df = pd.read_csv(file_path)

    # Remove any leading/trailing spaces in column names
    df.columns = df.columns.str.strip()  

    # Drop unnecessary columns that won't be used for model training
    df_cleaned = df.drop(columns=["Ticket id", "Interaction id", "Interaction date", "Mailbox", "Innso TYPOLOGY_TICKET"])
    df_cleaned.dropna(axis=1, how='all', inplace=True)

    # Fill any remaining missing values with the string "Unknown"
    df_cleaned = df_cleaned.fillna("Unknown")

    # Encode categorical variables using Label Encoding
    label_encoders = {} 
    for col in ["Type 1", "Type 2", "Type 3", "Type 4"]:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        label_encoders[col] = le

    # Vectorize text data using TF-IDF
    text_columns = ["Ticket Summary", "Interaction content"]
    vectorizers = {}
    tfidf_features = []

    # Process each text column with TF-IDF vectorizer (using a max of 100 features)
    for col in text_columns:
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(df_cleaned[col])
        vectorizers[col] = vectorizer

        # Convert the TF-IDF matrix to a DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"{col}_{i}" for i in range(tfidf_matrix.shape[1])],)
        tfidf_features.append(tfidf_df)

    # Merge the TF-IDF features with the cleaned numerical dataset
    df_numerical = pd.concat([df_cleaned.drop(columns=text_columns)] + tfidf_features, axis=1)

    # Split the dataset into features (X) and labels (y)
    X = df_numerical.drop(columns=["Type 2", "Type 3", "Type 4"])
    y_type2, y_type3, y_type4 = (
        df_numerical["Type 2"],
        df_numerical["Type 3"],
        df_numerical["Type 4"])

    # Split the data into training and test sets
    return (train_test_split(X, y_type2, y_type3, y_type4, test_size=0.2, random_state=42), label_encoders)
