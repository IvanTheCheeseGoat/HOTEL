import streamlit as st
import pandas as pd
from textblob import TextBlob
from rake_nltk import Rake
from io import BytesIO
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
import string
import joblib
import sqlite3
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set Streamlit page configuration
st.set_page_config(page_title="Hotel Review Sentiment Analysis", layout="wide")

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Initialize SQLite database
conn = sqlite3.connect('training_data.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review TEXT,
    sentiment TEXT
)
''')
conn.commit()

# Function to preprocess the text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Handle negations (e.g., "not good" -> "not_good")
    tokens = [f"{tokens[i]}_{tokens[i+1]}" if tokens[i].lower() == 'not' and i+1 < len(tokens) else tokens[i] for i in range(len(tokens))]
    # Remove punctuation and stopwords, and convert to lower case
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Function to extract key sentiments and keywords
def extract_key_sentiments_keywords(review):
    analysis = TextBlob(review)
    r = Rake()
    r.extract_keywords_from_text(review)
    keywords = ', '.join(r.get_ranked_phrases()[:5])  # Get top 5 keywords
    return analysis.sentiment, keywords

# Function to train or load the sentiment classifier with hyperparameter tuning
def train_or_load_model():
    # Load training data from the database
    df = pd.read_sql('SELECT * FROM reviews', conn)
    
    if 'sentiment' not in df.columns:
        st.error("Training data must include a 'sentiment' column for training the model.")
        return None, None
    
    df['review'] = df['review'].apply(preprocess_text)
    
    X = df['review']
    y = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0)
    
    # Check the minimum class count to set n_splits appropriately
    min_class_count = min(y.value_counts())
    n_splits = min(10, min_class_count) if min_class_count >= 2 else 2  # Ensure at least 2 splits
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), max_df=0.9, min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Apply SMOTE to balance the classes in the vectorized training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)
    
    # Define the model and hyperparameter grid
    model = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }
    
    # Use Stratified K-Fold with dynamic splits
    stratified_k_fold = StratifiedKFold(n_splits=n_splits)
    
    # Perform Grid Search with Stratified K-Fold Cross-Validation
    grid_search = GridSearchCV(model, param_grid, cv=stratified_k_fold, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_res, y_train_res)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(best_model, X_train_res, y_train_res, cv=stratified_k_fold)
    st.write(f"Stratified K-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    
    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return best_model, vectorizer

# Function to classify sentiment using the trained model
def classify_sentiment_model(review, model, vectorizer):
    review = preprocess_text(review)
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    
    # Additional rule-based check for obvious errors
    if 'not' in review or 'but' in review:
        analysis = TextBlob(review)
        if analysis.sentiment.polarity <= 0:
            return 'Negative'
    
    return 'Positive' if prediction == 1 else 'Negative'

# Streamlit app layout
st.title('Hotel Review Sentiment Analysis')
st.write('Input the source and upload an Excel file containing hotel reviews to get sentiment analysis.')

source = st.text_input("Source")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Load existing training data
training_data_file = st.file_uploader("Upload training data (optional, for improving the model)", type="xlsx")

# Initialize model and vectorizer
model = None
vectorizer = None

if training_data_file is not None:
    training_df = pd.read_excel(training_data_file)
    for _, row in training_df.iterrows():
        cursor.execute('INSERT INTO reviews (review, sentiment) VALUES (?, ?)', (row['Review'], row['Sentiment']))
    conn.commit()
    model, vectorizer = train_or_load_model()

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    if 'Review' in df.columns:
        st.write("Processing reviews, please wait...")

        df['Source'] = source
        df['Date'] = pd.to_datetime('today').date()

        # Extract sentiments and keywords with progress bar
        sentiments = []
        details = []
        keywords = []
        progress_bar = st.progress(0)
        for i, review in enumerate(df['Review']):
            if model and vectorizer:
                sentiment = classify_sentiment_model(review, model, vectorizer)
            else:
                analysis = TextBlob(review)
                sentiment = 'Positive' if analysis.sentiment.polarity > 0 else 'Negative'
            analysis, keyword = extract_key_sentiments_keywords(review)
            details.append(f'Polarity: {analysis.polarity}, Subjectivity: {analysis.subjectivity}')
            keywords.append(keyword)
            sentiments.append(sentiment)
            progress_bar.progress((i + 1) / len(df))

        df['Sentiment'] = sentiments
        df['Sentiment Details'] = details
        df['Keywords'] = keywords

        st.write("Processing complete.")
        st.write(df)
        
        # Display sentiment distribution as a bar chart
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        # Display a pie chart of sentiment distribution
        fig, ax = plt.subplots()
        sentiment_counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
        
        # Display a word cloud of keywords
        keyword_text = ' '.join(df['Keywords'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keyword_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # Display top 10 keywords
        keyword_series = pd.Series(' '.join(df['Keywords']).split(', ')).value_counts().head(10)
        st.table(keyword_series)

        # Display sentiment polarity histogram
        df['Polarity'] = df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
        plt.figure(figsize=(10, 5))
        plt.hist(df['Polarity'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Sentiment Polarity Distribution')
        plt.xlabel('Polarity')
        plt.ylabel('Frequency')
        st.pyplot(plt)
        
        # Display trend line for sentiment over time
        df['Date'] = pd.to_datetime(df['Date'])
        sentiment_trend = df.groupby(df['Date'].dt.to_period('D')).size().reset_index(name='Counts')
        sentiment_trend['Date'] = sentiment_trend['Date'].dt.to_timestamp()
        plt.figure(figsize=(10, 5))
        plt.plot(sentiment_trend['Date'], sentiment_trend['Counts'], marker='o')
        plt.title('Trend of Reviews Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Reviews')
        st.pyplot(plt)
        
        # Option to download as Excel
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
            processed_data = output.getvalue()
            return processed_data
        
        st.download_button(label="Download data as Excel", data=to_excel(df), file_name='reviews_with_sentiments.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        st.error("The uploaded file does not contain a 'Review' column. Please check your file and try again.")
else:
    st.info("Please upload an Excel file to proceed.")
