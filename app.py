import streamlit as st
import pandas as pd
from textblob import TextBlob
from rake_nltk import Rake
from io import BytesIO
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
import sqlite3
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set Streamlit page configuration
st.set_page_config(page_title="Hotel Review Sentiment Analysis", layout="wide")

# Download NLTK data
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
nltk.download('stopwords', download_dir=os.path.join(os.path.dirname(__file__), 'nltk_data'))
nltk.download('punkt', download_dir=os.path.join(os.path.dirname(__file__), 'nltk_data'))

stop_words = set(stopwords.words('english'))

# Initialize SQLite database
db_path = 'training_data.db'
if not os.path.exists(db_path):
    st.error(f"Database file not found at {db_path}")
else:
    st.success(f"Database found at {db_path}")

try:
    conn = sqlite3.connect(db_path)
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
    st.success("Database connected and table ensured.")
except Exception as e:
    st.error(f"Failed to connect to the database: {e}")

# Function to preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [f"{tokens[i]}_{tokens[i+1]}" if tokens[i].lower() == 'not' and i+1 < len(tokens) else tokens[i] for i in range(len(tokens))]
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Function to extract key sentiments and keywords
def extract_key_sentiments_keywords(review):
    analysis = TextBlob(review)
    r = Rake()
    r.extract_keywords_from_text(review)
    keywords = ', '.join(r.get_ranked_phrases()[:5])
    return analysis.sentiment, keywords

# Function to train or load the sentiment classifier with hyperparameter tuning
def train_or_load_model():
    try:
        df = pd.read_sql('SELECT * FROM reviews', conn)
    except Exception as e:
        st.error(f"Failed to read from database: {e}")
        return None, None
    
    if 'sentiment' not in df.columns:
        st.error("Training data must include a 'sentiment' column for training the model.")
        return None, None
    
    df['review'] = df['review'].apply(preprocess_text)
    
    X = df['review']
    y = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0)
    
    min_class_count = min(y.value_counts())
    n_splits = min(10, min_class_count) if min_class_count >= 2 else 2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), max_df=0.9, min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)
    
    model = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }
    
    stratified_k_fold = StratifiedKFold(n_splits=n_splits)
    
    grid_search = GridSearchCV(model, param_grid, cv=stratified_k_fold, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_res, y_train_res)
    
    best_model = grid_search.best_estimator_
    
    cv_scores = cross_val_score(best_model, X_train_res, y_train_res, cv=stratified_k_fold)
    st.write(f"Stratified K-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    
    y_pred = best_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return best_model, vectorizer

def classify_sentiment_model(review, model, vectorizer):
    review = preprocess_text(review)
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    
    if 'not' in review or 'but' in review:
        analysis = TextBlob(review)
        if analysis.sentiment.polarity <= 0:
            return 'Negative'
    
    return 'Positive' if prediction == 1 else 'Negative'

st.title('Hotel Review Sentiment Analysis')
st.write('Input the source and upload an Excel file containing hotel reviews to get sentiment analysis.')

source = st.text_input("Source")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

training_data_file = st.file_uploader("Upload training data (optional, for improving the model)", type="xlsx")

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
        
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        fig, ax = plt.subplots()
        sentiment_counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
        
        keyword_text = ' '.join(df['Keywords'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keyword_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        keyword_series = pd.Series(' '.join(df['Keywords']).split(', ')).value_counts().head(20)
        st.bar_chart(keyword_series)
        
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label="Download results as Excel",
            data=output,
            file_name="hotel_reviews_with_sentiments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Uploaded file does not contain 'Review' column. Please check the file and try again.")
