import pandas as pd
import random
import string

# Seed for reproducibility
random.seed(42)

# Common phrases for synthetic reviews
positive_phrases = [
    "excellent service", "friendly staff", "clean rooms", "great location", "comfortable bed",
    "delicious breakfast", "amazing view", "wonderful experience", "highly recommend", "best hotel",
    "fantastic amenities", "luxurious feel", "well-maintained", "exceptional hospitality", "perfect stay"
]

negative_phrases = [
    "poor service", "rude staff", "dirty rooms", "bad location", "uncomfortable bed",
    "terrible breakfast", "horrible view", "awful experience", "would not recommend", "worst hotel",
    "outdated amenities", "cheap feel", "poorly maintained", "unfriendly hospitality", "bad stay"
]

def generate_review(phrases, sentiment, num_reviews=5000):
    reviews = []
    for _ in range(num_reviews):
        review_length = random.randint(5, 15)  # Vary the length of reviews
        review = ' '.join(random.choices(phrases, k=review_length))
        reviews.append([review, sentiment])
    return reviews

# Generate synthetic reviews
positive_reviews = generate_review(positive_phrases, 'Positive', num_reviews=5000)
negative_reviews = generate_review(negative_phrases, 'Negative', num_reviews=5000)

# Combine and shuffle the reviews
all_reviews = positive_reviews + negative_reviews
random.shuffle(all_reviews)

# Create a DataFrame
reviews_df = pd.DataFrame(all_reviews, columns=['Review', 'Sentiment'])

# Save to Excel
reviews_df.to_excel('synthetic_hotel_reviews.xlsx', index=False)
print("Synthetic dataset created and saved to 'synthetic_hotel_reviews.xlsx'.")
