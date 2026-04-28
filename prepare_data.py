import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

def prepare_data():
    print("Loading and merging CSV files...")
    path = 'archive (2)'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    
    df = pd.concat(li, axis=0, ignore_index=True)
    
    # Cleaning
    print(f"Total movies loaded: {len(df)}")
    df.drop_duplicates(subset=['movie_id'], inplace=True)
    
    # Rename columns
    df.rename(columns={
        'movie_name': 'title',
        'description': 'overview',
        'rating': 'vote_average'
    }, inplace=True)
    
    # Handle missing values
    df.dropna(subset=['title', 'overview'], inplace=True)
    
    # Detect Language (Heuristic)
    print("Detecting languages...")
    hindi_keywords = [
        'Shah Rukh Khan', 'Salman Khan', 'Aamir Khan', 'Akshay Kumar', 'Deepika Padukone', 
        'Priyanka Chopra', 'Amitabh Bachchan', 'Hrithik Roshan', 'Ranbir Kapoor', 
        'Ranveer Singh', 'Alia Bhatt', 'Kareena Kapoor', 'Ajay Devgn', 'Katrina Kaif',
        'S.S. Rajamouli', 'Siddharth Anand', 'Karan Johar', 'Varun Dhawan', 'Anushka Sharma'
    ]
    
    def detect_lang(row):
        stars = str(row['star'])
        director = str(row['director'])
        if any(kd in stars for kd in hindi_keywords) or any(kd in director for kd in hindi_keywords):
            return 'Hindi'
        return 'English' # Defaulting to English as it's the majority in this dataset
    
    df['language'] = df.apply(detect_lang, axis=1)
    
    # Select top 15,000 movies by votes
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)
    df = df.sort_values(by='votes', ascending=False).head(15000).reset_index(drop=True)
    
    print(f"Final dataset size: {len(df)}")
    
    # Create tags for similarity
    df['tags'] = df['overview'] + " " + df['genre'] + " " + df['director'] + " " + df['star'] + " " + df['language']
    df['tags'] = df['tags'].apply(lambda x: x.lower() if isinstance(x, str) else "")
    
    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    
    # Similarity
    print("Calculating similarity matrix...")
    similarity = cosine_similarity(vectors)
    
    # Save artifacts
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
        
    print("Saving models...")
    pickle.dump(df.to_dict(), open('artifacts/movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
    
    print("Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
