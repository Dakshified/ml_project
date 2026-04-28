import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def recover_similarity():
    print("Loading movie_dict.pkl...")
    if not os.path.exists('artifacts/movie_dict.pkl'):
        print("Error: artifacts/movie_dict.pkl not found!")
        return

    movies = pd.DataFrame(pickle.load(open('artifacts/movie_dict.pkl', 'rb')))
    
    print("Vectorizing tags...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags'].fillna('')).toarray()
    
    print("Calculating similarity matrix (this may take a minute)...")
    similarity = cosine_similarity(vectors)
    
    print("Saving similarity.pkl...")
    pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
    print("Recovery complete! You can now run the app.")

if __name__ == "__main__":
    recover_similarity()
