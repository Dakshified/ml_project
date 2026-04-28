import urllib.request
import json

def test_tmdb(movie_id, title=None):
    tmdb_key = "8265bd1679663a7ea12ac168da84d2e8"
    
    print(f"Testing for ID: {movie_id}, Title: {title}")
    
    # 1. Try find
    if movie_id.startswith('tt'):
        try:
            url = f"https://api.themoviedb.org/3/find/{movie_id}?api_key={tmdb_key}&language=en-US&external_source=imdb_id"
            with urllib.request.urlopen(url) as response:
                res = json.loads(response.read().decode())
                if res.get('movie_results'):
                    path = res['movie_results'][0].get('poster_path')
                    if path:
                        print(f"Success (Find): https://image.tmdb.org/t/p/w500{path}")
                        return
        except Exception as e:
            print(f"Find failed: {e}")

    # 2. Try search
    if title:
        try:
            clean_title = title.split('(')[0].strip()
            # quote title for URL
            quoted_title = urllib.parse.quote(clean_title)
            url = f"https://api.themoviedb.org/3/search/movie?api_key={tmdb_key}&query={quoted_title}"
            with urllib.request.urlopen(url) as response:
                res = json.loads(response.read().decode())
                if res.get('results'):
                    path = res['results'][0].get('poster_path')
                    if path:
                        print(f"Success (Search): https://image.tmdb.org/t/p/w500{path}")
                        return
        except Exception as e:
            print(f"Search failed: {e}")
            
    print("Failed to find poster.")

if __name__ == "__main__":
    test_tmdb("tt0468569", "The Dark Knight")
    test_tmdb("nonexistent_id", "Inception")
