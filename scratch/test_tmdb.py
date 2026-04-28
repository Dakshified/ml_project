import requests

movie_id = "tt0468569" # The Dark Knight
api_key = "8265bd1679663a7ea12ac168da84d2e8"
url = f"https://api.themoviedb.org/3/find/{movie_id}?api_key={api_key}&language=en-US&external_source=imdb_id"

try:
    res = requests.get(url, timeout=5)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
except Exception as e:
    print(f"Error: {e}")
