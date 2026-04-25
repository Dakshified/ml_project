# CineMatch AI Pro - Evolution Edition
import pickle
from datetime import datetime
import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from db_models import SessionLocal, User, UserDNA, UserPreferences, WatchHistory, MovieRating, Watchlist

# Initialize DB Session
db_session = SessionLocal()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Movie DNA | Neural Recommender",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- GLOBAL STYLES ---
def inject_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800;900&family=Pacifico&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        /* Base App Styling - Darker with neon tint */
        @keyframes glowingBackground {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stApp { 
            background: linear-gradient(-45deg, #050508, #110022, #001122, #050508);
            background-size: 400% 400%;
            animation: glowingBackground 15s ease infinite;
            color: #e0e0e0; 
            font-family: 'Outfit', sans-serif; 
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6, .stMarkdown p { font-family: 'Outfit', sans-serif; }
        
        /* Unified Header */
        .hero-title {
            font-size: 4rem !important;
            font-weight: 900 !important;
            text-align: center;
            background: linear-gradient(135deg, #FF007A, #7928CA, #00C9FF, #00FF87);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px !important;
            letter-spacing: -1px;
            animation: gradient-shift 5s ease infinite;
            text-shadow: 0px 0px 30px rgba(255, 0, 122, 0.5);
        }
        .hero-subtitle {
            text-align:center; 
            color:#00FF87; 
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 30px;
            letter-spacing: 2px;
            text-transform: uppercase;
            text-shadow: 0px 0px 10px rgba(0, 255, 135, 0.4);
        }

        /* Animations */
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Login Card & Containers */
        div[data-testid="stForm"] {
            background: rgba(15, 15, 25, 0.7) !important;
            backdrop-filter: blur(20px) !important;
            border-radius: 24px !important;
            padding: 40px !important;
            border: 1px solid rgba(0, 201, 255, 0.3) !important;
            margin-bottom: 30px !important;
            box-shadow: 0 10px 50px 0 rgba(0, 201, 255, 0.15) !important;
            position: relative !important;
            overflow: hidden !important;
        }

        .category-header {
            font-size: 1.6rem;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 3px;
            font-weight: 800;
            margin: 40px 0 20px 0;
            display: flex;
            align-items: center;
            gap: 15px;
            text-shadow: 0px 0px 10px rgba(255,255,255,0.3);
        }
        .category-header::before {
            content: '';
            display: inline-block;
            width: 10px;
            height: 28px;
            background: linear-gradient(to bottom, #E100FF, #00C9FF);
            border-radius: 5px;
            box-shadow: 0px 0px 10px #E100FF;
        }

        /* Movie Cards */
        .movie-card {
            background: rgba(10, 10, 15, 0.9);
            border-radius: 20px;
            padding: 12px;
            height: 520px;
            display: flex;
            flex-direction: column;
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            margin-bottom: 10px;
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.6s ease-out forwards;
        }
        
        .movie-card:hover {
            transform: translateY(-15px) scale(1.03);
            border-color: #FF007A !important;
            box-shadow: 0 15px 40px rgba(255, 0, 122, 0.4), 0 0 20px rgba(0, 201, 255, 0.2);
            z-index: 10;
        }

        .card-img {
            width: 100%;
            height: 320px;
            object-fit: cover;
            border-radius: 12px;
            z-index: 1;
        }

        .card-content {
            z-index: 2;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            padding: 10px 5px 0 5px;
            height: 130px;
            overflow: hidden;
        }

        .card-title {
            font-size: 1.2rem;
            font-weight: 900;
            color: #fff;
            margin: 5px 0;
            line-height: 1.2;
            min-height: 40px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-shadow: 0 0 5px rgba(255,255,255,0.3);
        }

        .tag-logic {
            background: rgba(0, 201, 255, 0.1);
            color: #00C9FF;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            border: 1px solid rgba(0, 201, 255, 0.3);
            white-space: nowrap;
        }

        .watched-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(0, 255, 135, 0.95);
            color: #050508;
            padding: 6px 12px;
            border-radius: 12px;
            font-weight: 900;
            font-size: 0.85rem;
            z-index: 10;
            box-shadow: 0 0 15px rgba(0, 255, 135, 0.6);
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Navbar & Neon Buttons */
        .stButton button {
            background: linear-gradient(90deg, #FF007A, #7928CA) !important;
            border: none !important;
            color: #fff !important;
            box-shadow: 0 4px 15px rgba(255, 0, 122, 0.4) !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            font-size: 0.9rem !important;
            font-weight: 800 !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Neon Hover Effects */
        .stButton button:hover {
            box-shadow: 0 8px 25px rgba(255, 0, 122, 0.6) !important;
            background: linear-gradient(90deg, #FF007A, #00C9FF) !important;
            transform: translateY(-3px);
            color: #fff !important;
        }

        /* Form Submits */
        .stFormSubmitButton button {
            background: linear-gradient(90deg, #FF007A, #7928CA) !important;
            border: none !important;
            color: #fff !important;
            box-shadow: 0 4px 15px rgba(255, 0, 122, 0.4) !important;
        }
        .stFormSubmitButton button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 8px 25px rgba(255, 0, 122, 0.6) !important;
        }

        /* Inputs */
        .stSelectbox label, .stTextInput label, .stSlider label { 
            color: #00C9FF !important; 
            letter-spacing: 1.5px;
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 0.85rem !important;
        }
        
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            border: 1px solid rgba(0, 201, 255, 0.3) !important;
            background: rgba(10, 10, 15, 0.8) !important;
            color: #fff !important;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
        }
        
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
            border-color: #00FF87 !important;
            box-shadow: 0 0 10px rgba(0, 255, 135, 0.3) !important;
        }
        
        /* Dialog Container (Bounce up card simulation) */
        div[data-testid="stDialog"] {
            background: rgba(10, 10, 15, 0.95) !important;
            border: 2px solid #00C9FF !important;
            border-radius: 20px !important;
            box-shadow: 0 0 50px rgba(0, 201, 255, 0.4) !important;
        }
        
        /* Hide default sidebar button to enforce no sidebar look */
        [data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        
        hr { border-color: rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- CACHED IMAGE FETCH ---
@st.cache_data(show_spinner=False)
def get_poster(movie_id, title=None):
    # Multiple API keys to avoid limit issues
    keys = ["df06542d", "5d8628f2", "bb736f1c"]
    
    for key in keys:
        # 1. Try OMDb by IMDb ID (most accurate)
        try:
            url = f"http://www.omdbapi.com/?i={movie_id}&apikey={key}"
            res = requests.get(url, timeout=4).json()
            if res.get('Response') == 'True' and res.get('Poster') and res['Poster'] != "N/A":
                return res['Poster']
        except: pass
        
        # 2. Try OMDb by Title (fallback)
        if title:
            try:
                url = f"http://www.omdbapi.com/?t={title}&apikey={key}"
                res = requests.get(url, timeout=4).json()
                if res.get('Response') == 'True' and res.get('Poster') and res['Poster'] != "N/A":
                    return res['Poster']
            except: pass
            
    # 3. Fallback to TMDB (in case OMDb is down but TMDB works)
    try:
        url = f"https://api.themoviedb.org/3/find/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&external_source=imdb_id"
        res = requests.get(url, timeout=3).json()
        if res.get('movie_results'): return "https://image.tmdb.org/t/p/w500/" + res['movie_results'][0]['poster_path']
    except: pass
    
    return "https://placehold.co/500x750/1a1a24/444455.png?text=Poster+Unavailable"

# --- DATA LOAD ---
@st.cache_resource
def load_all_data():
    if not os.path.exists('artifacts/movie_dict.pkl'): return None, None, None
    m = pd.DataFrame(pickle.load(open('artifacts/movie_dict.pkl', 'rb')))
    s = pickle.load(open('artifacts/similarity.pkl', 'rb'))
    cv = CountVectorizer(max_features=5000, stop_words='english')
    v = cv.fit_transform(m['tags']).toarray()
    
    m['vote_average'] = pd.to_numeric(m['vote_average'], errors='coerce').fillna(0)
    
    # ML SYLLABUS INTEGRATION: Clustering & Regression
    kmeans = KMeans(n_clusters=20, random_state=42)
    m['ml_cluster'] = kmeans.fit_predict(v)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(v, m['vote_average'].values)
    m['predicted_rating'] = np.clip(ridge.predict(v), 1, 10).round(1)
    
    if 'release_date' in m.columns:
        m['year'] = pd.to_datetime(m['release_date'], errors='coerce').dt.year
    elif 'year' not in m.columns:
        m['year'] = 2000
    m['year'] = m['year'].fillna(2000)
    
    return m, s, (cv, v)

movies, similarity, nlp_engine = load_all_data()

# --- INITIALIZE SESSION STATE ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login'
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Discover"
if 'explore_section' not in st.session_state:
    st.session_state.explore_section = None

# --- AUTHENTICATION LOGIC ---
def login_user(email, password):
    user = db_session.query(User).filter(User.email == email).first()
    if user and user.check_password(password):
        st.session_state.user_id = user.id
        st.session_state.current_view = 'dashboard'
        return True
    return False

def signup_user(full_name, email, password, age, country, language):
    if db_session.query(User).filter(User.email == email).first(): return None
    new_user = User(full_name=full_name, email=email, age=age, country=country, preferred_language=language)
    new_user.set_password(password)
    db_session.add(new_user)
    db_session.commit()
    return new_user.id

def complete_onboarding(user_id, genres, favorite_movies, language, mood, runtime):
    prefs = UserPreferences(user_id=user_id, favorite_genres=",".join(genres), favorite_languages=language, mood_preferences=mood, runtime_preference=runtime)
    db_session.add(prefs)
    
    genre_scores = {g: 0.5 for g in genres}
    actor_scores = {}
    director_scores = {}
    
    for title in favorite_movies:
        movie_row = movies[movies['title'] == title].iloc[0]
        if pd.notna(movie_row['genre']):
            for g in movie_row['genre'].split(','):
                g = g.strip()
                genre_scores[g] = genre_scores.get(g, 0) + 0.3
        if pd.notna(movie_row['star']):
            for a in str(movie_row['star']).split(','):
                a = a.strip()
                actor_scores[a] = actor_scores.get(a, 0) + 0.5
        if pd.notna(movie_row['director']):
            d = str(movie_row['director']).strip()
            director_scores[d] = director_scores.get(d, 0) + 0.8
            
    genre_scores = {k: round(min(v, 1.0), 2) for k, v in genre_scores.items()}
    actor_scores = {k: round(min(v, 1.0), 2) for k, v in actor_scores.items()}
    director_scores = {k: round(min(v, 1.0), 2) for k, v in director_scores.items()}
    mood_scores = {f"Mood_{mood.replace(' ', '_')}": 0.8, f"Runtime_{runtime}": 0.8}
    
    dna = UserDNA(user_id=user_id, genre_vector_json=json.dumps(genre_scores), actor_vector_json=json.dumps(actor_scores), director_vector_json=json.dumps(director_scores), mood_scores_json=json.dumps(mood_scores), genre_snapshot_json=json.dumps(genre_scores))
    db_session.add(dna)
    db_session.commit()

def record_watch_and_update_dna(user_id, movie_id, movie_title):
    watch = db_session.query(WatchHistory).filter_by(user_id=user_id, movie_id=movie_id).first()
    if watch:
        watch.watch_count += 1
        watch.watched_at = datetime.utcnow()
    else:
        watch = WatchHistory(user_id=user_id, movie_id=movie_id, movie_title=movie_title)
        db_session.add(watch)
    
    dna = db_session.query(UserDNA).filter_by(user_id=user_id).first()
    if dna:
        current_genres = json.loads(dna.genre_vector_json)
        dna.genre_snapshot_json = dna.genre_vector_json
        movie_row = movies[movies['movie_id'] == movie_id].iloc[0]
        if pd.notna(movie_row['genre']):
            for g in movie_row['genre'].split(','):
                g = g.strip()
                current_genres[g] = round(min(current_genres.get(g, 0) + 0.1, 1.0), 2)
        dna.genre_vector_json = json.dumps(current_genres)
        dna.updated_at = datetime.utcnow()
    db_session.commit()

def rate_movie(user_id, movie_id, movie_title, rating):
    existing = db_session.query(MovieRating).filter_by(user_id=user_id, movie_id=movie_id).first()
    if existing:
        existing.rating = rating
    else:
        new_rating = MovieRating(user_id=user_id, movie_id=movie_id, movie_title=movie_title, rating=rating)
        db_session.add(new_rating)
        
    # --- ML SYLLABUS: Reinforcement Learning (Q-Value Update Approximation) ---
    # We treat the user's rating as an Environment Reward signal to tune the Policy (DNA vector weights)
    dna = db_session.query(UserDNA).filter_by(user_id=user_id).first()
    if dna:
        # Determine the Reward Signal (alpha learning rate)
        if rating >= 4: reward_alpha = 0.25      # Positive Reinforcement
        elif rating == 3: reward_alpha = 0.05    # Neutral/Slight Positive
        else: reward_alpha = -0.15               # Negative Reinforcement / Penalty
        
        current_genres = json.loads(dna.genre_vector_json)
        current_actors = json.loads(dna.actor_vector_json)
        current_directors = json.loads(dna.director_vector_json)
        
        movie_row = movies[movies['movie_id'] == movie_id].iloc[0]
        
        # Policy Update Rule: Q_new = Q_old + alpha * Reward
        if pd.notna(movie_row['genre']):
            for g in movie_row['genre'].split(','):
                g = g.strip()
                old_val = current_genres.get(g, 0.5)
                current_genres[g] = round(max(0.0, min(old_val + reward_alpha, 1.0)), 2)
                
        if pd.notna(movie_row['star']):
            for a in str(movie_row['star']).split(','):
                a = a.strip()
                old_val = current_actors.get(a, 0.1)
                current_actors[a] = round(max(0.0, min(old_val + reward_alpha, 1.0)), 2)
                
        if pd.notna(movie_row['director']):
            d = str(movie_row['director']).strip()
            old_val = current_directors.get(d, 0.1)
            current_directors[d] = round(max(0.0, min(old_val + (reward_alpha * 1.5), 1.0)), 2) # Heavily weigh director penalty/reward
            
        dna.genre_snapshot_json = dna.genre_vector_json # Save old state
        dna.genre_vector_json = json.dumps(current_genres)
        dna.actor_vector_json = json.dumps(current_actors)
        dna.director_vector_json = json.dumps(current_directors)
        dna.updated_at = datetime.utcnow()
        
    db_session.commit()

# --- VIEWS ---
if st.session_state.user_id is None:
    st.markdown("<h1 class='hero-title'>Movie DNA</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='hero-subtitle'>Neural Recommender Gateway</h3>", unsafe_allow_html=True)
    
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        if st.session_state.current_view == 'login':
            st.subheader("Login to Database")
            with st.form("login_form"):
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Access Profile")
                if submit:
                    if login_user(email, password):
                        st.toast("Welcome to Movie DNA! 🧬", icon="✨")
                        st.balloons()
                        st.success("Authentication successful! Loading neural link...")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Access denied.")
            st.write("Don't have a profile?")
            if st.button("Initialize New Profile"):
                st.session_state.current_view = 'signup'
                st.rerun()

        elif st.session_state.current_view == 'signup':
            st.subheader("Initialize Neural Profile")
            with st.form("signup_form"):
                full_name = st.text_input("Full Name *")
                email = st.text_input("Email Address *")
                password = st.text_input("Password *", type="password")
                c1, c2 = st.columns(2)
                with c1: age = st.number_input("Age (Optional)", min_value=10, max_value=120, value=25)
                with c2: country = st.text_input("Country *")
                language = st.selectbox("Preferred Language", ["English", "Hindi", "Spanish", "French", "Other"])
                
                submit = st.form_submit_button("Create Profile")
                if submit:
                    if full_name and email and password and country:
                        new_user_id = signup_user(full_name, email, password, age, country, language)
                        if new_user_id:
                            st.session_state.user_id = new_user_id
                            st.session_state.current_view = 'onboarding'
                            st.rerun()
                        else:
                            st.error("Email is already registered in the system.")
                    else:
                        st.error("Please fill in all required fields marked with *")
            if st.button("Return to Gateway"):
                st.session_state.current_view = 'login'
                st.rerun()
    st.stop()

# User is logged in!
current_user = db_session.query(User).get(st.session_state.user_id)
watched_ids = {w.movie_id for w in db_session.query(WatchHistory).filter_by(user_id=current_user.id).all()}

if st.session_state.current_view == 'onboarding':
    st.markdown("<h1 class='hero-title'>Neural Calibration</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='hero-subtitle'>Let's generate your unique Movie DNA</h3>", unsafe_allow_html=True)
    
    with st.form("onboarding_form"):
        st.subheader("Step 1: Core Preferences")
        col1, col2 = st.columns(2)
        with col1:
            genres = st.multiselect("Favorite Genres", ["Action", "Comedy", "Romance", "Thriller", "Sci-Fi", "Drama", "Horror", "Adventure"])
            language = st.selectbox("Preferred Content Language", sorted(movies['language'].unique()))
        with col2:
            mood = st.selectbox("Current Mood Preference", ["Feel good", "Emotional", "Dark", "Motivational", "Mind-bending"])
            runtime_map = {"Short (< 90m)": 90, "Medium (90-120m)": 120, "Long (> 120m)": 180}
            runtime = st.selectbox("Preferred Runtime", list(runtime_map.keys()))
            
        st.divider()
        st.subheader("Step 2: Database Anchors")
        st.write("Select 3 movies you absolutely love to build your genetic vector.")
        fav_movies = st.multiselect("Search Movies", movies['title'].values, max_selections=3)
        
        submit = st.form_submit_button("Generate DNA Profile 🚀")
        if submit:
            if len(fav_movies) == 3 and genres:
                with st.spinner("Compiling Neural Genetic Vectors..."):
                    complete_onboarding(current_user.id, genres, fav_movies, language, mood, runtime_map[runtime])
                st.session_state.current_view = 'dashboard'
                st.rerun()
            else:
                st.error("Please select at least 1 genre and exactly 3 anchor movies.")
    st.stop()

# --- NAVBAR AT THE VERY TOP ---
st.write("") # small spacing
n1, n2, n3, n4, n5, n6 = st.columns(6)
with n1:
    if st.button("DISCOVER", icon=":material/explore:", use_container_width=True): st.session_state.current_tab = "Discover"; st.rerun()
with n2:
    if st.button("NEURAL SEARCH", icon=":material/search:", use_container_width=True): st.session_state.current_tab = "Neural Search"; st.rerun()
with n3:
    if st.button("DIRECTOR'S LENS", icon=":material/movie_creation:", use_container_width=True): st.session_state.current_tab = "Director's Lens"; st.rerun()
with n4:
    if st.button("CAST UNIVERSE", icon=":material/groups:", use_container_width=True): st.session_state.current_tab = "Cast Universe"; st.rerun()
with n5:
    if st.button("WATCHLIST", icon=":material/bookmark:", use_container_width=True): st.session_state.current_tab = "Watchlist"; st.rerun()
with n6:
    if st.button("LOGOUT", icon=":material/logout:", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.current_view = 'login'
        st.rerun()

active_tabs = ["Discover", "Neural Search", "Director's Lens", "Cast Universe", "Watchlist"]
active_idx = active_tabs.index(st.session_state.current_tab) + 1 if st.session_state.current_tab in active_tabs else 1

st.markdown(f"""
<style>
div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child({active_idx}) .stButton button {{
    box-shadow: 0 0 25px #00C9FF, inset 0 0 15px rgba(255,255,255,0.5) !important;
    background: linear-gradient(90deg, #00C9FF, #00FF87) !important;
    transform: translateY(-3px);
}}
</style>
""", unsafe_allow_html=True)

st.divider()

# --- DASHBOARD HEADING ---
st.markdown("<h1 class='hero-title'>Movie DNA</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='hero-subtitle'>Experience the Next Evolution of Entertainment</h3>", unsafe_allow_html=True)

if movies is None:
    st.error("System initializing. Please ensure data models exist in the 'artifacts' folder.")
    st.stop()


# --- DIALOG COMPONENT ---
@st.dialog("🎬 Cinematic Details", width="large")
def movie_details_dialog(movie_id, title):
    row = movies[movies['movie_id'] == movie_id].iloc[0]
    poster = get_poster(movie_id)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(poster, use_container_width=True)
    with col2:
        st.markdown(f"<h2 style='color: #00FF87; text-shadow: 0 0 10px rgba(0,255,135,0.5); font-weight: 900;'>{title}</h2>", unsafe_allow_html=True)
        st.write(f"**⭐ Rating:** {row['vote_average']}/10")
        st.write(f"**📅 Year:** {int(row['year'])}")
        if pd.notna(row.get('genre')):
            st.write(f"**🎭 Genre:** {row['genre']}")
        if pd.notna(row.get('director')):
            st.write(f"**🎥 Director:** {row['director']}")
        if pd.notna(row.get('star')):
            st.write(f"**⭐ Actors:** {row['star']}")
            
        st.divider()
        st.markdown("#### Actions")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶ WATCH NOW", key=f"d_watch_{movie_id}", use_container_width=True):
                record_watch_and_update_dna(current_user.id, movie_id, title)
                st.success(f"Started watching {title}!")
        with b2:
            if st.button("➕ ADD TO WATCHLIST", key=f"d_list_{movie_id}", use_container_width=True):
                existing = db_session.query(Watchlist).filter_by(user_id=current_user.id, movie_id=movie_id).first()
                if not existing:
                    new_item = Watchlist(user_id=current_user.id, movie_id=movie_id, movie_title=title)
                    db_session.add(new_item)
                    db_session.commit()
                    st.success(f"Added to Watchlist!")
                else:
                    st.info("Already in Watchlist.")
                    
        rating = st.select_slider("Rate Movie", options=[1,2,3,4,5], key=f"d_rate_{movie_id}")
        if st.button("Submit Rating", key=f"d_btn_rate_{movie_id}", use_container_width=True):
            rate_movie(current_user.id, movie_id, title, rating)
            st.success(f"Rated {title} as {rating} stars!")

# --- HELPER COMPONENT ---
def render_movie_card(row, match_score=None, highlight_tags=None, border_color="rgba(0, 201, 255, 0.3)", key_prefix=""):
    with st.container():
        poster = get_poster(row.movie_id, title=row.title)
        if highlight_tags:
            tags_html = "".join([f"<span class='tag-logic'>{s}</span>" for s in highlight_tags])
        elif pd.notna(row.genre):
            genres = [g.strip() for g in row.genre.split(',')][:3]
            tags_html = "".join([f"<span class='tag-logic'>{g}</span>" for g in genres])
        else:
            tags_html = ""
            
        watched_html = ""
        if row.movie_id in watched_ids:
            watched_html = "<div class='watched-badge'><i class='bi bi-check-circle-fill'></i> WATCHED</div>"
        
        score_html = ""
        if match_score:
            score_html = f"<div style='color:#00FF87; font-size:0.9rem; font-weight:800; margin-bottom:5px; text-shadow: 0 0 5px #00FF87;'><i class='bi bi-lightning-charge-fill'></i> {match_score}% MATCH</div>"
            
        # IMPORTANT: No leading whitespace before HTML tags to prevent Streamlit from escaping it as markdown code blocks.
        html_header = f"<div class='movie-card' style='border-color: {border_color};'>{watched_html}<img src='{poster}' class='card-img' loading='lazy'><div class='card-content'><div class='card-title'>{row.title}</div><div style='color:#a0aec0; font-size:0.85rem; margin-bottom:8px; font-weight:600;'><i class='bi bi-star-fill' style='color:#FFD700;'></i> {row.vote_average:.1f}/10 • {int(row.year)}</div>{score_html}<div style='min-height:40px; display:flex; flex-wrap:wrap; gap:4px; align-content: flex-start; margin-bottom: 10px;'>{tags_html}</div></div></div>"
        
        st.markdown(html_header, unsafe_allow_html=True)
        
        if st.button("VIEW DETAILS", icon=":material/visibility:", key=f"{key_prefix}btn_{row.movie_id}", use_container_width=True):
            movie_details_dialog(row.movie_id, row.title)
        st.write("")


# --- TABS LOGIC ---
if st.session_state.current_tab == "Discover":
    # Global Filters at top instead of sidebar
    with st.container(border=True):
        st.markdown("<h4 style='color:#E100FF; margin-top:0;'><i class='bi bi-sliders2'></i> Neural Parameters</h4>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1: selected_lang = st.selectbox("🌍 Global Language", sorted(movies['language'].unique()))
        all_genres = set()
        for gs in movies['genre'].dropna(): 
            for g in gs.split(','): all_genres.add(g.strip())
        with f2: selected_genre = st.selectbox("🎭 Genre Focus", ["All"] + sorted(list(all_genres)))
        with f3: min_rating = st.slider("⭐ Minimum Rating Limit", min_value=0.0, max_value=10.0, value=6.0, step=0.5)
        
    filtered_df = movies[(movies['language'] == selected_lang) & (movies['vote_average'] >= min_rating)]
    if selected_genre != "All": filtered_df = filtered_df[filtered_df['genre'].str.contains(selected_genre, na=False)]

    user_dna = db_session.query(UserDNA).filter(UserDNA.user_id == current_user.id).first()
    user_prefs = db_session.query(UserPreferences).filter(UserPreferences.user_id == current_user.id).first()
    
    if user_dna:
        g_vec = json.loads(user_dna.genre_vector_json) if user_dna.genre_vector_json else {}
        a_vec = json.loads(user_dna.actor_vector_json) if user_dna.actor_vector_json else {}
        d_vec = json.loads(user_dna.director_vector_json) if user_dna.director_vector_json else {}
        pref_lang = user_prefs.favorite_languages if user_prefs else ""
        
        # Evolution DNA visual
        st.markdown(f"<p class='category-header'>🧬 Your Extracted Movie DNA</p>", unsafe_allow_html=True)
        g_data = pd.DataFrame(list(g_vec.items()), columns=['Genre', 'Score'])
        fig = px.line_polar(g_data, r='Score', theta='Genre', line_close=True, color_discrete_sequence=['#00FF87'])
        fig.update_polars(radialaxis_visible=False, bgcolor="rgba(0,0,0,0)")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

        max_votes = movies['votes'].max()
        def compute_dna_score(row):
            dna = 0.0
            if pd.notna(row.get('genre')):
                m_genres = [g.strip() for g in row['genre'].split(',')]
                g_score = sum([g_vec.get(g, 0) for g in m_genres]) / max(len(m_genres), 1)
                dna += min(g_score, 0.4)
            if pd.notna(row.get('star')):
                m_stars = [s.strip() for s in str(row['star']).split(',')]
                a_score = sum([a_vec.get(s, 0) for s in m_stars]) / max(len(m_stars), 1)
                dna += min(a_score, 0.3)
            if pd.notna(row.get('director')):
                dna += min(d_vec.get(str(row['director']).strip(), 0), 0.2)
            if row.get('language') == pref_lang: dna += 0.1
            return min(dna, 1.0)
            
        with st.spinner("Compiling DNA matches..."):
            cv, vectors = nlp_engine
            movies_subset = filtered_df.copy()
            
            if not movies_subset.empty:
                movies_subset['dna_score'] = movies_subset.apply(compute_dna_score, axis=1)
                
                # Supervised Learning: Fit a Ridge Regressor on the fly to predict exactly how much THIS user will like ANY movie
                from sklearn.linear_model import Ridge
                user_model = Ridge(alpha=1.0)
                
                subset_indices = movies_subset.index
                X_train = vectors[subset_indices]
                y_train = movies_subset['dna_score'].values
                
                user_model.fit(X_train, y_train)
                movies_subset['ml_personalized_score'] = user_model.predict(X_train)
                
                movies_subset['v_avg_norm'] = movies_subset['vote_average'] / 10.0
                movies_subset['pop_norm'] = movies_subset['votes'] / max_votes if max_votes > 0 else 0
                
                movies_subset['rec_score'] = (0.7 * movies_subset['ml_personalized_score']) + (0.2 * movies_subset['v_avg_norm']) + (0.1 * movies_subset['pop_norm'])
                movies_subset['hidden_score'] = (0.6 * movies_subset['dna_score']) + (0.3 * movies_subset['v_avg_norm']) - (0.4 * movies_subset['pop_norm'])
                movies_subset['trending_score'] = (0.5 * movies_subset['pop_norm']) + (0.4 * movies_subset['dna_score']) + (0.1 * movies_subset['v_avg_norm'])
                
                # Helper for variety - sample randomly from top candidates!
                # Create a deterministic seed so movies don't shuffle just by clicking "View Details"
                seed_str = f"{selected_lang}_{selected_genre}_{min_rating}_{user_dna.updated_at if user_dna else 0}"
                # In Python, hash() is randomized per process. We can use a stable hash or just stick to hash() 
                # since the Streamlit process usually stays alive for the session.
                import hashlib
                seed_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
                
                def get_dynamic_recs(df, sort_col, n=4, pool_size=20, seed=seed_val):
                    pool_df = df.nlargest(pool_size, sort_col)
                    return pool_df.sample(min(n, len(pool_df)), random_state=seed).sort_values(by=sort_col, ascending=False)
                
                def render_section(title, df, sort_col, color, prefix):
                    c1, c2 = st.columns([5, 1])
                    with c1: st.markdown(f"<p class='category-header'>{title}</p>", unsafe_allow_html=True)
                    with c2: 
                        st.write("")
                        if st.button("Explore More", key=f"exp_{prefix}", use_container_width=True):
                            if st.session_state.explore_section == prefix: st.session_state.explore_section = None
                            else: st.session_state.explore_section = prefix
                            st.rerun()
                            
                    if st.session_state.explore_section == prefix:
                        recs = get_dynamic_recs(df, sort_col, n=20, pool_size=50)
                        st.markdown("<p style='color:#00FF87; font-size:0.9rem;'>Expanded View (20 Results)</p>", unsafe_allow_html=True)
                        for row_idx in range(5):
                            cols = st.columns(4)
                            for col_idx in range(4):
                                idx = row_idx * 4 + col_idx
                                if idx < len(recs):
                                    with cols[col_idx]: 
                                        render_movie_card(recs.iloc[idx], match_score=int(recs.iloc[idx].get(sort_col, 0)*100) if sort_col in ['rec_score'] else None, border_color=color, key_prefix=f"{prefix}_exp_{idx}")
                    else:
                        recs = get_dynamic_recs(df, sort_col, n=4, pool_size=20)
                        cols = st.columns(4)
                        for i, (idx, row) in enumerate(recs.iterrows()):
                            with cols[i]: render_movie_card(row, match_score=int(row.get(sort_col, 0)*100) if sort_col in ['rec_score'] else None, border_color=color, key_prefix=f"{prefix}_")

                render_section("🎯 Recommended For You", movies_subset, 'rec_score', "#FF007A", "rec")
                
                top_2_genres = [g for g, _ in sorted(g_vec.items(), key=lambda x: x[1], reverse=True)[:2]]
                genre_str = " + ".join(top_2_genres) if top_2_genres else "Your Favorites"
                genre_mask = movies_subset['genre'].fillna('').str.contains('|'.join(top_2_genres), case=False)
                if not movies_subset[genre_mask].empty:
                    render_section(f"🎭 Because You Like {genre_str}", movies_subset[genre_mask], 'rec_score', "#00FF87", "gen")
                    
                if not movies_subset[movies_subset['votes'] < movies_subset['votes'].median()].empty:
                    render_section("💎 Out of the Box (Hidden Gems)", movies_subset[movies_subset['votes'] < movies_subset['votes'].median()], 'hidden_score', "#7928CA", "hid")
                    
                render_section("📈 Trending In Your Taste", movies_subset, 'trending_score', "#00C9FF", "tre")
            else: st.warning("Expand your filters to see recommendations.")
    else:
        st.warning("DNA Data missing.")

elif st.session_state.current_tab == "Neural Search":
    # Global Filters at top instead of sidebar
    with st.container(border=True):
        st.markdown("<h4 style='color:#E100FF; margin-top:0;'><i class='bi bi-sliders2'></i> Search Parameters</h4>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1: selected_lang = st.selectbox("🌍 Global Language", sorted(movies['language'].unique()), key="ns_lang")
        all_genres = set()
        for gs in movies['genre'].dropna(): 
            for g in gs.split(','): all_genres.add(g.strip())
        with f2: selected_genre = st.selectbox("🎭 Genre Focus", ["All"] + sorted(list(all_genres)), key="ns_gen")
        with f3: min_rating = st.slider("⭐ Minimum Rating Limit", min_value=0.0, max_value=10.0, value=6.0, step=0.5, key="ns_rate")
        
    filtered_df = movies[(movies['language'] == selected_lang) & (movies['vote_average'] >= min_rating)]
    if selected_genre != "All": filtered_df = filtered_df[filtered_df['genre'].str.contains(selected_genre, na=False)]

    st.markdown("<h2 style='color:#00C9FF; font-weight:900;'><i class='bi bi-search'></i> Advanced Movie Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0aec0; font-size:1.1rem;'>Find exactly what you're looking for, tailored to your DNA.</p>", unsafe_allow_html=True)
    st.write("")
    
    # Establish DNA context for Hybrid Search
    user_dna = db_session.query(UserDNA).filter(UserDNA.user_id == current_user.id).first()
    g_vec = json.loads(user_dna.genre_vector_json) if user_dna and user_dna.genre_vector_json else {}
    a_vec = json.loads(user_dna.actor_vector_json) if user_dna and user_dna.actor_vector_json else {}
    d_vec = json.loads(user_dna.director_vector_json) if user_dna and user_dna.director_vector_json else {}
    
    def compute_dna_for_row(row):
        dna = 0.0
        if pd.notna(row.get('genre')):
            m_genres = [g.strip() for g in row['genre'].split(',')]
            g_score = sum([g_vec.get(g, 0) for g in m_genres]) / max(len(m_genres), 1)
            dna += min(g_score, 0.4)
        if pd.notna(row.get('star')):
            m_stars = [s.strip() for s in str(row['star']).split(',')]
            a_score = sum([a_vec.get(s, 0) for s in m_stars]) / max(len(m_stars), 1)
            dna += min(a_score, 0.3)
        if pd.notna(row.get('director')):
            dna += min(d_vec.get(str(row['director']).strip(), 0), 0.2)
        return min(dna, 1.0)

    tab_vibe, tab_movie = st.tabs(["✨ Semantic Vibe Search", "🎬 Movie Lookalike Search"])
    
    with tab_vibe:
        vibe = st.text_input("What are you in the mood for?", placeholder="e.g., 'A mind-bending sci-fi thriller in space'")
        vibe_btn = st.button('SEARCH BY VIBE', use_container_width=True, key="btn_vibe")
        
        if vibe_btn and vibe:
            cv, vectors = nlp_engine
            v_vec = cv.transform([vibe]).toarray()
            v_sim = cosine_similarity(v_vec, vectors)[0]
            
            results = []
            for idx in filtered_df.index:
                pure_score = v_sim[idx]
                dna_score = compute_dna_for_row(movies.iloc[idx])
                hybrid_score = (0.65 * pure_score) + (0.35 * dna_score)
                results.append((idx, pure_score, hybrid_score))
                
            st.markdown(f"<p class='category-header'>1️⃣ Pure Search Matches: '{vibe}'</p>", unsafe_allow_html=True)
            pure_recs = sorted(results, key=lambda x: x[1], reverse=True)[:4]
            cols_p = st.columns(4)
            for i, (idx, p_sc, _) in enumerate(pure_recs):
                with cols_p[i]: render_movie_card(movies.iloc[idx], match_score=int(p_sc*100), border_color="#00C9FF", key_prefix="v_p_")
                
            st.markdown(f"<p class='category-header'>2️⃣ DNA-Tailored Results (Search + Profile)</p>", unsafe_allow_html=True)
            hybrid_recs = sorted(results, key=lambda x: x[2], reverse=True)[:4]
            cols_h = st.columns(4)
            for i, (idx, _, h_sc) in enumerate(hybrid_recs):
                with cols_h[i]: render_movie_card(movies.iloc[idx], match_score=int(h_sc*100), border_color="#FF007A", key_prefix="v_h_")

    with tab_movie:
        movie_opts = [""] + list(filtered_df['title'].values) if not filtered_df.empty else [""]
        selected_movie = st.selectbox("Select a masterpiece to find similar movies:", movie_opts)
        movie_btn = st.button('FIND LOOKALIKES', use_container_width=True, key="btn_movie")
        
        if movie_btn and selected_movie:
            m_idx = movies[movies['title'] == selected_movie].index[0]
            target_cluster = movies.iloc[m_idx]['ml_cluster']
            
            # Unsupervised Learning: Filter lookalikes by K-Means Cluster explicitly
            cluster_df = filtered_df[filtered_df['ml_cluster'] == target_cluster]
            
            results = []
            for idx in cluster_df.index:
                if idx == m_idx: continue
                pure_score = similarity[m_idx][idx]
                results.append((idx, pure_score))
                
            st.markdown(f"<p class='category-header'>1️⃣ ML Clustering Lookalikes (Cluster #{target_cluster})</p>", unsafe_allow_html=True)
            pure_recs = sorted(results, key=lambda x: x[1], reverse=True)[:8]
            
            for row_idx in range(2):
                cols_pm = st.columns(4)
                for col_idx in range(4):
                    r_idx = row_idx * 4 + col_idx
                    if r_idx < len(pure_recs):
                        idx, p_sc = pure_recs[r_idx]
                        with cols_pm[col_idx]: render_movie_card(movies.iloc[idx], match_score=int(p_sc*100), border_color="#00FF87", key_prefix=f"m_p_{r_idx}")

elif st.session_state.current_tab == "Director's Lens":
    st.markdown("<p class='category-header'><i class='bi bi-camera-reels'></i> Director's Lens</p>", unsafe_allow_html=True)
    st.write("Explore movies crafted by your favorite visionary directors.")
    
    all_directors = set()
    for d in movies['director'].dropna():
        all_directors.add(str(d).strip())
    sorted_directors = sorted(list(all_directors))
    
    selected_director = st.selectbox("Select a Director", [""] + sorted_directors)
    
    if selected_director:
        dir_movies = movies[movies['director'].astype(str).str.contains(selected_director, regex=False, na=False)].sort_values(by='vote_average', ascending=False)
        st.markdown(f"#### Exploring {len(dir_movies)} masterpieces by {selected_director}")
        
        # Supervised Learning: Random Forest Classification
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        
        # Predict if movie is a Masterpiece (vote_average >= 7.5) based on its features
        cv, vectors = nlp_engine
        X_dir = vectors[dir_movies.index]
        y_dir = (dir_movies['vote_average'] >= 7.5).astype(int)
        
        if len(set(y_dir)) > 1: # Only train if both classes exist
            clf.fit(X_dir, y_dir)
            dir_movies['is_masterpiece_prob'] = clf.predict_proba(X_dir)[:, 1]
        else:
            dir_movies['is_masterpiece_prob'] = y_dir.values
        
        dir_movies = dir_movies.sort_values(by='is_masterpiece_prob', ascending=False)
        
        # Display in grid
        cols = st.columns(4)
        for i, (idx, row) in enumerate(dir_movies.iterrows()):
            with cols[i % 4]:
                render_movie_card(row, match_score=int(row.get('is_masterpiece_prob', 0)*100), border_color="#FF007A", key_prefix="dir_")

elif st.session_state.current_tab == "Cast Universe":
    st.markdown("<p class='category-header'><i class='bi bi-stars'></i> Cast Universe</p>", unsafe_allow_html=True)
    st.write("Find movies featuring your favorite stars tailored to your preferences.")
    
    all_stars = set()
    for s in movies['star'].dropna():
        for actor in str(s).split(','):
            if actor.strip(): all_stars.add(actor.strip())
    sorted_stars = sorted(list(all_stars))
    
    selected_star = st.selectbox("Search for an Actor", [""] + sorted_stars)
    
    if selected_star:
        star_movies = movies[movies['star'].astype(str).str.contains(selected_star, regex=False, na=False)].sort_values(by='vote_average', ascending=False)
        st.markdown(f"#### Exploring {len(star_movies)} movies featuring {selected_star}")
        
        # Supervised Learning: Logistic Regression
        from sklearn.linear_model import LogisticRegression
        user_dna = db_session.query(UserDNA).filter(UserDNA.user_id == current_user.id).first()
        g_vec = json.loads(user_dna.genre_vector_json) if user_dna and user_dna.genre_vector_json else {}
        a_vec = json.loads(user_dna.actor_vector_json) if user_dna and user_dna.actor_vector_json else {}
        
        def compute_quick_dna(row):
            dna = 0.0
            if pd.notna(row.get('genre')):
                m_g = [g.strip() for g in row['genre'].split(',')]
                dna += min(sum([g_vec.get(g, 0) for g in m_g]) / max(len(m_g), 1), 0.5)
            if pd.notna(row.get('star')):
                m_s = [s.strip() for s in str(row['star']).split(',')]
                dna += min(sum([a_vec.get(s, 0) for s in m_s]) / max(len(m_s), 1), 0.5)
            return min(dna, 1.0)
            
        star_movies['dna_score'] = star_movies.apply(compute_quick_dna, axis=1)
        
        log_reg = LogisticRegression()
        cv, vectors = nlp_engine
        X_star = vectors[star_movies.index]
        y_star = (star_movies['dna_score'] > 0.25).astype(int) # 1 if strong match
        
        if len(set(y_star)) > 1:
            log_reg.fit(X_star, y_star)
            star_movies['like_prob'] = log_reg.predict_proba(X_star)[:, 1]
        else:
            star_movies['like_prob'] = y_star.values
            
        star_movies = star_movies.sort_values(by='like_prob', ascending=False)
            
        # Display in grid
        cols = st.columns(4)
        for i, (idx, row) in enumerate(star_movies.iterrows()):
            with cols[i % 4]:
                render_movie_card(row, match_score=int(row.get('like_prob', 0)*100), border_color="#00C9FF", key_prefix="cast_")

elif st.session_state.current_tab == "Watchlist":
    st.markdown("<p class='category-header'>📋 Your Cinematic Watchlist</p>", unsafe_allow_html=True)
    st.write("Movies you've queued up for later.")
    
    watchlist_items = db_session.query(Watchlist).filter_by(user_id=current_user.id).order_by(Watchlist.added_at.desc()).all()
    
    if not watchlist_items:
        st.info("Your watchlist is currently empty. Go to Discover to add some movies!")
    else:
        cols = st.columns(4)
        for i, item in enumerate(watchlist_items):
            # Find the movie row from the dataframe
            matching_rows = movies[movies['movie_id'] == item.movie_id]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                with cols[i % 4]:
                    render_movie_card(row, border_color="#00FF87", key_prefix="wl_")

# Footer
st.divider()
st.markdown("<p style='text-align:center; color:#555; font-size:0.85rem;'>Neural Entertainment System • Powered by Machine Learning • Engineered by Shlok Chorghe</p>", unsafe_allow_html=True)
