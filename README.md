# 🧬 Movie DNA: Neural Entertainment Dashboard

**Movie DNA** is a comprehensive Machine Learning project that implements a sophisticated, multi-paradigm recommendation system. This project was developed as a full-stack ML application to demonstrate the integration of NLP, Supervised Learning, Unsupervised Learning, and Reinforcement Learning in a single ecosystem.

---

## 📈 Project Implementation Workflow (Step-by-Step)

This project follows a rigorous Data Science lifecycle, from raw data processing to real-time model deployment.

### Step 1: Data Pre-processing & Cleaning
Before any modeling, the raw TMDB dataset underwent significant transformation:
*   **Feature Extraction:** We extracted key metadata including `genres`, `keywords`, `cast` (Top 3 actors), and `crew` (Director).
*   **Data Cleaning:** Handled missing values and converted string-based JSON fields into usable Python lists.
*   **Text Standardizing:** Removed spaces between words in names (e.g., "Johnny Depp" becomes "JohnnyDepp") to ensure the vectorizer treats them as unique tokens.

### Step 2: Feature Engineering (The "Tags" System)
To create a rich semantic profile for every movie, we engineered a consolidated `tags` column. This column combines:
`Movie Overview + Genres + Keywords + Top Cast + Director + Language`
This results in a single string of text that encapsulates the entire "DNA" of the film.

### Step 3: NLP Vectorization
We converted the textual "Tags" into mathematical format using **Bag-of-Words (BoW)**:
*   **CountVectorizer:** Used to extract the top 5,000 most frequent words across the entire dataset.
*   **Stop-word Removal:** Filtered out common English words (the, is, in, etc.) that do not add semantic value.
*   **Vector Space:** Every movie is now represented as a 5,000-dimensional vector in a sparse matrix.

### Step 4: Machine Learning Model Implementation

#### 1. Unsupervised Learning (Clustering)
*   **Model:** K-Means Clustering (k=20).
*   **Purpose:** To segment the database into 20 thematic "Galaxies" (Clusters). This allows the system to perform high-speed lookalike searches within mathematically similar categories.

#### 2. Supervised Learning (Regression)
*   **Model:** Ridge Regression.
*   **Purpose:** Used in the **Discover Feed**. The system trains a regressor on the fly to predict a user’s "Match Score" for any movie by mapping movie vectors to the user's personal DNA vector.

#### 3. Supervised Learning (Classification)
*   **Models:** Random Forest & Logistic Regression.
*   **Purpose:** Used in the **Director** and **Cast** modules to classify movies as "Masterpieces" or "User Likes" by calculating the probability of class membership using trained decision boundaries.

#### 4. Reinforcement Learning (Real-time Evolution)
*   **Model:** Q-Learning Approximation.
*   **Purpose:** Every time a user rates a movie, a Reward/Penalty signal is issued. The system updates the weights of the User DNA vector (the agent's policy), ensuring the recommendation engine evolves dynamically without requiring a full system retrain.

---

## 🚀 App Modules & Interactivity

### 🧬 Neural Discovery (Home Tab)
The heart of the application. It provides a real-time, personalized ranked list of movies.
*   **How it Works:** The app fetches your unique User DNA vector and runs a **Ridge Regression** model against the movie database to predict a "Personalized Score."
*   **DNA Radar:** Displays a real-time Plotly Radar Chart showing your current taste across all genres.

### 🔍 Neural Search
An advanced dual-mode search engine:
*   **Semantic Vibe Search:** Uses NLP to match your natural language query (e.g., "Dark sci-fi space thriller") with movie tags.
*   **Movie Lookalikes:** Select a movie you love, and the engine finds its closest neighbors within its **ML-calculated Galaxy (Cluster)**.

### 🎬 Director's Lens & Cast Universe
*   **Predictive Analytics:** These tabs allow you to explore movies by specific creators.
*   **Masterpiece Prediction:** In the Director tab, a **Random Forest Classifier** predicts the probability of a film being a masterpiece.
*   **Affinity Prediction:** In the Cast tab, a **Logistic Regression** model predicts the exact probability of *you* liking that movie based on your specific DNA.

### 📋 Interactive Features (The "View Details" Dialog)
Clicking on any movie card opens a high-tech dialog window with the following actions:
*   **Add to Watchlist:** Queues the movie in your persistent "Watchlist" tab for later viewing.
*   **Mark as Watched:** Signals the environment that you have consumed the content. 
    *   *DNA Impact:* Triggers an implicit update to your DNA vector, slightly increasing the weights of the movie's genres and actors.
*   **Rate This Movie (The Feedback Loop):**
    *   **The RL Model:** This is where the **Reinforcement Learning** happens. 
    *   **5 Stars:** Positive Reinforcement (+0.25 Reward).
    *   **1-2 Stars:** Negative Reinforcement (-0.15 Penalty).
    *   *Immediate Update:* Your DNA vector is instantly recalculated and saved, and all dashboard recommendations are updated to reflect your new taste.

---

## ⚙️ Setup & Installation

### 1. Environment Preparation
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Initialization
If you are running this for the first time, run the data preparation script to generate the NLP artifacts:
```bash
python prepare_data.py
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🛠️ Technical Stack
*   **Language:** Python
*   **Framework:** Streamlit
*   **ML Libraries:** Scikit-learn, Pandas, NumPy
*   **Visualization:** Plotly, Matplotlib
*   **Database:** SQLAlchemy (SQLite)

---

**Project Developed by:** Shlok Chorghe  
**Submission Category:** Machine Learning Course Project
