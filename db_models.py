import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Float
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import bcrypt

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    age = Column(Integer, nullable=True)
    country = Column(String(100), nullable=True)
    preferred_language = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    watch_history = relationship('WatchHistory', backref='user')
    preferences = relationship('UserPreferences', backref='user', uselist=False)
    dna = relationship('UserDNA', backref='user', uselist=False)
    ratings = relationship('MovieRating', backref='user')
    watchlist = relationship('Watchlist', backref='user')

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class WatchHistory(Base):
    __tablename__ = 'watch_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    movie_id = Column(String(100), nullable=False)
    movie_title = Column(String(250), nullable=False)
    watched_at = Column(DateTime, default=datetime.utcnow)
    watch_count = Column(Integer, default=1)

class MovieRating(Base):
    __tablename__ = 'movie_ratings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    movie_id = Column(String(100), nullable=False)
    movie_title = Column(String(250), nullable=False)
    rating = Column(Float, nullable=False)  # 1.0 to 5.0
    rated_at = Column(DateTime, default=datetime.utcnow)

class UserPreferences(Base):
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    favorite_genres = Column(Text, nullable=True)
    favorite_languages = Column(Text, nullable=True)
    mood_preferences = Column(Text, nullable=True)
    runtime_preference = Column(Integer, nullable=True)

class UserDNA(Base):
    __tablename__ = 'user_dna'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    genre_vector_json = Column(Text, nullable=True)
    actor_vector_json = Column(Text, nullable=True)
    director_vector_json = Column(Text, nullable=True)
    mood_scores_json = Column(Text, nullable=True)
    genre_snapshot_json = Column(Text, nullable=True)  # Previous snapshot for evolution delta tracking
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Watchlist(Base):
    __tablename__ = 'watchlist'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    movie_id = Column(String(100), nullable=False)
    movie_title = Column(String(250), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

# Database Setup
engine = create_engine('sqlite:///cinematch.db', echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
