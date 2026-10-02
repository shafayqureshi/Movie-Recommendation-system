# Movie Recommendation System

A Python-based movie recommendation system developed using collaborative filtering and the MovieLens dataset. The application provides both **similar-movie recommendations** and **mood/genre-based recommendations** through a graphical user interface.

## Features

- Item-to-item collaborative filtering
- Cosine similarity for finding similar movies
- Mood and genre-based movie recommendations
- Bayesian rating-based ranking
- Movie search functionality
- Interactive Tkinter graphical user interface
- Model caching for faster subsequent runs
- Memory-efficient sampling for the MovieLens 32M dataset

## Recommendation Methods

### 1. Similar Movie Recommendation

The system uses **item-item collaborative filtering**. A user selects a movie, and the system calculates similarity between movies using cosine similarity on the user-item rating matrix.

Example:

```text
Selected Movie: Toy Story
→ Recommended similar movies based on user rating patterns
```

### 2. Mood-Based Recommendation

Users can select a mood or preferred category such as:

- Action
- Comedy
- Romantic
- Drama
- Horror
- Sci-Fi
- Thriller
- Animation
- Fantasy
- Crime
- Family
- Adventure

Movies matching the corresponding genres are ranked using rating statistics and a Bayesian weighted score.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Tkinter
- Collaborative Filtering
- Cosine Similarity

## Dataset

This project uses the **MovieLens 32M dataset**.

The main files required by the program are:

```text
movies.csv
ratings.csv
```

Due to the large size of the dataset, these files are not included in this GitHub repository.

Download the MovieLens dataset from:

https://grouplens.org/datasets/movielens/

After downloading, place `movies.csv` and `ratings.csv` in the same folder as `movie_recommender.py`.

## Project Structure

```text
Movie-Recommendation-System/
│
├── movie_recommender.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── results/
    ├── main_interface.png
    ├── mood_recommendation.png
    └── similar_movies.png
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Movie-Recommendation-System
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Make sure `movies.csv` and `ratings.csv` are in the project directory.

Run:

```bash
python movie_recommender.py
```

The graphical interface will open.

Click **Load/Train Model** to build the recommendation model.

The first run may take some time because the model needs to process the rating data. The trained model is cached locally so that subsequent runs can load it more quickly.


## How It Works

The recommendation system:

1. Loads movie and user-rating data.
2. Selects active users and frequently rated movies for efficient processing.
3. Builds a user-item rating matrix.
4. Mean-centers the ratings and handles missing values.
5. Calculates item-item cosine similarity.
6. Generates similar-movie recommendations based on rating patterns.
7. Provides mood-based recommendations using genres and rating statistics.

## Author

Developed as an Information Retrieval and Data Mining project.