# Movie Recommender System

A movie recommendation system built with Python that implements multiple recommendation strategies using the [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

## Recommendation Approaches

1. **Top-N by Weighted Rating** — Ranks movies using the IMDB weighted rating formula (Bayesian average), with support for genre-specific top lists.
2. **Content-Based Filtering (TF-IDF)** — Recommends similar movies based on plot overview and tagline similarity using TF-IDF vectorization and cosine similarity.
3. **Improved Content-Based Filtering** — Enhances recommendations by incorporating cast, crew (director), keywords, and genres via a CountVectorizer "metadata soup".
4. **Collaborative Filtering (SVD)** — Predicts user ratings using matrix factorization (SVD) from the Surprise library trained on user rating data.
5. **Hybrid Recommender** — Combines content-based similarity with collaborative filtering to produce personalized recommendations per user.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the dataset from [Kaggle: The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and place the following CSV files in a `data/` directory:

- `movies_metadata.csv`
- `links_small.csv`
- `links.csv`
- `credits.csv`
- `keywords.csv`
- `ratings_small.csv`

## Usage

Open and run the notebook:

```bash
jupyter notebook movie_recommendation_system.ipynb
```
