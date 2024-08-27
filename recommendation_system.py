import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("movies_data/movies.csv",encoding="latin-1",sep='\t',usecols=["title","genres"])

movies["genres"] = movies["genres"].apply(lambda s:s.replace("|", " ").replace("-",""))
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies["genres"])
# print(vectorizer.vocabulary_)
# {'animation': 2, 'children': 3, 'comedy': 4, 'adventure': 
# 1, 'fantasy': 8, 'romance': 15, 'drama': 7, 'action': 0, 'crime': 5, 'thriller': 17, 'horror': 11, 'sci': 16, 'fi': 
# 9, 'documentary': 6, 'war': 18, 'musical': 12, 'mystery': 
# 13, 'film': 10, 'noir': 14, 'western': 19}

# {'animation': 2, 'children': 3, 'comedy': 4, 'adventure': 
# 1, 'fantasy': 8, 'romance': 13, 'drama': 7, 'action': 0, 'crime':
#  5, 'thriller': 15, 'horror': 10, 'scifi': 14, 'documentary': 6, 'war':
#    16, 'musical': 11, 'mystery': 12, 'filmnoir': 9, 'western': 17}

cosine_sm = cosine_similarity(tfidf_matrix)
cosine_similarity_df = pd.DataFrame(cosine_sm, index=movies["title"],columns=movies["title"])

top_k = 20
user_input = "Wings of Courage (1995)"
data = cosine_similarity_df.loc[user_input, :]
result = data.sort_values(ascending=False)[:top_k]


print(result)

