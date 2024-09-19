# OC-P6

# NLP - Topic Modeling - Identifying dissatisfaction topics in negative feedback
# Cumputer Vision - Feasibility study for automatic photo classification

## Project scenario
Avis Restau, a fictitious company connecting restaurants and customers, is launching a new social feature to connect food lovers and help them better understand their dining experiences.

- Commenting functionality
- Photo-sharing capability

Potential applications:
- Data mining and analysis
- User behavior modeling

Here, the objective is to study the feasibility of :
- Detecting dissatisfaction themes in negative feedback
- Automatically tagging photos posted on the platform (For the preliminary study, simply perform a clustering analysis)
- Using the *Yelp* dataset

## Solutions :
### Topic modeling
- comments selection
- cleaning :
  - lower case
  - remove escape sequences
  - remove url
  - remove repeated characters
  - stopwords, figures, Part-of-Speech filtering,
  - tokenization
  - dictionary
  - TF-IDF
- Modeling :
  - Latent Dirichlet Allocation
  - Optimisations
### Computer Vision
- photos selection
- preprocessing
- features extraction (using SIFT)
- visual words dictionary
- dimensionality reduction
- Kmeans clustering
- Evaluation using ARI (clusters VS real categories)
### Using Yelp API to collect new data
- developer account on Yelp platform
- use Yelp API using the query language *GraphQL*
