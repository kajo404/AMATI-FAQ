# Adapted from http://brandonrose.org/clustering

import json
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
import mpld3

from argparse import ArgumentParser

from sklearn import feature_extraction
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score

from nltk.stem.snowball import SnowballStemmer

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="input dataset in json format", metavar="FILE")
parser.add_argument("-c", "--clusters",
                    dest="num_clusters", help="number of clusters")

args = parser.parse_args()

with open(args.filename, "r") as read_file:
    data = json.load(read_file)

documents = []
slideSets = []
for elem in data:
    if elem["content"]:
        documents.append(elem["content"])
    if elem["slideSet"]:
        slideSets.append(elem["slideSet"])
    else:
        slideSets.append(0)

# stop words, stemming and tokenizing

nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in documents:
    # for each item in 'synopses', tokenize/stem
    allwords_stemmed = tokenize_and_stem(i)
    # extend the 'totalvocab_stemmed' list
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame(
    {'words': totalvocab_tokenized}, index=totalvocab_stemmed)
print('Built vocabulary of ' + str(vocab_frame.shape[0]) + ' items')


# tfidf
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                   min_df=0.03, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

#
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print('Shape of TF-IDF matrix: ', tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()


dist = 1 - cosine_similarity(tfidf_matrix)


# Determine optimal number of clusters using WSS and elbow method

distortions = []
K = range(5, 15)
tfidf_array = tfidf_matrix.todense()
for k in K:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(tfidf_matrix)
    distortions.append(sum(np.min(cdist(
        tfidf_array, km.cluster_centers_, 'euclidean'), axis=1)) / tfidf_array.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Silhouette Analysis


for k in K:
    # Create two column plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-1, 1])
    # additional blank space for the plot
    ax1.set_ylim([0, len(tfidf_array) + (k + 1) * 10])

    km = KMeans(n_clusters=k, random_state=1)
    cluster_labels = km.fit_predict(tfidf_array)

    silhouette_avg = silhouette_score(tfidf_array, cluster_labels)
    print("For n_clusters =", k,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(tfidf_array, cluster_labels)

    y_lower = 10
    for i in range(k):
        # Aggregate scores for samples belonging to cluster i
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.8, -0.6-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Transform into 2D representation for plotting

    MDS()

    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
    ax2.scatter(pos[:, 0], pos[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # This currently doesn't work, needs to transform cluster centers to 2D
    # Labeling the clusters
    # centers = km.cluster_centers_
    # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #            c="white", alpha=1, s=200, edgecolor='k')

    # for i, c in enumerate(centers):
    #    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % k),
                 fontsize=14, fontweight='bold')

plt.show()

# Run Clustering for given k and generate plot
km = KMeans(n_clusters=int(args.num_clusters), random_state=1)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

questions = {'content': documents, 'cluster': clusters, 'slideSet': slideSets}

frame = pd.DataFrame(questions, index=[clusters], columns=[
                     'content', 'cluster', 'slideSet'])

print(frame['cluster'].value_counts())


print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(int(args.num_clusters)):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :5]:  # 5 is the number of representative words
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[
              0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()

    #print("Cluster %d slideSets:" % i, end='')
    # for title in frame.ix[i]['slideSet'].values.tolist():
    #    print(' %i,' % title, end='')
    # print(set(frame.ix[i]['slideSet'].values.tolist()))
    # print(frame.ix[i]['content'].values.tolist())
    print()  # add whitespace
    print()  # add whitespace

print()
print()


MDS()

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

# set up colors per clusters using a dict
cluster_colors = {0: '#800000',
                  1: '#808000',
                  2: '#008080',
                  3: '#000075',
                  4: '#f58231',
                  5: '#46f0f0',
                  6: '#4363d8',
                  7: '#3cb44b',
                  8: '#f032e6',
                  9: '#9a6324',
                  10: '#000000',
                  11: '#e6194b',
                  12: '#ffe119',
                  13: '#911eb4',
                  14: '#808080',
                  15: '#aaffc3'}

cluster_names = {0: '0',
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9',
                    10: '10',
                    11: '11',
                    12: '12',
                    13: '13',
                    14: '14',
                    15: '15'}

# create data frame that has the result of the MDS plus the cluster numbers and questions
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=slideSets))
# group by cluster
groups = df.groupby('label')
# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(
        axis='y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.savefig('clusters.png', dpi=300)
