####################################################################
# This TD is using glove to explore :
#  - Word embeddings
#  - euclidian distance / cosine similarity
#  - Similarity matrix
#  - Hierarchical clustering
#  - Multi dimensional scaling
#  - PCA
#  - k-means


################# ################# #################
# Import packages
################# ################# #################
import numpy as np
import torchtext
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pdb as debugger

pdb = debugger.set_trace

################# ################# #################
# Download glove
################# ################# #################
# can represent words as a vector of 100 values
glove = torchtext.vocab.GloVe(name="6B", dim=100)
# semantically related words have similar glove representations


################# ################# #################
# Word representation
################# ################# #################
# Q1 Look at the representation of the word 'lamp', 'candle'
A = glove['lamp']
B = glove['candle']

# Q2 measure the euclidian dissimilarity between the two words
euclidian = np.sqrt(
    np.sum([(A[i]-B[i])**2 for i in range(np.shape(A)[0])]))  # 4.2

# Q3 measure the cosine similarity between the two words
cosine = np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))  # 0.67

# Q4 Redo the same for 'mouse' and 'lamp'
A = glove['mouse']
B = glove['lamp']
euclidian = np.sqrt(
    np.sum([(A[i]-B[i])**2 for i in range(np.shape(A)[0])]))  # 6.56
cosine = np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))  # 0.26

# Q5 What can you conclude ?
# => Since lamp and candle have similar meanings compared to mouse and lamp, the distance between them is smaller

# Q6 using cosine similarity, find the 10 closest words to the word lamp
# A = glove['lamp']
# allwords = np.array(glove.itos)  # 400.000 words
# Similarity = []
# for word in allwords:
#     B = glove[word]
#     cosine = np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))
#     Similarity.append(cosine)

# sorted_similarity = sorted(Similarity)

# indexes = [Similarity.index(e) for e in sorted_similarity[-10:]]
# ['illuminated', 'fluorescent', 'kerosene', 'incandescent', 'halogen', 'lights', 'candle', 'bulb', 'lamps', 'lamp']

# Q7 Write a function that takes as arg the embedding of a word and N and return the N closest words.


def returnclosest(word, n):
    allwords = np.array(glove.itos)  # 400.000 words
    Similarity = []
    for w in allwords:
        B = glove[w]
        cosine = np.dot(word, B)/(np.linalg.norm(word)*np.linalg.norm(B))
        Similarity.append(cosine)

    sorted_similarity = sorted(Similarity)
    indexes = [Similarity.index(e) for e in sorted_similarity[-n-1:-1]]
    return [allwords[i] for i in indexes]


# Q8 Use this function to return the 10 closest words to 'lamp', 'rabbit' & 'bottle'
# close = returnclosest(glove['lamp'], n=10)
# close = returnclosest(glove['rabbit'], n=10)
# close = returnclosest(glove['bottle'], n=10)

# Q9 Now that words are vectors, we can move into that space. For example find the embedding of 'queen' - 'woman' + 'man'
emb = glove['queen'] - glove['woman'] + glove['man']

# Q10 Find the closest word to this new embeding
# close = returnclosest(emb, n=1)

# Q11 Check that the other relation is true = 'king' - 'man' + 'woman' -> 'queen'
emb = glove['king'] - glove['man'] + glove['woman']
# close = returnclosest(emb, n=1)


# Q12 Now let's investigating bias in glove embeddings : look at the 3 closest words to : doctor - man + woman
emb = glove['doctor'] - glove['man'] + glove['woman']
# close = returnclosest(emb, n=3)

# Q13 and do in the other direction : doctor - woman + man
emb = glove['doctor'] - glove['woman'] + glove['man']
# close = returnclosest(emb, n=3)

# -> Embeddings are just reflecting the bias that can be found in the human made text used for training.


################# ################# #################
# Dimensionality reduction
################# ################# #################
# Q21 Now let's consider the 20 following words = ['rabbit', 'mouse', 'horse', 'bear', 'monkey','whale', 'dolphin', 'tuna', 'swordfish', 'wolf', 'sofa', 'table', 'chair', 'rug', 'lamp','bag','computer', 'phone', 'keyboard', 'screen']
# Compute the similarity matrix (using cosine similarity) between all those words

words = ['rabbit', 'mouse', 'horse', 'bear', 'monkey', 'whale', 'dolphin', 'tuna', 'swordfish',
         'wolf', 'sofa', 'table', 'chair', 'rug', 'lamp', 'bag', 'computer', 'phone', 'keyboard', 'screen']
sim = [[np.dot(glove[a], glove[b])/(np.linalg.norm(
    glove[a])*np.linalg.norm(glove[b])) for a in words] for b in words]

# Q22 plot this similarity matrix, do you notice something ?
# plt.imshow(sim)
# plt.colorbar()
# plt.show()
# => it a symmetric matrix that for which diagonal values are equal to 1

# Q23 using a function from seaborn, plot the similarity matrix and its hierarchical clustering
sim = pd.DataFrame(sim)  # Convert to pandas dataframe
sim = sim.rename(mapper=pd.Series(words), axis=1)  # Add word names as labels
ax = sns.heatmap(sim)
plt.show()

# Q24 What can you say about this clustering ? Look carefully at the word 'mouse' what do you notice ?


# Q24 perform multi-dimensional scaling on this similarity matrix
mds = MDS(random_state=0)  # MDS is implemented sklearn.manifold
scaled_df = XXXXXXXXX

# Q25 plot the results of MDS by showing the position of each word in this space
for i in range(np.shape(words)[0]):
    XXXXXXXXX
    plt.annotate(words[i], (XXXXXXXXX))


# Q26 perform PCA with one dimension on the matrix, and plot the result
pca = XXXXXXXXX
fig = plt.figure()
ax = fig.add_subplot(111)
X_reduced = XXXXXXXXX
for i in range(np.shape(words)[0]):
    ax.scatter(XXXXXXXXX)
    ax.text(XXXXXXXXX, rotation=35)


# Q27 redo the same with a PCA with 2 dimensions
XXXXXXXXX

# Q28 and now with 3 dimensions
XXXXXXXXX


# Q29 Now compute the similarity matrix for the 800 first words of glove.
words = glove.itos[0:800]
XXXXXXXXX

# Q30 And plot the hierarchical clustering
XXXXXXXXX


# Q31 plot the 2D dimension of this matrix
XXXXXXXXX

# Q32 what can you notice ?

# Q33 using KMeans function from sklearn.cluster cluster the similarity matrix in 6 clusters
kmeans = XXXXXXXXX
kmeans.labels_

# Q34 recompute the 2D PCA and plot it with color corresponding to the cluster
pca = XXXXXXXXX
X_reduced = XXXXXXXXX
colors = sns.color_palette("husl", 6)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(np.shape(words)[0]):
    ax.scatter(XXXXXXXXX)
    ax.text(XXXXXXXXX)
