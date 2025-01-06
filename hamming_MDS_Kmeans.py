from Bio import Seq, SeqIO
import scipy
import numpy as np
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances as pdist
from matplotlib.colors import ListedColormap
import scipy.spatial.distance as distance
dist= []
np.array(dist)

s=[i for i in SeqIO.parse(f"HW2.fas",'fasta')]
    #seq_record.id is the name of the sequence
    #seq_record.seq is the sequence itself
#print(s.seq)
def hamming_distance(seq1, seq2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq1, seq2))

for i in range(119):
    #dist.append(scipy.spatial.distance.hamming(np.array(seq_record.seq[i]), np.array(seq_record.seq[i+1])))
    dist.append(hamming_distance(s[i].seq, s[i+1].seq))
print(dist)
a=np.array(dist)
def pairwise_dist(seqs,nseqs):
    h_dist = np.zeros((nseqs, nseqs))
    for i in range(nseqs):
        for j in range(nseqs):
            if i != j:
                temp = hamming_distance(seqs[i],seqs[j])
                h_dist[i][j] = temp
               # h_dist.append(hamming_distance(seqs[i],seqs[j]))
    return h_dist

#b=a.reshape(-1, 1)
#a1=pdist(dist, Y=None, metric='hamming', n_jobs=None, force_all_finite=True)

'''
a1=[[],[]]
for j in range(119):
    for(n) in range(119):
        a1.append(hamming_distance(s[j].seq, s[n].seq))
'''
#print(a1.shape)
dist1 = []
for seq_record in SeqIO.parse("HW2.fas", "fasta"):
    dist1.append(seq_record.seq)
fv = pairwise_dist(dist1, 120)
mds = MDS(random_state=0,n_components=2,dissimilarity='precomputed')
dist_transform = mds.fit_transform(fv)
dist_transform_df=pd.DataFrame(dist_transform)
#print(dist_transform)
plt.scatter(dist_transform_df[0],dist_transform_df[1],s=10)
plt.show()
############################################################################################################
#from visual inspection k=5
k=5

def kmeans(dist_transform_df, k=5,max_iter=100):
    if isinstance(dist_transform_df, pd.DataFrame): dist_transform_df = dist_transform_df.values
    idx=np.random.choice(len(dist_transform_df),k,replace=False)
    centroids=dist_transform_df[idx,:]
    R =np.argmin(distance.cdist(dist_transform_df,centroids,'euclidean'),axis=1)
    for i in range(max_iter):
        centroids = np.vstack([dist_transform_df[R==i,:].mean(axis=0) for i in range(k)])
        temp=np.argmin(distance.cdist(dist_transform_df,centroids,'euclidean'),axis=1)
        if np.array_equal(R,temp): break
        R=temp
        return R,centroids    

R,centroids=kmeans(dist_transform_df,k)
centers=np.array(centroids)
plt.scatter(dist_transform_df[0],dist_transform_df[1],c=R,s=10)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
plt.show()