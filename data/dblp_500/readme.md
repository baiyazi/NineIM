数据集下载页面：http://snap.stanford.edu/data/com-DBLP.html

The DBLP computer science bibliography provides a comprehensive list of research papers in computer science. 
We construct a co-authorship network where two authors are connected if they publish at least one paper together. 
Publication venue, e.g, journal or conference, defines an individual ground-truth community; 
authors who published to a certain journal or conference form a community.

We regard each connected component in a group as a separate ground-truth community. 
We remove the ground-truth communities which have less than 3 nodes. 
We also provide the top 5,000 communities with highest quality which are described in our paper. 
As for the network, we provide the largest connected component.

Dataset statistics
Nodes	317080
Edges	1049866
Nodes in largest WCC	317080 (1.000)
Edges in largest WCC	1049866 (1.000)
Nodes in largest SCC	317080 (1.000)
Edges in largest SCC	1049866 (1.000)
Average clustering coefficient	0.6324
Number of triangles	2224385
Fraction of closed triangles	0.1283
Diameter (longest shortest path)	21
90-percentile effective diameter	8


Source (citation)
J. Yang and J. Leskovec. Defining and Evaluating Network Communities based on Ground-truth. ICDM, 2012.

数据集下载地址：
Files
File	                                          Description
com-dblp.ungraph.txt.gz	          Undirected DBLP co-authorship network          link: http://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
com-dblp.all.cmty.txt.gz	DBLP communities            link: http://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz
com-dblp.top5000.cmty.txt.gz	DBLP communities (Top 5,000)            link: http://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz


