import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans

plt.style.use('ggplot')

def main():
    c_1 = np.array([105, 96, 71])/255
    c_2 = np.array([127, 198, 164])/255

    data = np.loadtxt("MLWeedCropTrain.csv",delimiter=',')
    P_matrix = np.loadtxt("projection_matrix.csv",delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    X_projected = X.dot(P_matrix.T)

    plt.figure(figsize=(12,12))
    plt.scatter(X_projected[:,0], X_projected[:,1], c=[c_1 if i == 0 else c_2 for i in y])

    patch1 = mpatches.Patch(color=c_1, label='weed')
    patch2 = mpatches.Patch(color=c_2, label='crop')
    plt.legend(handles=[patch1, patch2])

    plt.axis("equal")
    plt.title("Projection of dataset on the PCs")
    plt.gcf().savefig("ex1.png", bbox_inches='tight')

    starting_point = np.vstack(
        (
            (X[0, ]),
            (X[1, ])
        )
    )
    kmeans = KMeans(n_clusters=2, algorithm='full', n_init=1,
                    init=starting_point).fit(X)

    centers = kmeans.cluster_centers_
    projected_centers = centers.dot(P_matrix.T)

    # plt.clf()
    plt.figure(figsize=(12,12))
    plt.scatter(X_projected[:,0], X_projected[:,1], c=[c_1 if i == 0 else c_2 for i in y])
    plt.scatter(projected_centers[:,0], projected_centers[:,1], color='red')
    
    patch1 = mpatches.Patch(color=c_1, label='weed')
    patch2 = mpatches.Patch(color=c_2, label='crop')
    patch3 = mpatches.Patch(color='red', label='cluster centers')
    plt.legend(handles=[
        patch1, 
        patch2, 
        patch3
    ])
    plt.axis("equal")
    plt.title("Projection of dataset on the PCs, with KMeans cluster centers")
    plt.gcf().savefig("ex2.png", bbox_inches='tight')

    

if __name__ == "__main__":
    main()