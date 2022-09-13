import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

path = "../Assignment2/pca.txt"


def pca_apply(data, n_components):
    pca = PCA(n_components=n_components)
    new_data = pca.fit_transform(data)
    return new_data, pca


if __name__ == "__main__":
    data = pd.read_csv(path, delimiter=" ", header=None,
                       index_col=0,
                       names=['distance', 'diameter', 'density', 'output'])
    print(data)
    norm_data = data.iloc[:, :-1]
    norm_data = (norm_data - norm_data.mean()) / norm_data.std();
    print(norm_data)

    data_values = norm_data.values
    print(data_values)
    new_data, pca = pca_apply(data_values, 3)
    
    print(np.mean(new_data, axis=0))
    data['PCA_1'] = new_data[:, 0]
    data['PCA_2'] = new_data[:, 1]
    print(data)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    ratio = pca.explained_variance_ratio_
    ratio = np.cumsum(ratio)
    print(ratio)

    classes = list(set(data['output']))
    print(classes)
    z = pd.factorize(data['output'])[0]
    print(z)
    # z = range(1, len(classes))
    for i in range(len(classes)):
        ind = data['output'] == classes[i]
        plt.scatter(data['PCA_1'][ind], data['PCA_2'][ind],
                    label=classes[i])
    plt.xlabel('PCA_1')
    plt.ylabel('PCA_2')
    plt.title('Scatter Plot for Principle Components in 2 Dimensions')

    plt.legend()
    plt.show()
    print("Hello World")
