import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

path = "../Assignment3/lda.txt"
normal_if = True


def pca_apply(data, n_components):
    pca = PCA(n_components=n_components)
    new_data = pca.fit_transform(data)
    return new_data, pca


def lda_details(data):
    # Split data into class 1 and class 2
    data_1 = data.loc[data['output'] == '(black)'].copy()
    data_2 = data.loc[data['output'] == '(blue)'].copy()

    # Compute Mean 1 and Mean 2
    mean_1 = np.reshape(np.mean(data_1.iloc[:, :-1].values, axis=0),
                        [data_1.shape[1]-1, 1])
    mean_2 = np.reshape(np.mean(data_2.iloc[:, :-1].values, axis=0),
                        [data_2.shape[1]-1, 1])
    mean_tmp = mean_1 - mean_2

    # Compute S_B
    S_B = np.matmul(mean_tmp, np.transpose(mean_tmp))

    # Compute S_W
    S_1 = np.cov(data_1.iloc[:, :-1].values.T) * (data_1.shape[0] - 1)
    S_2 = np.cov(data_2.iloc[:, :-1].values.T) * (data_2.shape[0] - 1)
    S_W = S_1 + S_2

    # Compute target mat
    S_W_inv = np.linalg.inv(S_W)
    print(S_W_inv.shape, mean_tmp.shape)
    Chai_vector = np.matmul(S_W_inv, mean_tmp)
    print(Chai_vector.shape)
    Chai_vector = Chai_vector / np.linalg.norm(Chai_vector)
    print("Chai_vector is:\n", Chai_vector)
    S = np.matmul(S_W_inv, S_B)

    # Eigen Decomposition
    eigen_val, eigen_vec = np.linalg.eig(S)
    ind = eigen_val.argsort()[::-1]
    transform = eigen_vec[:, ind]
    print("My vector is:\t", transform[:, 0])
    transform = transform[:, 0][:, None]
    transforms = np.matmul(transform, transform.T)

    new_data_1 = np.matmul(data_1.iloc[:, :-1].values, transforms)
    new_data_2 = np.matmul(data_2.iloc[:, :-1].values, transforms)

    data_1.iloc[:, [0, 1]] = new_data_1
    data_1['output'] = 'transformed (black)'
    data_2.iloc[:, [0, 1]] = new_data_2
    data_2['output'] = 'transformed (blue)'

    newdata = pd.concat([data_1, data_2])
    return newdata, transform


def plot_data(new_data):
    classes = list(set(new_data['output']))
    print(classes)
    z = pd.factorize(new_data['output'])[0]
    print(z)

    print(new_data)
    # z = range(1, len(classes))
    for i in range(len(classes)):
        ind = new_data['output'] == classes[i]
        plt.scatter(new_data.iloc[:, 0][ind], new_data.iloc[:, 1][ind],
                    label=classes[i])
    # plt.xlabel('PCA_1')
    # plt.ylabel('PCA_2')
    # plt.title('Scatter Plot for Principle Components in 2 Dimensions')
    #
    plt.legend()
    plt.show()


def lda_apply(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    lda = LDA()


if __name__ == "__main__":
    data = pd.read_csv(path, delimiter=" ", header=None,
                       index_col=0,
                       names=['distance', 'diameter', 'density', 'output'])
    print("The given data is like follows:\n", data)

    if normal_if:
        norm_data = data.iloc[:, :-1]
        norm_data = (norm_data - norm_data.mean()) / norm_data.std()
        data.iloc[:, :-1] = norm_data

    data_1_2 = data.iloc[:, [0, 1, 3]]
    data_1_3 = data.iloc[:, [0, 2, 3]]
    data_2_3 = data.iloc[:, [1, 2, 3]]
    target_data = data_2_3

    # print("The given data in (1,3) is follows:\n", data_1_2)
    # print("The given data in (2,3) is follows:\n", data_2_3)

    # norm_data = data.iloc[:, :-1]
    # norm_data = (norm_data - norm_data.mean()) / norm_data.std();
    # print(norm_data)
    #
    # data_values = norm_data.values
    # print(data_values)
    # new_data, pca = pca_apply(data_values, 3)

    new_data, transform = lda_details(target_data)
    new_data = pd.concat([new_data, target_data])
    plot_data(new_data)

    # print(np.mean(new_data, axis=0))
    # data['PCA_1'] = new_data[:, 0]
    # data['PCA_2'] = new_data[:, 1]
    # print(data)
    # print(pca.explained_variance_)
    # print(pca.explained_variance_ratio_)
    # ratio = pca.explained_variance_ratio_
    # ratio = np.cumsum(ratio)
    # print(ratio)
    #

    print("Hello World")
