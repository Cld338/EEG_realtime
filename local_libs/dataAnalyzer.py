from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sklearn.svm as svm
import pandas as pd
import numpy as np

# =========================================================================

def bandpass_filter(data, sample_rate, cutoff_low, cutoff_high):
    fft_data = fft(data)  # 주파수 영역으로 데이터를 변환

    # 주파수 영역에서 절단 주파수(cutoff frequency)를 벗어나는 주파수를 제거
    fft_data[:int(cutoff_low * len(data) / sample_rate)] = 0
    fft_data[int(cutoff_high * len(data) / sample_rate):] = 0

    filtered_data = ifft(fft_data)  # 필터링된 데이터를 시간 영역으로 변환
    filtered_data = np.real(filtered_data)  # 실수 부분만 사용

    return filtered_data

# =========================================================================

def covariance(X):
    cov_data = np.dot(X, np.transpose(X))
    covariance_matrix = cov_data / X.shape[0]
    return covariance_matrix

def whitening_transform(matrix):
    Lambda, U = np.linalg.eig(matrix)
    Lambda[Lambda < 0] = 0
    Q = np.dot(np.diag(np.sqrt(Lambda)), U.T)
    return Q

def CSP_filter(class1, class2):
    covClass1 = np.array([covariance(i) for i in class1])
    covClass2 = np.array([covariance(i) for i in class2])
    covMeanClass1 = np.mean(covClass1, axis=0)
    covMeanClass2 = np.mean(covClass2, axis=0)
    Q = whitening_transform(covMeanClass1 + covMeanClass2)
    S1 = np.dot(np.dot(Q, covMeanClass2), Q.T) # eigen
    _, B = np.linalg.eig(S1)
    W = np.dot(Q, B)
    return W

# =========================================================================

class PrincipalComponuntAnalysis():
    def __init__(self, n_componunts, data):
        self.analyzer = PCA(n_components=n_componunts) # 주성분을 몇개로 할지 결정
        printcipalComponents = self.analyzer.fit_transform(data)
        self.principalDf = pd.DataFrame(data=printcipalComponents, columns = [f'principal component{i+1}' for i in range(n_componunts)])

    def explained_variance_ratio_(self):
        return self.analyzer.explained_variance_ratio_

# =========================================================================

def LDATransform(data, label, n_components, solver="svd"):
    lda = LinearDiscriminantAnalysis(n_components=n_components, solver=solver)
    lda.fit(data, label)
    transformedData = lda.transform(data)
    return transformedData

# =========================================================================

def plotDF3D(data :pd.DataFrame, num_of_classes :int, colors :list=['r', 'g', 'b', 'c']) -> None:
    data.columns = [f"axis{i+1}" for i in range(len(data.columns)-1)]+["label"]
    # 3D scatter plot 그리기
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')

    # 클래스별로 색상을 다르게 설정
    for i in range(num_of_classes):  # 클래스 개수에 맞게 범위 설정
        subset = data[data['label'] == i]
        ax.scatter(subset['axis1'], subset['axis2'], subset['axis3'], c=colors[i], label=f'Class {i}', alpha=1)

    ax.set_xlabel('axis 1')
    ax.set_ylabel('axis 2')
    ax.set_zlabel('axis 3')
    ax.set_title('3D Scatter Plot of axises')
    ax.legend()
    plt.show()
    return

# =========================================================================

def cross_validation(kernel, data, label):
    # 교차검증
    svm_clf = svm.SVC(kernel = kernel, random_state=100)
    scores = cross_val_score(svm_clf, data, label, cv = 5)
    print('교차검증 평균: ', scores.mean())
    return scores.mean()

# =========================================================================