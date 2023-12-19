#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Структура для представления узла дерева
struct TreeNode {
    size_t index;
    TreeNode* left;
    TreeNode* right;
    TreeNode(size_t i) : index(i), left(nullptr), right(nullptr) {}
};

// Обновление матрицы расстояний после объединения кластеров
void updateDistMatrix(MatrixXd& distMat, size_t newClusterFrontIndex) {
    size_t N = distMat.rows();
    VectorXd newDists = distMat.col(newClusterFrontIndex);
    distMat = 0.5 * (distMat + distMat.transpose());
    distMat.bottomRows(1).noalias() = newDists.transpose();
    distMat.rightCols(1).noalias() = newDists;
    distMat(N - 1, N - 1) = 0.0;
}

// Иерархическая кластеризация
void hierarchicalClustering(MatrixXd& corrMat, TreeNode*& root) {
    size_t N = corrMat.rows();
    MatrixXd distMat = sqrt(0.5 * (1.0 - corrMat.array()));
    cout << "Заданная матрица корреляций:\n" << corrMat << endl;
    while (N > 1) {
        size_t minI = 0, minJ = 0;
        double minDist = numeric_limits<double>::max();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i + 1; j < N; ++j) {
                double dist = distMat(i, j);
                if (dist < minDist) {
                    minDist = dist;
                    minI = i;
                    minJ = j;
                }
            }
        size_t newClusterFrontIndex = minI;
        updateDistMatrix(distMat, newClusterFrontIndex);
        TreeNode* newNode = new TreeNode(N);
        newNode->left = new TreeNode(minI);
        newNode->right = new TreeNode(minJ);
        root = newNode;
        if (N > 2) {
            cout << "\nПолученный на данном этапе кластер:\n" << distMat << endl;
        }
        distMat.row(minI).swap(distMat.row(N - 1));
        distMat.col(minI).swap(distMat.col(N - 1));
        distMat.row(minJ).swap(distMat.row(N - 2));
        distMat.col(minJ).swap(distMat.col(N - 2));
        distMat.conservativeResize(N - 1, N - 1);
        N--;
    }
}

// Квазидиагонализация матрицы
void quasidiagonalisation(MatrixXd& mat) {
    size_t N = mat.rows();
    for (size_t i = 0; i < N; ++i) {
        size_t maxIndex = i;
        for (size_t j = i + 1; j < N; ++j) {
            if (mat(i, j) > mat(i, maxIndex)) {
                maxIndex = j;
            }
        }
        if (maxIndex != i) {
            for (size_t j = 0; j < N; ++j) {
                swap(mat(j, i), mat(j, maxIndex));
            }
        }
        for (size_t k = i + 1; k < N; ++k)
            for (size_t l = i + 1; l < N - 1; ++l)
                if (mat(l, i) < mat(l + 1, i))
                    swap(mat(l, i), mat(l + 1, i));
        for (size_t k = i; k < N - 1; ++k)
            for (size_t l = i; l < N - 1; ++l)
                if (mat(i, l) < mat(i, l + 1))
                    swap(mat(i, l), mat(i, l + 1));
    }
}

// Рекурсивная бисекция
void bisection(TreeNode* node, const MatrixXd& covMatrix, VectorXd& assetWeights) {
    if (node) {
        if (node->left && node->right) {
            bisection(node->left, covMatrix, assetWeights);
            bisection(node->right, covMatrix, assetWeights);
            if (!node->left->left && !node->right->right) {
                size_t clusterIndex = node->index;
                VectorXd invDiagVar = covMatrix.diagonal().cwiseInverse();
                VectorXd omega = invDiagVar / invDiagVar.sum();
                VectorXd alphas = VectorXd::Zero(assetWeights.size());
                for (size_t i = 0; i < assetWeights.size(); ++i) {
                    double sumV = covMatrix.col(clusterIndex).sum();
                    alphas(i) = 1.0 - (covMatrix(i, clusterIndex) / sumV);
                }
                assetWeights = alphas.cwiseProduct(omega);
                cout << "\nПолученные веса активов:\n" << assetWeights.transpose() << endl;
            }
        }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    MatrixXd corrMat(10, 10);
    corrMat <<
        1.00, 0.20, -0.30, 0.50, -0.15, 0.80, -0.10, 0.45, 0.25, -0.05,
        0.20, 1.00, 0.10, -0.25, 0.40, -0.15, 0.60, -0.20, -0.30, 0.70,
        -0.30, 0.10, 1.00, -0.15, 0.20, -0.05, 0.30, 0.40, 0.15, -0.25,
        0.50, -0.25, -0.15, 1.00, -0.10, 0.55, -0.35, 0.10, -0.20, 0.30,
        -0.15, 0.40, 0.20, -0.10, 1.00, -0.25, 0.75, 0.05, 0.60, -0.30,
        0.80, -0.15, -0.05, 0.55, -0.25, 1.00, -0.30, 0.20, -0.40, 0.10,
        -0.10, 0.60, 0.30, -0.35, 0.75, -0.30, 1.00, 0.15, -0.05, 0.50,
        0.45, -0.20, 0.40, 0.10, 0.05, 0.20, 0.15, 1.00, 0.25, -0.15,
        0.25, -0.30, 0.15, -0.20, 0.60, -0.40, -0.05, 0.25, 1.00, -0.10,
        -0.05, 0.70, -0.25, 0.30, -0.30, 0.10, 0.50, -0.15, -0.10, 1.00;

    TreeNode* root = nullptr;
    hierarchicalClustering(corrMat, root);
    quasidiagonalisation(corrMat);
    cout << "\nПолученная квазидиагональная матрица:\n\n" << corrMat << endl;
    VectorXd assetWeights = VectorXd::Ones(corrMat.rows());
    bisection(root, corrMat, assetWeights);
    return 0;
}