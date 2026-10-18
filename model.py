import numpy as np

"""多元线性回归模型的实现"""
class Model:
    def __init__(self, alpha=10.0):
        """
        初始化线性回归模型：weights 权重矩阵 和 bias 偏执
        预测：y_hat = X @ weights + bias
        alpha 控制 L2 正则化的程度，默认为 1.0
        """
        self.weights = None
        self.bias = None
        self.alpha = float(alpha)

    def fit(self, X, y):
        """
        用带L2正则化项的目标函数，在数据集(X,y)上训练本模型

        Args:
            X: 训练集特征值
            y: 训练集真实标签
        """
        # 参数的合法性检查
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X 必须是二维矩阵")

        if y.shape != (X.shape[0],):
            raise ValueError("y 必须是一维数组，且长度等于 X 的行数")

        # 为了把偏置 b 也纳入参数向量 θ 中，给 X 补一列 1
        X = np.column_stack((np.ones(len(X)),X))

        # 求出让目标函数最小的参数
        # 目标函数：sum((X @ w + b - y) ** 2) + alpha * sum(w ** 2)
        # 使用正规方程直接求解：(X.T @ X + alpha * I) @ θ = X.T @ y
        XT_X = X.T @ X
        XT_y = X.T @ y
        L2 = self.alpha * np.eye(X.shape[1])
        L2[0,0] = 0.0 # 不需要对偏执进行正则化

        # np.linalg.solve() 直接解线性方程组
        theta = np.linalg.solve(XT_X + L2, XT_y)

        # 第一个元素为偏置，其余的为权重
        self.bias = float(theta[0])
        self.weights = theta[1:]
        return self


    def predict(self, X) -> np.ndarray:
        """
        预测样本的值

        Args:
            X: 输入特征，形状为 (n_samples, n_features) 的numpy数组

        Returns:
            numpy.ndarray: 形状为 (n_samples,) 的预测值数组
        """
        return np.dot(X, self.weights) + self.bias