import pandas as pd
import numpy as np
from model import Model


# 全局对数据列的分类（不包含age）
NUMERIC_COLUMNS = ("balance", "day", "duration", "campaign", "pdays", "previous")
CATEGORICAL_COLUMNS = (
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
)

TRAIN_PATH = "train.csv"

class Solution:
    def __init__(self):
        """
        初始化随机权重示例，供你检查接口；它不是教师评分基线。
        """
        self.model = Model()
        np.random.seed(42)
        n_features = 15

        self.model.weights = np.random.randn(n_features) * 0.01
        self.model.bias = np.random.randn() * 0.01

        self.means = np.zeros(n_features)
        self.stds = np.ones(n_features)

        self.fit()

    def fit(self):
        """
        读取训练数据集，完成数据预处理和训练。
        """
        # 读取数据，不需要使用 ID 列
        columns = list(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) + ['age']
        train = pd.read_csv(TRAIN_PATH, usecols=columns)

        # 处理数据，得到最终训练使用的X和y
        y_train = train['age'].to_numpy(dtype=np.float64)
        X_train = self.preprocess(train, fit=True)
        self.model = Model().fit(X_train,y_train)

    def preprocess(self, data:pd.DataFrame, fit=False):
        """
        对数据集进行预处理
        Args:
            fit: 在训练时/测试时使用
        """
        
        # 取出数值型特征，做标准化
        numeric = data[list(NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")

        # 新增："以前是否联系过"，因为没联系过与普通的天数含义不同（from gpt）
        numeric["has_previous_contact"] = (numeric["pdays"]>=0).astype(float)
        # 把原来的"pdays"中未联系过直接用0来表示，以区分状态
        numeric["pdays"] = numeric["pdays"].clip(lower=0)

        # 新增：余额和练习时长的对数特征(from gpt)
        numeric["balance_log"] = (
            np.sign(numeric["balance"])
            * np.log1p(numeric["balance"].abs())
        )

        # 新增：duration 的对数特征(from gpt)
        numeric["duration_log"] = np.log1p(
            numeric["duration"].clip(lower=0)
        )


        # 做标准化，测试集内缺失的数据用训练集的补
        if fit:
            self.means = numeric.mean().fillna(0)
        numeric = numeric.fillna(self.means)

        if fit:
            self.stds = numeric.std(ddof=0).replace(0,1)
        numeric = (numeric - self.means) / self.stds

        # 取出类别型特征，做独热编码
        category_data = data[list(CATEGORICAL_COLUMNS)].copy()

        # 对 job 和 education 做交叉
        category_data["job+education"] = (
            category_data["job"].fillna("unknown").astype(str)
            + "|"
            + category_data["education"].fillna("unknown").astype(str)
        )

        # 对 default + housing + loan 做交叉
        category_data["default+housing+loan"] = (
            category_data["default"].fillna("unknown").astype(str)
            + "|"
            + category_data["housing"].fillna("unknown").astype(str)
            + "|"
            + category_data["loan"].fillna("unknown").astype(str)
        )

        # 对类别型特征做独热编码
        categorical = pd.get_dummies(
            category_data, columns=category_data.columns.tolist(),
            prefix_sep="=", dtype=float
        )

        # 拼接处理好的数值型特征和类别型特征
        features = pd.concat([numeric, categorical], axis=1)

        if fit:
            self.feature_names = features.columns.tolist()
            return features

        return features.reindex(
            columns=self.feature_names,
            fill_value=0.0,
        )
    
    def forward(self, sample: dict) -> dict:
        """输入特征，输出年龄预测值"""
        sample = pd.DataFrame([sample])
        X = self.preprocess(sample)
        prediction = float(self.model.predict(X)[0])

        return {'prediction': float(prediction)}
