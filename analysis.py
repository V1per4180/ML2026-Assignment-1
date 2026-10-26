import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from solution import Solution


class ShapAnalysis:
    """训练一次模型、计算一次 SHAP，供各项分析共用。"""

    def __init__(self, background_size=200, explained_size=500, seed=42):
        # Solution 内部使用相对路径读取训练数据。
        os.chdir(Path(__file__).resolve().parent)
        self.output = Path("shap_results")
        self.output.mkdir(exist_ok=True)
        self.seed = seed
        self.sol = Solution()
        self.train = pd.read_csv("train.csv")
        self._prepare_samples(background_size, explained_size)
        self._compute_shap()

    def _prepare_samples(self, background_size, explained_size):
        """抽取不重叠的背景和解释样本，保留原始类别及年龄。"""
        if background_size < 1 or explained_size < 1:
            raise ValueError("背景样本数和解释样本数必须为正数。")
        if background_size + explained_size > len(self.train):
            raise ValueError("背景样本数与解释样本数之和超过训练集大小。")

        rng = np.random.default_rng(self.seed)
        rows = rng.choice(
            len(self.train), size=background_size + explained_size, replace=False
        )
        features = self.train.drop(columns=["id", "age"])
        self.background = self.sol.preprocess(features.iloc[rows[:background_size]])
        self.raw_explained = self.train.iloc[rows[background_size:]].copy()
        self.explained = self.sol.preprocess(features.iloc[rows[background_size:]])

    def _compute_shap(self):
        """计算 SHAP 值，所有分析共享相同背景和解释结果。"""
        masker = shap.maskers.Independent(
            self.background, max_samples=len(self.background)
        )
        self.explainer = shap.LinearExplainer(
            (self.sol.model.weights, self.sol.model.bias), masker
        )
        self.explanation = self.explainer(self.explained)
        self.values = self.explanation.values

    def analyze_feature_importance(self):
        """导出编码列重要性，并绘制 mean(|SHAP|) 前十名。
        （from GPT）
        """
        importance = pd.DataFrame({
            "feature": self.explained.columns,
            "mean_abs_shap": np.abs(self.values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(self.output / "importance.csv", index=False)
        print("贡献最大的 10 个特征列：")
        print(importance.head(10).to_string(index=False))

        top10 = importance.head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top10["feature"], top10["mean_abs_shap"], color="#4C72B0")
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_xlabel("Mean absolute SHAP value (years)")
        ax.set_title("Top 10 Features by Mean Absolute SHAP Value")
        ax.margins(x=0.15)
        fig.tight_layout()
        fig.savefig(self.output / "mean_abs_shap_top10.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        return importance

    def plot_shap_bar(self):
        """绘制 SHAP 默认的全局重要性条形图。（from GPT）"""
        shap.summary_plot(self.values, self.explained, plot_type="bar",
                          max_display=15, show=False)
        fig = plt.gcf()
        fig.savefig(self.output / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_shap_summary(self):
        """绘制各编码列的 SHAP 分布图。"""
        np.random.seed(self.seed)
        shap.summary_plot(self.values, self.explained, max_display=15, show=False)
        fig = plt.gcf()
        fig.savefig(self.output / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def analyze_categorical_features(self, features=("marital", "job", "education")):
        """分别分析婚姻状态、职业、学历，并导出合并统计表。"""
        summaries = [self.analyze_categorical_feature(feature) for feature in features]
        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(self.output / "category_shap_summary.csv", index=False)
        return summary

    def analyze_categorical_feature(self, feature):
        """按实际类别统计该特征组的贡献，不混入交叉特征。

        先对每个样本的全部同组 one-hot 列做带符号求和，然后按
        实际类别统计；不是仅统计该类别对应的一个 dummy 列。
        Independent LinearExplainer 下，同一类别的组贡献为常数，
        因而采用均值条形图，而不是退化成线的箱线图。
        （from GPT）
        """
        if feature not in self.raw_explained.columns:
            raise ValueError(f"原始数据中不存在特征：{feature}")
        # 完整的 'feature=' 前缀避免将 job+education=... 混入。
        columns = self.explained.columns.str.startswith(f"{feature}=")
        if not columns.any():
            raise ValueError(f"找不到 {feature} 对应的独热编码列。")

        samples = pd.DataFrame({
            "source_row": self.raw_explained.index.to_numpy(),
            "id": self.raw_explained["id"].to_numpy(),
            "feature": feature,
            # NaN 未被原始 one-hot 编码为 unknown，使用单独的显示标签。
            "category": self.raw_explained[feature].fillna("<missing>").to_numpy(),
            "shap_value": self.values[:, columns].sum(axis=1),
        })
        samples.to_csv(self.output / f"{feature}_shap_samples.csv", index=False)

        summary = samples.groupby("category", sort=True)["shap_value"].agg(
            count="size",
            mean_shap="mean",
            median_shap="median",
            std_shap=lambda values: values.std(ddof=0),
            mean_abs_shap=lambda values: values.abs().mean(),
            min_shap="min",
            max_shap="max",
        ).reset_index()
        summary.insert(0, "feature", feature)
        summary["sample_fraction"] = summary["count"] / len(samples)
        summary = summary.sort_values("mean_shap").reset_index(drop=True)
        summary.to_csv(self.output / f"{feature}_shap_summary.csv", index=False)
        self._plot_category_contributions(feature, summary)

        print(f"\n{feature} 按实际类别的 SHAP 贡献（单位：岁，不含交叉项）：")
        print(summary[["category", "count", "mean_shap", "mean_abs_shap"]]
              .to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return summary

    def analyze_education_job(self):
        """按实际学历、职业汇总交叉特征组的 SHAP，并绘制热力图。

        每个样本先对全部 job+education=... 列带符号求和。
        这是交叉特征组在当前模型中的贡献，不是纯交互效应。
        （from GPT）
        """
        columns = self.explained.columns.str.startswith("job+education=")
        if not columns.any():
            raise ValueError("找不到 job+education 对应的独热编码列。")

        samples = pd.DataFrame({
            "source_row": self.raw_explained.index.to_numpy(),
            "id": self.raw_explained["id"].to_numpy(),
            # 与 Solution 中构造交叉特征时的缺失值处理保持一致。
            "education": self.raw_explained["education"].fillna("unknown").astype(str).to_numpy(),
            "job": self.raw_explained["job"].fillna("unknown").astype(str).to_numpy(),
            "shap_value": self.values[:, columns].sum(axis=1),
        })
        samples.to_csv(self.output / "education_job_shap_samples.csv", index=False)

        education_levels = self.train["education"].fillna("unknown").astype(str).unique()
        preferred_order = ["primary", "secondary", "tertiary", "unknown"]
        education_levels = (
            [level for level in preferred_order if level in education_levels]
            + sorted(set(education_levels) - set(preferred_order))
        )
        job_levels = sorted(self.train["job"].fillna("unknown").astype(str).unique())
        combinations = pd.MultiIndex.from_product(
            [education_levels, job_levels], names=["education", "job"]
        )
        summary = samples.groupby(["education", "job"])["shap_value"].agg(
            count="size", mean_shap="mean",
            mean_abs_shap=lambda values: values.abs().mean(),
        ).reindex(combinations)
        # 未出现在解释样本中的组合保留 NaN，不能将其贡献当作零。
        summary["count"] = summary["count"].fillna(0).astype(int)
        summary = summary.reset_index()
        summary.to_csv(self.output / "education_job_shap_summary.csv", index=False)
        means = summary.pivot(index="education", columns="job", values="mean_shap").reindex(
            index=education_levels, columns=job_levels
        )
        counts = summary.pivot(index="education", columns="job", values="count").reindex(
            index=education_levels, columns=job_levels
        )
        self._plot_education_job_heatmap(means, counts)

        print("\n学历 × 职业交叉特征组的重要性 mean(|组 SHAP|)："
              f"{samples['shap_value'].abs().mean():.3f} 岁")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return summary

    def _plot_education_job_heatmap(self, means, counts):
        """红色推高预测年龄，蓝色压低；灰色表示没有解释样本。
            （from GPT）
        """
        values = means.to_numpy(dtype=float)
        limit = max(float(np.nanmax(np.abs(values))), 1e-6)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#E5E5E5")
        fig, ax = plt.subplots(figsize=(16, 5.5))
        heatmap = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                            vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(np.arange(len(means.columns)), labels=means.columns,
                      rotation=40, ha="right")
        ax.set_yticks(np.arange(len(means.index)), labels=means.index)
        ax.set_xlabel("Job")
        ax.set_ylabel("Education")
        ax.set_title("Education x Job: Mean SHAP Contribution of the Cross Feature Group")
        for row in range(len(means.index)):
            for col in range(len(means.columns)):
                value = values[row, col]
                count = int(counts.iloc[row, col])
                label = "No samples" if count == 0 else f"{value:+.2f}\n(n={count})"
                color = "white" if np.isfinite(value) and abs(value) > limit * 0.65 else "#222222"
                ax.text(col, row, label, ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(np.arange(len(means.columns) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(means.index) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(heatmap, ax=ax, label="Mean signed SHAP contribution (years)")
        fig.text(0.5, 0.02,
                 "Cross feature group only; education and job main features excluded.\n"
                 "n = explained sample count (sampled from training data); gray = no samples.",
                 ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(self.output / "education_job_shap_heatmap.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    def _plot_category_contributions(self, feature, summary):
        """并排展示带正负号的类别贡献和解释样本量（from GPT）。"""
        positions = np.arange(len(summary))
        colors = np.where(summary["mean_shap"] >= 0, "#C44E52", "#4C72B0")
        fig, (ax, count_ax) = plt.subplots(
            1, 2, figsize=(12, max(4, 0.45 * len(summary) + 2)),
            sharey=True, gridspec_kw={"width_ratios": [3, 1]}
        )
        bars = ax.barh(positions, summary["mean_shap"], color=colors)
        ax.bar_label(bars, fmt="%+.3f", padding=4, fontsize=9)
        ax.axvline(0, color="#444444", linewidth=1)
        ax.set_yticks(positions, labels=summary["category"])
        ax.set_xlabel("Mean signed SHAP contribution (years)")
        ax.set_title(f"{feature}: contribution by actual category")
        ax.margins(x=0.25)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)

        count_bars = count_ax.barh(positions, summary["count"], color="#999999")
        count_ax.bar_label(count_bars, fmt="%d", padding=4, fontsize=9)
        count_ax.set_xlabel("Explained sample count")
        count_ax.set_title("Sample size")
        count_ax.tick_params(axis="y", left=False, labelleft=False)
        count_ax.margins(x=0.25)

        fig.tight_layout(rect=(0, 0.15, 1, 1))
        fig.savefig(self.output / f"{feature}_shap_by_category.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def main():
    analysis = ShapAnalysis()
    analysis.analyze_feature_importance()
    analysis.plot_shap_bar()
    analysis.plot_shap_summary()
    analysis.analyze_categorical_features()
    analysis.analyze_education_job()


if __name__ == "__main__":
    main()
