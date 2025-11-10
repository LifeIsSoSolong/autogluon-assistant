# -*- coding: utf-8 -*-
# @Time    : 2025/8/8 16:41
# @Author  : likaixiang
# @FileName: standard_module.py
# @Software: frontis

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeaturePreprocessor:
    """
    专为时序预测任务设计的特征标准化与归一化工具类
    所有方法默认不改变 index 顺序，适配 pandas.Series
    """

    @staticmethod
    def zscore_standardize(series: pd.Series) -> pd.Series:
        """
        Z-score 标准化（StandardScaler）
        适用场景：大多数模型（线性、树模型、神经网络），适合正态分布或近似正态分布特征

        优点：处理异常值比MinMax更鲁棒；适配大多数模型
        """
        scaler = StandardScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index)

    @staticmethod
    def minmax_scale(series: pd.Series) -> pd.Series:
        """
        Min-Max 归一化
        适用场景：需要将特征缩放到固定区间（如神经网络、距离计算）

        优点：结果在 [0, 1]，便于权重平衡；适合模型对尺度敏感的情况
        缺点：受极端值影响较大
        """
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index)

    @staticmethod
    def log_standardize(series: pd.Series) -> pd.Series:
        """
        对数变换后标准化（适用于长尾、高波动）
        适用场景：极度右偏、方差过大、价格类或宏观经济类特征

        优点：稳定大幅度波动；避免 log(0) 错误使用 log1p
        """
        log_transformed = np.log1p(series.clip(lower=0))  # 避免负数
        scaled = StandardScaler().fit_transform(log_transformed.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index)

    @staticmethod
    def diff(series: pd.Series, period: int = 1) -> pd.Series:
        """
        差分变化（环比变化）
        适用场景：短期变化跟踪（如利润率、库存、就业人数等）

        优点：去趋势、去季节性；模型更聚焦在“变化”而非绝对值
        """
        return series.diff(period)

    @staticmethod
    def pct_change(series: pd.Series, period: int = 1) -> pd.Series:
        """
        百分比变化（相对变化）
        适用场景：金融类指标、库存、持仓量等

        优点：反映变化强度；适合建模趋势强弱
        """
        return series.pct_change(period)

    @staticmethod
    def rolling_mean(series: pd.Series, window: int = 4) -> pd.Series:
        """
        滚动平均（平滑波动）
        适用场景：高频震荡指标（如开工率、船舶数量、持仓变化）

        优点：平滑极端值；增加时序结构性
        """
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def rolling_std(series: pd.Series, window: int = 4) -> pd.Series:
        """
        滚动标准差（反映局部波动性）
        适用场景：风险建模、预测波动性（如库存变化）

        优点：动态度量不确定性；增强模型对高波动区段的感知
        """
        return series.rolling(window=window, min_periods=1).std()

    @staticmethod
    def robust_scale(series: pd.Series) -> pd.Series:
        """
        鲁棒缩放（中位数+IQR）
        适用场景：异常值非常多（如极端产量、港口船只）

        优点：不受极端值影响，分布鲁棒性高
        """
        median = series.median()
        iqr = series.quantile(0.95) - series.quantile(0.05)
        scaled = (series - median) / (iqr + 1e-6)
        return pd.Series(scaled, index=series.index)

    @staticmethod
    def winsorize_standardize(series: pd.Series, lower_pct=0.05, upper_pct=0.95) -> pd.Series:
        """
        对pandas.Series进行上下百分位数截断（Winsorize）+ 标准化（Z-score）

        优点：不受极端值影响，分布鲁棒性高
        """
        # Step 1: Winsorize（截断极端值）
        lower = series.quantile(lower_pct)
        upper = series.quantile(upper_pct)
        capped = series.clip(lower=lower, upper=upper)

        # Step 2: 标准化
        mean = capped.mean()
        std = capped.std()
        standardized = (capped - mean) / std

        return standardized
