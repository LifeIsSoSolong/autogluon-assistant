# -*- coding: utf-8 -*-
# @Time    : 2025/7/24 18:57
# @Author  : likaixiang
# @FileName: feature_fusion.py
# @Software: frontis
import json
import os
import traceback

import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from src.data_process.standard_module import FeaturePreprocessor
# 构建特征数据集

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/feature_fusion_weekly.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 填充方法映射
FILL_METHODS = {
    'ffill': '向前填充',
    'bfill': '向后填充',
    'interpolate': '线性插值',
    'zero': '填充0',
    'mean': '填充均值',
    'median': '填充中位数',
    'drop': '删除缺失值',
    'weekly_lag1': '周粒度滞后1期并对齐到日',
    'monthly_lag1_daily': '月度粒度滞后1期并对齐到日',
    'monthly_lag1_weekly': '月度粒度滞后1期并对齐到周',
    'seasonal_mean': '季节性均值填充',
    'weekly_lag1_weekly': '周粒度滞后1期并对齐到周' # weekly 新增
}


class FeatureFusion:
    def __init__(self, config_path: str):
        """初始化特征融合器，从配置文件加载配置"""
        self.config = self._load_config(config_path)
        self.target_df = None
        self.feature_dfs = {}
        self.final_df = None
        self.feature_groups = list(self.config.get('features', {}).keys())
        self.weekly_freq = 'W-FRI'

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"成功加载配置文件: {config_path}")
                return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def load_data(self) -> None:
        """加载目标文件和特征文件"""
        load_method = "series"


        if load_method == "file": #这部分没改成周频
            # 加载目标文件
            target_path = self.config['target_file']
            try:
                self.target_df = pd.read_csv(target_path, parse_dates=['date'])
                self.target_df.set_index('date', inplace=True)
                logger.info(f"成功加载目标文件: {target_path}, 行数: {len(self.target_df)}")
            except Exception as e:
                logger.error(f"加载目标文件失败: {e}")
                logger.error(traceback.format_exc())
                raise

            # 加载特征文件（按分组）
            for group_name, features in self.config.get('features', {}).items():
                logger.info(f"开始加载特征组: {group_name}")
                for feature in features:
                    feature_path = feature['file_path']
                    feature_name = feature['feature_name']
                    try:
                        df = pd.read_csv(feature_path, parse_dates=['date'])
                        df.set_index('date', inplace=True)
                        self.feature_dfs[feature_name] = df
                        logger.info(f"成功加载特征文件: {feature_path}, 行数: {len(df)}")
                    except Exception as e:
                        logger.error(f"加载特征文件失败: {feature_name}, 错误: {e}")
                        logger.error(traceback.format_exc())  # 新增：输出完整堆栈信息
                        raise

        elif load_method == "series":
            data_path = self.config['data_file']
            try:
                target_name = self.config['target_name']
                # 读取完整的日度数据表
                daily_df = pd.read_excel(data_path, parse_dates=['date'])
                daily_df.set_index('date', inplace=True)

                # 【修改点】将目标变量（价格）聚合为周度，使用每周的最后一个值
                target_series_weekly = daily_df[target_name].resample(self.weekly_freq).last().dropna()
                self.target_df = pd.DataFrame(target_series_weekly)
                self.target_df.rename(columns={target_name: "value"}, inplace=True)
                logger.info(f"成功加载并聚合目标'{target_name}'为周频, 行数: {len(self.target_df)}")

            except Exception as e:
                logger.error(f"加载或聚合目标文件失败: {e}")
                logger.error(traceback.format_exc())
                raise

            # 【修改点】加载所有特征，并根据其性质聚合到周频。加载数据后就进行聚合了
            for group_name, features in self.config.get('features', {}).items():
                logger.info(f"开始加载并转换特征组: {group_name}")
                for feature_config in features:
                    feature_name = feature_config['feature_name']
                    # 【修改点】从配置中读取聚合方法，默认为'last'
                    agg_method = feature_config.get('agg_method', 'identity') # json里没写agg_method就不聚合,默认identity表示不变
                    try:
                        feature_series_daily = daily_df[[feature_name]].dropna()
                        
                        # 根据原始数据频率判断是否需要聚合
                        if agg_method == 'last':
                            feature_series_weekly = feature_series_daily.resample(self.weekly_freq).last()
                        elif agg_method == 'mean':
                            feature_series_weekly = feature_series_daily.resample(self.weekly_freq).mean()
                        elif agg_method == 'sum':
                            feature_series_weekly = feature_series_daily.resample(self.weekly_freq).sum()
                        else: # 如果已经是周频或月频，则不重采样
                             feature_series_weekly = feature_series_daily
                        
                        feature_df_weekly = pd.DataFrame(feature_series_weekly)
                        feature_df_weekly.rename(columns={feature_name: "value"}, inplace=True)
                        self.feature_dfs[feature_name] = feature_df_weekly
                        logger.info(f"成功加载并转换特征'{feature_name}'为周频 (使用 {agg_method} 聚合), 行数: {len(feature_df_weekly)}")

                    except Exception as e:
                        logger.error(f"加载或转换特征失败: {feature_name}, 错误: {e}")
                        logger.error(traceback.format_exc())
                        raise

    def normalize_feature(self,feature_series: pd.Series, normalize_method: str, feature_name: str = "") -> pd.Series:
        """
        根据配置的 normalize_method 对特征序列进行标准化/归一化处理
        参数:
            feature_series: 原始特征序列（pd.Series）
            normalize_method: 标准化/归一化方法名（例如 'zscore', 'minmax', 'log_zscore' 等）
            feature_name: 当前处理的特征名，用于日志记录
        返回:
            处理后的 pd.Series
        """
        try:
            if normalize_method == 'zscore':
                return FeaturePreprocessor.zscore_standardize(feature_series)

            elif normalize_method == 'minmax':
                return FeaturePreprocessor.minmax_scale(feature_series)

            elif normalize_method == 'log_zscore':
                return FeaturePreprocessor.log_standardize(feature_series)

            elif normalize_method == 'diff':
                return FeaturePreprocessor.diff(feature_series)

            elif normalize_method == 'pct_change':
                return FeaturePreprocessor.pct_change(feature_series)

            elif normalize_method == 'rolling_mean':
                return FeaturePreprocessor.rolling_mean(feature_series)

            elif normalize_method == 'rolling_std':
                return FeaturePreprocessor.rolling_std(feature_series)

            elif normalize_method == 'robust':
                return FeaturePreprocessor.robust_scale(feature_series)

            elif normalize_method == 'winsorize_standardize':
                return FeaturePreprocessor.winsorize_standardize(feature_series)

            elif normalize_method == 'none' or normalize_method is None:
                return feature_series

            else:
                logger.warning(f"未知标准化方法: {normalize_method}，特征名: {feature_name}，默认不处理")
                return feature_series

        except Exception as e:
            logger.error(f"处理特征标准化失败: {feature_name}, 方法: {normalize_method}, 错误: {e}")
            logger.error(traceback.format_exc())
            raise

    def process_feature(self, feature: Dict[str, Any]) -> pd.Series:
        """处理单个特征，根据指定方法进行填充"""
        feature_name = feature['feature_name']
        fill_method = feature['fill_method']
        normalize_method = feature.get('normalize_method')
        source_column = feature.get('source_column', 'value')

        logger.info(f"处理特征: {feature_name}, 填充方法: {FILL_METHODS[fill_method]}, 标准化方法：{normalize_method}")

        # 获取特征数据
        feature_series = self.feature_dfs[feature_name][source_column]

        # 先对齐到目标日期，生成缺失值
        aligned_series = feature_series.reindex(self.target_df.index, method='ffill')

        # 计算对齐前的缺失值情况
        original_nan = feature_series.isna().sum()
        aligned_nan = aligned_series.isna().sum()
        logger.info(f"特征: {feature_name}, 原始缺失值: {original_nan}, 对齐后缺失值: {aligned_nan}")

        # 根据填充方法处理缺失值（填充）
        try:
            if fill_method == 'weekly_lag1':
                filled_feature = self._apply_weekly_lag(feature_series)
            elif fill_method == 'weekly_lag1_weekly': #新增，周数据只滞后就行
                filled_feature = aligned_series.shift(1)
            elif fill_method == 'monthly_lag1_daily': 
                filled_feature = self._apply_monthly_lag_to_daily(feature_series)
            elif fill_method == 'monthly_lag1_weekly': #月度数据在这里完成fill
                filled_feature = self._apply_monthly_lag_to_weekly(feature_series)
            elif fill_method == 'seasonal_mean':
                filled_feature = self._apply_seasonal_mean(aligned_series)
            elif fill_method == 'ffill':
                filled_feature = aligned_series.ffill()
                filled_feature = filled_feature.shift(1)
            elif fill_method == 'bfill':
                filled_feature = aligned_series.bfill()
            elif fill_method == 'interpolate':
                filled_feature = aligned_series.interpolate(method='linear')
            elif fill_method == 'zero':
                filled_feature = aligned_series.fillna(0)
            elif fill_method == 'mean':
                filled_feature = aligned_series.fillna(aligned_series.mean())
            elif fill_method == 'median':
                filled_feature = aligned_series.fillna(aligned_series.median())
            elif fill_method == 'drop':
                filled_feature = aligned_series.dropna()
            else:
                logger.warning(f"未知填充方法: {fill_method}，使用默认填充: ffill")
                filled_feature = aligned_series.ffill()
        except Exception as e:
            logger.error(f"处理特征填充失败: {feature_name}, 填充方法: {fill_method}, 错误: {e}")
            logger.error(traceback.format_exc())  # 新增：输出完整堆栈信息
            raise

        # 记录最终填充统计信息
        final_nan = filled_feature.isna().sum()
        logger.info(f"特征: {feature_name}, 最终缺失值: {final_nan}")

        # 根据根据标准化方法进行标准化处理
        filled_feature = self.normalize_feature(filled_feature,normalize_method,feature_name)

        return filled_feature

    def _apply_weekly_lag(self, feature_series: pd.Series) -> pd.Series:
        """应用周粒度滞后特征工程"""
        # 确保索引是日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(feature_series.index):
            feature_series.index = pd.to_datetime(feature_series.index)

        # 将周粒度数据滞后1期
        weekly_lagged = feature_series.shift(1, freq='W')

        # 转换为日粒度并向前填充
        daily_lagged = weekly_lagged.resample('D').ffill()

        # 对齐到目标日期
        return daily_lagged.reindex(self.target_df.index)
    def _apply_monthly_lag_to_daily(self, feature_series: pd.Series) -> pd.Series:
        """应用月度粒度滞后特征工程并转换为日度"""
        # 确保索引是日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(feature_series.index):
            feature_series.index = pd.to_datetime(feature_series.index)

        # 将月度数据滞后1期
        monthly_lagged = feature_series.shift(1, freq='M')

        # 转换为日粒度并向前填充
        daily_lagged = monthly_lagged.resample('D').ffill()

        # 对齐到目标日期
        return daily_lagged.reindex(self.target_df.index)

    def _apply_monthly_lag_to_weekly(self, feature_series: pd.Series) -> pd.Series:
        """应用月度粒度滞后特征工程并转换为周度"""
        # 确保索引是日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(feature_series.index):
            feature_series.index = pd.to_datetime(feature_series.index)

        # 将月度数据滞后1期
        monthly_lagged = feature_series.shift(1, freq='M')

        # 转换为周粒度并向前填充
        weekly_lagged = monthly_lagged.resample(self.weekly_freq).ffill()

        # 对齐到目标日期
        return weekly_lagged.reindex(self.target_df.index)

    def _apply_seasonal_mean(self, aligned_series: pd.Series) -> pd.Series:
        """应用季节性均值填充（处理对齐后的序列）"""
        # 确保索引是日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(aligned_series.index):
            aligned_series.index = pd.to_datetime(aligned_series.index)

        # 分析数据频率（改进版，处理非连续数据）
        frequency = self._analyze_data_frequency(aligned_series.index)

        if frequency == 'hourly':
            seasonal_means = aligned_series.groupby(aligned_series.index.hour).transform('mean')
        elif frequency == 'daily':
            seasonal_means = aligned_series.groupby(aligned_series.index.dayofweek).transform('mean')
        elif frequency == 'weekly':
            seasonal_means = aligned_series.groupby(aligned_series.index.week).transform('mean')
        elif frequency == 'monthly':
            seasonal_means = aligned_series.groupby(aligned_series.index.month).transform('mean')
        else:
            logger.warning(f"无法确定季节性类型，使用向前填充")
            return aligned_series.ffill()

        # 使用季节性均值填充缺失值
        filled = aligned_series.fillna(seasonal_means)
        return filled

    def _analyze_data_frequency(self, index: pd.DatetimeIndex) -> str:
        """分析数据频率，处理非连续数据"""
        # 计算平均时间间隔
        if len(index) < 2:
            return 'unknown'

        diffs = index.to_series().diff().dropna()
        avg_diff = diffs.mean()

        # 检查是否主要是工作日数据（处理非连续日数据）
        if avg_diff >= pd.Timedelta('1D') and avg_diff <= pd.Timedelta('3D'):
            # 检查是否大部分间隔是1天或2天（周末）
            weekday_diffs = diffs.apply(lambda x: x.days)
            if (weekday_diffs.isin([1, 3])).mean() > 0.8:
                return 'daily'  # 工作日数据

        # 根据平均间隔判断频率
        if avg_diff < pd.Timedelta('1H'):
            return 'hourly'
        elif avg_diff < pd.Timedelta('1D'):
            return 'daily'
        elif avg_diff < pd.Timedelta('7D'):
            return 'weekly'
        elif avg_diff < pd.Timedelta('30D'):
            return 'monthly'
        else:
            return 'unknown'

    def merge_features(self) -> None:
        """合并所有特征到目标数据"""
        if self.target_df is None:
            logger.error("目标数据未加载")
            return

        # 复制目标数据
        self.final_df = self.target_df.copy()

        # 按分组合并特征
        for group_name, features in self.config.get('features', {}).items():
            logger.info(f"开始合并特征组: {group_name}")
            for feature in features:
                feature_name = feature['feature_name']

                try:
                    # 处理特征
                    processed_feature = self.process_feature(feature)

                    # 合并到最终数据框
                    self.final_df[feature_name] = processed_feature
                    logger.info(f"成功合并特征: {feature_name}")
                except Exception as e:
                    logger.error(f"合并特征失败: {feature_name}, 错误: {e}")
                    logger.error(traceback.format_exc())  # 新增：输出完整堆栈信息
                    continue

    def remove_nan_rows(self) -> None:
        """移除含有NaN值的行并记录数据变动"""
        if self.final_df is None:
            logger.error("没有可处理的数据")
            return

        # 记录原始行数
        original_rows = len(self.final_df)

        # 记录各列的NaN情况
        nan_counts = self.final_df.isna().sum()
        logger.info("合并后各列的NaN数量:")
        for col, count in nan_counts.items():
            logger.info(f"{col}: {count}")

        # 移除含有NaN的行
        self.final_df = self.final_df.dropna()

        # 记录处理后的行数
        final_rows = len(self.final_df)
        removed_rows = original_rows - final_rows

        logger.info(f"数据清洗: 原始行数={original_rows}, 移除行数={removed_rows}, 最终行数={final_rows}")
        logger.info(f"数据变动百分比: {removed_rows / original_rows * 100:.2f}%")

    def save_result(self) -> None:
        """保存结果到CSV文件"""
        if self.final_df is None:
            logger.error("没有可保存的结果")
            return

        output_path = self.config.get('output_file', 'feature_fusion_result.csv')

        try:
            self.final_df.to_csv(output_path)
            logger.info(f"成功保存结果到: {output_path}, 行数: {len(self.final_df)}, 列数: {len(self.final_df.columns)}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def align_model(self):
        """
        对齐特征数据到最新日期
        :return:
        """
        df_sorted = self.final_df.sort_values("date")  # 按日期升序排序
        self.final_df = df_sorted.ffill()  # 前向填充（用最新非空值填充）


def batch_process_configs(config_dir: str) -> None:
    """批量处理配置目录中的所有配置文件"""
    if not os.path.exists(config_dir):
        logger.error(f"配置目录不存在: {config_dir}")
        return

    # 获取所有配置文件
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]

    if not config_files:
        logger.warning(f"配置目录中没有找到JSON配置文件: {config_dir}")
        return

    logger.info(f"找到 {len(config_files)} 个配置文件")

    # 逐个处理配置文件
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        logger.info(f"开始处理配置: {config_file}")

        try:
            fusion = FeatureFusion(config_path)
            fusion.load_data()
            fusion.merge_features()
            fusion.align_model()
            fusion.remove_nan_rows()
            fusion.save_result()
            logger.info(f"配置 {config_file} 处理完成")
        except Exception as e:
            logger.error(f"处理配置 {config_file} 失败: {e}")
            logger.error(traceback.format_exc())  #
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='特征融合工具-根据特征配置列表，将不同的特征融合成一个文件')
    # parser.add_argument('--config', default="script/mysteel/期货-05合约-v0.json",help='单个配置文件路径')
    parser.add_argument('--config', default="src/data_process/data_config/期货-01合约-v0-0929_week_2.1.json",help='单个配置文件路径')
    parser.add_argument('--config_dir', default=None, help='配置文件目录路径')

    args = parser.parse_args()


    if args.config_dir:
        # 批量处理配置目录
        logger.info(f"使用批量配置文件模式: {args.config_dir}")
        batch_process_configs(args.config_dir)
    elif args.config:
        # 处理单个配置文件
        logger.info(f"使用单个配置文件模式: {args.config}")
        try:
            fusion = FeatureFusion(args.config)
            fusion.load_data()
            fusion.merge_features()
            fusion.align_model()
            # fusion.remove_nan_rows()
            fusion.save_result()
            logger.info("特征融合完成")
        except Exception as e:
            logger.error(f"特征融合失败: {e}")
            logger.error(traceback.format_exc())

    else:
        logger.error("请指定配置文件或配置目录")


