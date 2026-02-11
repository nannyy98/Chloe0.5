"""
Feature Builder for Forecast Service
Creates market microstructure and regime-aware features
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """Builds features for forecasting models"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}
        
        logger.info("⚙️ Feature Builder initialized")
    
    def build_features(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """Build comprehensive feature set for forecasting"""
        df_features = df.copy()
        
        # Basic price features
        df_features = self._add_price_features(df_features)
        
        # Log returns (multi-horizon)
        df_features = self._add_log_returns(df_features, horizon)
        
        # Volatility features
        df_features = self._add_volatility_features(df_features)
        
        # Volume features
        df_features = self._add_volume_features(df_features)
        
        # Momentum and reversal features
        df_features = self._add_momentum_features(df_features)
        
        # Regime features
        df_features = self._add_regime_features(df_features)
        
        # Market microstructure features
        df_features = self._add_microstructure_features(df_features)
        
        # Target variables
        df_features = self._add_targets(df_features, horizon)
        
        # Clean and validate features
        df_features = self._clean_features(df_features)
        
        logger.info(f"📊 Built {len([c for c in df_features.columns if c.startswith('feature_')])} features")
        return df_features
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        df['feature_price_normalized'] = (
            (df['close'] - df['close'].rolling(252).mean()) / 
            df['close'].rolling(252).std()
        )
        
        # Price relative to moving averages
        for ma_period in [5, 10, 20, 50, 200]:
            df[f'feature_price_ma{ma_period}'] = (
                df['close'] / df['close'].rolling(ma_period).mean() - 1
            )
        
        # High/Low ratios
        df['feature_high_low_ratio'] = df['high'] / df['low']
        df['feature_body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        return df
    
    def _add_log_returns(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add log returns features"""
        # Multi-horizon returns
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'feature_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Return volatility
        for period in [5, 10, 20]:
            df[f'feature_return_vol_{period}d'] = (
                df['feature_return_1d'].rolling(period).std() * np.sqrt(252)
            )
        
        # Return skewness and kurtosis
        for period in [5, 10, 20]:
            df[f'feature_return_skew_{period}d'] = (
                df['feature_return_1d'].rolling(period).skew()
            )
            df[f'feature_return_kurt_{period}d'] = (
                df['feature_return_1d'].rolling(period).kurt()
            )
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Realized volatility
        df['feature_rv_5d'] = df['feature_return_1d'].rolling(5).std() * np.sqrt(252)
        df['feature_rv_20d'] = df['feature_return_1d'].rolling(20).std() * np.sqrt(252)
        df['feature_rv_60d'] = df['feature_return_1d'].rolling(60).std() * np.sqrt(252)
        
        # Volatility term structure
        df['feature_vol_term_structure'] = df['feature_rv_20d'] / df['feature_rv_5d']
        
        # Volatility regime
        df['feature_vol_regime'] = (
            df['feature_rv_20d'] / df['feature_rv_20d'].rolling(252).mean()
        )
        
        # GARCH-style features
        df['feature_vol_clustering'] = (
            df['feature_return_1d']**2
        ).rolling(20).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        for period in [5, 10, 20, 60]:
            df[f'feature_vol_ma_{period}d'] = df['volume'].rolling(period).mean()
        
        # Volume ratios
        df['feature_volume_ratio'] = df['volume'] / df['feature_vol_ma_20d']
        
        # Volume-price correlation
        df['feature_vol_price_corr'] = (
            df['volume'].rolling(20).corr(df['feature_return_1d'])
        )
        
        # Volume imbalance
        df['feature_volume_imbalance'] = (
            (df['volume'] - df['feature_vol_ma_20d']) / df['feature_vol_ma_20d']
        )
        
        # Volume acceleration
        df['feature_volume_acceleration'] = (
            df['feature_volume_ratio'].diff()
        )
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and mean reversion features"""
        # Relative Strength Index
        for period in [7, 14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'feature_rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Price position in range
        for period in [5, 10, 20, 50]:
            df[f'feature_price_position_{period}'] = (
                (df['close'] - df['low'].rolling(period).min()) /
                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
            ).clip(0, 1)
        
        # Momentum oscillators
        for period in [5, 10, 20]:
            df[f'feature_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Stochastic oscillator
        for period in [5, 14, 28]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'feature_stoch_{period}'] = (
                (df['close'] - low_min) / (high_max - low_min)
            ).clip(0, 1)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        # Trend strength
        for period in [20, 50, 100]:
            df[f'feature_trend_strength_{period}'] = (
                df['close'] / df['close'].rolling(period).mean() - 1
            ).abs()
        
        # Acceleration/deceleration
        df['feature_price_acceleration'] = df['feature_return_1d'].diff()
        
        # Volatility trend
        df['feature_vol_trend'] = df['feature_rv_20d'].diff()
        
        # Correlation with trend
        df['feature_trend_vol_corr'] = (
            df['feature_return_1d'].rolling(20).corr(df['feature_rv_20d'])
        )
        
        # Regime classification features
        df['feature_regime_state'] = self._classify_regime(
            df['feature_rv_20d'], 
            df['feature_return_1d'].rolling(20).mean()
        )
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread proxy
        df['feature_spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price efficiency
        df['feature_price_efficiency'] = (
            abs(df['close'] - df['open']) / (df['high'] - df['low'])
        )
        
        # Volume-weighted features
        df['feature_vwap'] = (
            (df['high'] + df['low'] + df['close']) / 3
        ).rolling(20).mean()
        df['feature_vwap_deviation'] = df['close'] / df['feature_vwap'] - 1
        
        # Overnight gap
        df['feature_overnight_gap'] = df['open'] / df['close'].shift(1) - 1
        
        # High-low symmetry
        df['feature_symmetry'] = (
            (df['high'] - df['close']) - (df['close'] - df['low'])
        ) / (df['high'] - df['low'])
        
        return df
    
    def _add_targets(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add target variables for forecasting"""
        # Future returns (log returns)
        for h in [1, 3, 5, 10]:
            df[f'target_return_{h}d'] = np.log(df['close'].shift(-h) / df['close'])
        
        # Directional targets
        for h in [1, 3, 5]:
            df[f'target_direction_{h}d'] = (df[f'target_return_{h}d'] > 0).astype(int)
        
        # Volatility targets
        for h in [5, 10]:
            future_returns = df['feature_return_1d'].shift(-(h-1)).rolling(h).std()
            df[f'target_volatility_{h}d'] = future_returns * np.sqrt(252)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Normalize extreme values (winsorize)
        for col in numeric_cols:
            if col.startswith('feature_'):
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        return df
    
    def _classify_regime(self, volatility: pd.Series, trend: pd.Series) -> pd.Series:
        """Classify market regime based on volatility and trend"""
        # Simple regime classification
        high_vol = volatility > volatility.rolling(252).quantile(0.7)
        strong_trend = trend.abs() > trend.rolling(252).std()
        
        regime = pd.Series(0, index=volatility.index)  # Default: NORMAL
        regime.loc[high_vol & strong_trend] = 3  # CRISIS
        regime.loc[high_vol & ~strong_trend] = 2  # HIGH_VOL
        regime.loc[~high_vol & strong_trend] = 1  # TREND
        
        return regime
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names"""
        return [col for col in df.columns if col.startswith('feature_')]
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate feature quality"""
        validation_results = {}
        
        feature_cols = self.get_feature_names(df)
        validation_results['has_features'] = len(feature_cols) > 0
        validation_results['no_missing_values'] = not df[feature_cols].isnull().any().any()
        validation_results['finite_values'] = np.isfinite(df[feature_cols].values).all()
        
        # Check for constant features (zero variance)
        variances = df[feature_cols].var()
        validation_results['no_constant_features'] = not (variances == 0).any()
        
        return validation_results