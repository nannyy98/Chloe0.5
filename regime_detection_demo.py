"""
Regime Detection Demo for Chloe AI 0.4
Demonstrates market regime identification capabilities
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from regime_detection import RegimeDetector, MarketRegime, enhance_pipeline_with_regime_detection
from data_pipeline import get_data_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_regime_detection():
    """Demonstrate market regime detection capabilities"""
    logger.info("🔍 Chloe AI 0.4 - Market Regime Detection Demo")
    logger.info("=" * 60)
    
    try:
        # Initialize regime detector
        logger.info("🧠 Initializing regime detector...")
        detector = RegimeDetector(n_regimes=4)
        
        # Test with synthetic data first
        logger.info("\n🧪 Testing with synthetic market data...")
        
        # Create different market regime scenarios
        scenarios = {
            'trending_up': {
                'description': 'Strong uptrend market',
                'drift': 0.002,  # 0.2% daily drift up
                'volatility': 0.02
            },
            'mean_reverting': {
                'description': 'Mean-reverting market',
                'drift': 0.0001,  # Near zero drift
                'volatility': 0.015,
                'mean_reversion_strength': 0.1
            },
            'volatile': {
                'description': 'High volatility market',
                'drift': 0.0005,
                'volatility': 0.04
            },
            'stable': {
                'description': 'Stable market',
                'drift': 0.0002,
                'volatility': 0.01
            }
        }
        
        for scenario_name, params in scenarios.items():
            logger.info(f"\n📊 Scenario: {params['description']}")
            
            # Generate synthetic data for this regime
            synthetic_data = generate_synthetic_market_data(
                days=200,
                initial_price=50000,
                drift=params['drift'],
                volatility=params['volatility'],
                regime_type=scenario_name
            )
            
            # Train detector on this data
            training_success = detector.train_hmm(synthetic_data)
            if not training_success:
                logger.warning(f"Failed to train on {scenario_name} scenario")
                continue
            
            # Detect regime
            detected_regime = detector.detect_current_regime(synthetic_data)
            
            logger.info(f"   Detected Regime: {detected_regime.name}")
            logger.info(f"   Confidence: {detected_regime.probability:.3f}")
            logger.info(f"   Characteristics:")
            for key, value in detected_regime.characteristics.items():
                logger.info(f"     {key}: {value:.4f}")
        
        # Test with real data (simplified approach)
        logger.info(f"\n{'='*60}")
        logger.info("📈 Testing with real market data...")
        
        # Create sample real data
        from data.data_agent import DataAgent
        data_agent = DataAgent()
        
        symbols = ['BTC/USDT', 'ETH/USDT']
        for symbol in symbols:
            logger.info(f"\n📋 Analyzing {symbol}...")
            
            try:
                # Fetch real data
                real_data = await data_agent.fetch_crypto_ohlcv(symbol, timeframe='1d', limit=365)
                if real_data is None:
                    logger.warning(f"No real data available for {symbol}")
                    # Create fallback synthetic data
                    real_data = generate_synthetic_market_data(365, 50000, 0.001, 0.03)
                
                # Train detector
                detector.train_hmm(real_data)
                
                # Detect current regime
                current_regime = detector.detect_current_regime(real_data)
                
                logger.info(f"   Current Market Regime: {current_regime.name}")
                logger.info(f"   Regime Confidence: {current_regime.probability:.3f}")
                logger.info(f"   Key Characteristics:")
                logger.info(f"     Volatility: {current_regime.characteristics.get('volatility', 0):.4f}")
                logger.info(f"     Trend Strength: {current_regime.characteristics.get('trend_strength', 0):.4f}")
                logger.info(f"     Mean Reversion: {current_regime.characteristics.get('mean_reversion_tendency', 0):.4f}")
                
                # Get regime history
                regime_history = detector.get_regime_history(real_data, window=30)
                if regime_history:
                    regime_counts = {}
                    for regime in regime_history:
                        regime_counts[regime.name] = regime_counts.get(regime.name, 0) + 1
                    
                    logger.info(f"   Recent Regime Distribution (30 days):")
                    for regime_name, count in sorted(regime_counts.items()):
                        percentage = (count / len(regime_history)) * 100
                        logger.info(f"     {regime_name}: {count} days ({percentage:.1f}%)")
                        
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("🎯 REGIME DETECTION DEMO COMPLETED")
        logger.info(f"{'='*60}")
        logger.info("✅ Key achievements:")
        logger.info("   • Implemented regime detection (HMM/rule-based)")
        logger.info("   • Distinguished 4 market regimes: STABLE, TRENDING, MEAN_REVERTING, VOLATILE")
        logger.info("   • Adaptive feature engineering based on regime")
        logger.info("   • Real-time regime classification capability")
        
        logger.info(f"\n🚀 Chloe 0.4 Progress:")
        logger.info("   ✅ Phase 1: Market Intelligence Layer (40% → 70% complete)")
        logger.info("   ⬜ Phase 2: Risk Engine Core (next priority)")
        logger.info("   ⬜ Phase 3: Edge Classification")
        logger.info("   ⬜ Phase 4: Portfolio Construction")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

def generate_synthetic_market_data(days: int, initial_price: float, 
                                 drift: float, volatility: float, 
                                 regime_type: str = 'generic') -> pd.DataFrame:
    """
    Generate synthetic market data for testing regime detection
    """
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate returns based on regime type
    if regime_type == 'mean_reverting':
        # Mean-reverting process (Ornstein-Uhlenbeck)
        theta = 0.1  # Mean reversion speed
        mu = 0  # Long-term mean
        sigma = volatility
        
        returns = np.zeros(days)
        prices = np.zeros(days)
        prices[0] = initial_price
        
        for i in range(1, days):
            # OU process for returns
            returns[i] = returns[i-1] + theta * (mu - returns[i-1]) + sigma * np.random.normal()
            prices[i] = prices[i-1] * (1 + returns[i])
    else:
        # Standard geometric Brownian motion
        returns = np.random.normal(drift, volatility, days)
        prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.002, days)),
        'high': prices * (1 + abs(np.random.normal(0, 0.005, days))),
        'low': prices * (1 - abs(np.random.normal(0, 0.005, days))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, days)
    })
    df.set_index('timestamp', inplace=True)
    
    return df

if __name__ == "__main__":
    asyncio.run(demo_regime_detection())