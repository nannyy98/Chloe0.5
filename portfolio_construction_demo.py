"""
Portfolio Construction Demo for Chloe AI 0.4
Demonstrates professional portfolio optimization combining edge classification, risk management, and regime awareness
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from portfolio_constructor import PortfolioConstructor, get_portfolio_constructor
from regime_detection import RegimeDetector
from enhanced_risk_engine import EnhancedRiskEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_portfolio_construction():
    """Demonstrate portfolio construction capabilities"""
    logger.info("📊 Chloe AI 0.4 - Portfolio Construction Demo")
    logger.info("=" * 60)
    
    try:
        # Initialize portfolio constructor
        logger.info("🔧 Initializing portfolio constructor...")
        portfolio_mgr = get_portfolio_constructor(initial_capital=50000.0)
        
        # Initialize components
        regime_detector = RegimeDetector(n_regimes=4)
        risk_engine = EnhancedRiskEngine(initial_capital=50000.0)
        risk_engine.initialize_portfolio_tracking()
        
        logger.info("✅ All components initialized")
        logger.info(f"   Capital: ${portfolio_mgr.initial_capital:,.2f}")
        logger.info(f"   Max positions: {portfolio_mgr.constraints.max_positions}")
        logger.info(f"   Min edge threshold: {portfolio_mgr.constraints.minimum_edge_threshold}")
        
        # Generate synthetic market data for multiple assets
        logger.info("\n📈 Generating multi-asset market data...")
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        market_data_dict = {}
        
        # Create different market behaviors for each asset
        behaviors = {
            'BTC/USDT': {'base_drift': 0.0005, 'volatility': 0.03, 'trend': 'strong'},
            'ETH/USDT': {'base_drift': 0.0003, 'volatility': 0.04, 'trend': 'moderate'},
            'SOL/USDT': {'base_drift': 0.0001, 'volatility': 0.05, 'trend': 'volatile'},
            'ADA/USDT': {'base_drift': 0.0002, 'volatility': 0.025, 'trend': 'stable'},
            'DOT/USDT': {'base_drift': 0.0004, 'volatility': 0.035, 'trend': 'mixed'}
        }
        
        for symbol in symbols:
            behavior = behaviors[symbol]
            prices = []
            current_price = np.random.uniform(1000, 20000)  # Random starting price
            
            for i in range(len(dates)):
                # Different dynamics based on behavior
                if behavior['trend'] == 'strong':
                    drift = behavior['base_drift'] + np.random.normal(0, 0.002)
                elif behavior['trend'] == 'volatile':
                    drift = np.random.normal(0, behavior['base_drift'])
                else:
                    drift = behavior['base_drift'] + np.random.normal(0, 0.001)
                
                price_change = drift + np.random.normal(0, behavior['volatility'])
                current_price = current_price * (1 + price_change)
                current_price = max(current_price, 10)  # Floor price
                prices.append(current_price)
            
            # Create market data DataFrame
            market_data = pd.DataFrame({
                'timestamp': dates,
                'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': [np.random.uniform(1000, 10000) * p for p in prices]
            })
            
            market_data_dict[symbol] = market_data
        
        logger.info(f"✅ Generated data for {len(market_data_dict)} assets")
        for symbol, data in market_data_dict.items():
            logger.info(f"   {symbol}: {len(data)} days, price ${data['close'].iloc[-1]:.2f}")
        
        # Test 1: Regime Detection
        logger.info(f"\n{'='*50}")
        logger.info("🎭 Test 1: Market Regime Detection")
        logger.info(f"{'='*50}")
        
        # Detect regime using BTC data as market proxy
        btc_data = market_data_dict['BTC/USDT'].tail(100)
        regime_result = regime_detector.detect_current_regime(btc_data[['close']])
        
        regime_context = {
            'name': regime_result.name if regime_result else 'STABLE',
            'probability': regime_result.probability if regime_result else 0.5
        } if regime_result else None
        
        if regime_context:
            logger.info(f"   Detected regime: {regime_context['name']}")
            logger.info(f"   Confidence: {regime_context['probability']:.3f}")
        
        # Test 2: Portfolio Construction
        logger.info(f"\n{'='*50}")
        logger.info("🏗️ Test 2: Portfolio Construction")
        logger.info(f"{'='*50}")
        
        # Construct optimal portfolio
        allocations = portfolio_mgr.construct_optimal_portfolio(
            market_data_dict=market_data_dict,
            regime_context=regime_context
        )
        
        logger.info(f"   Generated {len(allocations)} portfolio allocations")
        
        if allocations:
            logger.info("   Allocation details:")
            total_weight = 0
            for alloc in allocations:
                total_weight += alloc.weight
                logger.info(f"     {alloc.symbol}: {alloc.weight*100:.1f}% "
                           f"(Edge: {alloc.edge_probability:.3f}, "
                           f"Price: ${alloc.entry_price:.2f})")
            
            logger.info(f"   Total allocated: {total_weight*100:.1f}%")
        
        # Test 3: Portfolio Summary
        logger.info(f"\n{'='*50}")
        logger.info("📋 Test 3: Portfolio Summary")
        logger.info(f"{'='*50}")
        
        summary = portfolio_mgr.get_portfolio_summary()
        logger.info(f"   Status: {summary['status']}")
        logger.info(f"   Positions: {summary['positions']}")
        logger.info(f"   Total value: ${summary['total_value']:,.2f}")
        logger.info(f"   Cash available: ${summary['cash']:,.2f}")
        logger.info(f"   Leverage: {summary['leverage']*100:.1f}%")
        logger.info(f"   Allocation decisions: {summary['allocation_history']}")
        
        # Test 4: Risk Integration
        logger.info(f"\n{'='*50}")
        logger.info("🛡️ Test 4: Risk Engine Integration")
        logger.info(f"{'='*50}")
        
        if allocations:
            # Update risk engine with portfolio state
            portfolio_positions = {alloc.symbol: alloc.position_size for alloc in allocations}
            portfolio_value = sum(alloc.position_size * alloc.entry_price for alloc in allocations)
            
            risk_update = risk_engine.update_portfolio_state(
                portfolio_value=portfolio_value,
                positions=portfolio_positions
            )
            
            risk_report = risk_engine.get_risk_report()
            metrics = risk_report['portfolio_metrics']
            
            logger.info(f"   Portfolio value: ${metrics['current_value']:,.2f}")
            logger.info(f"   Current drawdown: {metrics['current_drawdown']*100:.2f}%")
            logger.info(f"   Portfolio volatility: {metrics['volatility']*100:.2f}%")
            logger.info(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"   Risk status: {risk_report['status']}")
            
            if not risk_update:
                logger.warning("   ⚠️ Risk limits exceeded - portfolio needs adjustment")
        
        # Test 5: Rebalancing
        logger.info(f"\n{'='*50}")
        logger.info("🔄 Test 5: Portfolio Rebalancing")
        logger.info(f"{'='*50}")
        
        # Simulate market changes
        logger.info("   Simulating market movements...")
        
        # Modify some prices to create rebalancing need
        modified_data = {}
        for symbol, data in market_data_dict.items():
            modified_data[symbol] = data.copy()
            # Add some price movement
            recent_prices = modified_data[symbol]['close'].tail(10)
            modified_prices = recent_prices * (1 + np.random.normal(0, 0.02, len(recent_prices)))
            modified_data[symbol].loc[modified_data[symbol].tail(10).index, 'close'] = modified_prices
        
        # Rebalance portfolio
        rebalanced_allocations = portfolio_mgr.rebalance_portfolio(
            market_data_dict=modified_data,
            regime_context=regime_context
        )
        
        logger.info(f"   Rebalanced to {len(rebalanced_allocations)} positions")
        
        if rebalanced_allocations and allocations:
            # Compare before/after
            before_symbols = {alloc.symbol for alloc in allocations}
            after_symbols = {alloc.symbol for alloc in rebalanced_allocations}
            
            changes = before_symbols.symmetric_difference(after_symbols)
            if changes:
                logger.info(f"   Position changes: {changes}")
            else:
                logger.info("   No position changes required")
        
        # Test 6: Constraint Analysis
        logger.info(f"\n{'='*50}")
        logger.info("⚙️ Test 6: Constraint Analysis")
        logger.info(f"{'='*50}")
        
        logger.info("   Portfolio constraints:")
        logger.info(f"     Max positions: {portfolio_mgr.constraints.max_positions}")
        logger.info(f"     Min edge threshold: {portfolio_mgr.constraints.minimum_edge_threshold}")
        logger.info(f"     Max portfolio volatility: {portfolio_mgr.constraints.max_portfolio_volatility*100:.1f}%")
        logger.info(f"     Max sector exposure: {portfolio_mgr.constraints.max_sector_exposure*100:.1f}%")
        
        # Test different edge thresholds
        original_threshold = portfolio_mgr.constraints.minimum_edge_threshold
        test_thresholds = [0.3, 0.5, 0.7, 0.9]
        
        logger.info("   Testing different edge thresholds:")
        for threshold in test_thresholds:
            portfolio_mgr.constraints.minimum_edge_threshold = threshold
            test_allocations = portfolio_mgr.construct_optimal_portfolio(
                market_data_dict=market_data_dict,
                regime_context=regime_context
            )
            logger.info(f"     Threshold {threshold}: {len(test_allocations)} positions")
        
        # Restore original threshold
        portfolio_mgr.constraints.minimum_edge_threshold = original_threshold
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("🎯 PORTFOLIO CONSTRUCTION DEMO COMPLETED")
        logger.info(f"{'='*60}")
        logger.info("✅ Key achievements:")
        logger.info("   • Implemented professional portfolio optimization")
        logger.info("   • Integrated edge classification with risk management")
        logger.info("   • Built regime-aware allocation logic")
        logger.info("   • Created automated rebalancing capabilities")
        logger.info("   • Added comprehensive constraint handling")
        logger.info("   • Demonstrated full portfolio lifecycle")
        
        logger.info(f"\n🚀 Chloe 0.4 Progress:")
        logger.info("   ✅ Phase 1: Market Intelligence Layer (70% complete)")
        logger.info("   ✅ Phase 2: Risk Engine Core Enhancement (complete)")
        logger.info("   ✅ Phase 3: Edge Classification Model (complete)")
        logger.info("   ✅ Phase 4: Portfolio Construction Logic (now complete)")
        logger.info("   ⬜ Phase 5: Simulation Lab")
        
        # Final portfolio status
        final_summary = portfolio_mgr.get_portfolio_summary()
        logger.info(f"\n📊 Final Portfolio Status:")
        logger.info(f"   Active positions: {final_summary['positions']}")
        logger.info(f"   Portfolio value: ${final_summary['total_value']:,.2f}")
        logger.info(f"   Cash reserves: ${final_summary['cash']:,.2f}")
        logger.info(f"   Utilization: {(1 - final_summary['cash']/portfolio_mgr.initial_capital)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    asyncio.run(demo_portfolio_construction())