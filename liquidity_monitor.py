"""
流動性監測與 Momentum 策略整合系統
Liquidity Monitor + Momentum Strategy Integration

監測指標（來自 tradingview-dashboard/liquidity.html）：
1. Net Liquidity (WALCL/1000 - WTREGEN - RRPONTSYD)
2. Reverse Repo (RRPONTSYD)
3. TGA (WTREGEN)
4. SOFR Rate (SOFR)
5. Liquidity Stress Spread (SOFR - IORB)
6. USD/JPY Carry Trade
7. US 10Y/30Y Yield
8. HY/Corp Spread
9. BTC (Risk Sentiment)

目標：
- 實時監測流動性狀態
- 根據流動性調整 Momentum 倉位
- 預警流動性危機
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings("ignore")

class LiquidityMonitor:
    """流動性監測系統"""

    # FRED API Key (需要自行設定)
    FRED_API_KEY = ''

    # 流動性指標定義
    INDICATORS = {
        'WALCL': 'FED Total Assets',
        'WTREGEN': 'TGA (Treasury General Account)',
        'RRPONTSYD': 'RRP (Reverse Repo)',
        'SOFR': 'SOFR Overnight Rate',
        'IORB': 'Interest On Reserve Balances',
        'DGS10': 'US 10Y Yield',
        'DGS30': 'US 30Y Yield',
        'BAMLH0A0HYM2': 'HY Spread (Junk)',
        'BAMLC0A0CM': 'Corp Spread (Inv Grade)',
    }

    # 閾值定義
    THRESHOLDS = {
        'net_liquidity_critical': 1.5,      # < 1.5T = 危機
        'net_liquidity_tight': 2.0,         # 1.5-2.0T = 緊張
        'sofr_rate_high': 5.5,              # > 5.5% = 流動性緊張
        'stress_spread_high': 50,           # > 50 bps = 壓力信號
        'hy_spread_wide': 450,              # > 450 bps = 風險厭惡
        'corp_spread_wide': 150,            # > 150 bps = 風險厭惡
    }

    def __init__(self, api_key=None):
        """初始化監測系統"""
        if api_key:
            self.FRED_API_KEY = api_key
        self.current_state = {}
        self.historical_data = {}

    def fetch_fred_data(self, series_id, days=30):
        """從 FRED 獲取指標數據"""
        if not self.FRED_API_KEY:
            print(f"⚠️  FRED_API_KEY 未設定，無法獲取 {series_id}")
            return None

        try:
            url = f"https://api.stlouisfed.org/fred/series/data"
            params = {
                'series_id': series_id,
                'api_key': self.FRED_API_KEY,
                'file_type': 'json',
                'observation_start': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'observation_end': datetime.now().strftime('%Y-%m-%d'),
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df.dropna(subset=['value']).sort_values('date')

        except Exception as e:
            print(f"❌ 無法獲取 {series_id}: {e}")

        return None

    def calculate_net_liquidity(self):
        """
        計算淨流動性 = (WALCL/1000 - WTREGEN - RRPONTSYD)
        即: (聯準會資產 - TGA - RRP) / 1000
        """
        try:
            walcl = self.fetch_fred_data('WALCL', days=90)
            wtregen = self.fetch_fred_data('WTREGEN', days=90)
            rrpontsyd = self.fetch_fred_data('RRPONTSYD', days=90)

            if walcl is not None and wtregen is not None and rrpontsyd is not None:
                # 合併數據
                df = walcl[['date', 'value']].rename(columns={'value': 'walcl'})
                df['wtregen'] = wtregen.set_index('date')['value']
                df['rrpontsyd'] = rrpontsyd.set_index('date')['value']

                df['net_liquidity'] = (df['walcl'] / 1000 - df['wtregen'] - df['rrpontsyd'])

                self.historical_data['net_liquidity'] = df
                return df

        except Exception as e:
            print(f"❌ 無法計算淨流動性: {e}")

        return None

    def calculate_stress_spread(self):
        """
        計算流動性壓力指標 = SOFR - IORB
        代表超額準備金成本，越高 = 流動性越緊張
        """
        try:
            sofr = self.fetch_fred_data('SOFR', days=90)
            iorb = self.fetch_fred_data('IORB', days=90)

            if sofr is not None and iorb is not None:
                df = sofr[['date', 'value']].rename(columns={'value': 'sofr'})
                df['iorb'] = iorb.set_index('date')['value']
                df['stress_spread'] = (df['sofr'] - df['iorb']) * 10000  # 轉換為 bps

                self.historical_data['stress_spread'] = df
                return df

        except Exception as e:
            print(f"❌ 無法計算壓力指標: {e}")

        return None

    def assess_liquidity_state(self):
        """
        評估流動性狀態
        返回: {
            'state': 'abundant' | 'normal' | 'tight' | 'critical',
            'score': 0-100 (100 = 最充足),
            'signals': [相關信號],
            'risk_level': 'low' | 'medium' | 'high' | 'critical'
        }
        """
        signals = []
        score = 100

        # 1. 淨流動性評估
        net_liq_df = self.calculate_net_liquidity()
        if net_liq_df is not None and len(net_liq_df) > 0:
            latest_net_liq = net_liq_df['net_liquidity'].iloc[-1]
            self.current_state['net_liquidity'] = latest_net_liq

            if latest_net_liq < self.THRESHOLDS['net_liquidity_critical']:
                signals.append(f"🚨 淨流動性危機: ${latest_net_liq:.2f}T < ${self.THRESHOLDS['net_liquidity_critical']}T")
                score -= 50
            elif latest_net_liq < self.THRESHOLDS['net_liquidity_tight']:
                signals.append(f"⚠️  淨流動性緊張: ${latest_net_liq:.2f}T < ${self.THRESHOLDS['net_liquidity_tight']}T")
                score -= 25
            else:
                signals.append(f"✅ 淨流動性充足: ${latest_net_liq:.2f}T")

        # 2. SOFR 利率評估
        try:
            sofr_df = self.fetch_fred_data('SOFR', days=30)
            if sofr_df is not None and len(sofr_df) > 0:
                latest_sofr = sofr_df['value'].iloc[-1]
                self.current_state['sofr'] = latest_sofr

                if latest_sofr > self.THRESHOLDS['sofr_rate_high']:
                    signals.append(f"🚨 SOFR 高企: {latest_sofr:.2f}% > {self.THRESHOLDS['sofr_rate_high']}%")
                    score -= 20
                else:
                    signals.append(f"✅ SOFR 正常: {latest_sofr:.2f}%")
        except:
            pass

        # 3. 壓力指標評估
        stress_df = self.calculate_stress_spread()
        if stress_df is not None and len(stress_df) > 0:
            latest_stress = stress_df['stress_spread'].iloc[-1]
            self.current_state['stress_spread'] = latest_stress

            if latest_stress > self.THRESHOLDS['stress_spread_high']:
                signals.append(f"⚠️  壓力指標升高: {latest_stress:.0f} bps > {self.THRESHOLDS['stress_spread_high']} bps")
                score -= 15
            else:
                signals.append(f"✅ 壓力指標正常: {latest_stress:.0f} bps")

        # 4. 信用利差評估
        try:
            hy_df = self.fetch_fred_data('BAMLH0A0HYM2', days=30)
            corp_df = self.fetch_fred_data('BAMLC0A0CM', days=30)

            if hy_df is not None and len(hy_df) > 0:
                latest_hy = hy_df['value'].iloc[-1]
                self.current_state['hy_spread'] = latest_hy

                if latest_hy > self.THRESHOLDS['hy_spread_wide']:
                    signals.append(f"⚠️  高收益利差擴大: {latest_hy:.0f} bps > {self.THRESHOLDS['hy_spread_wide']} bps")
                    score -= 15

            if corp_df is not None and len(corp_df) > 0:
                latest_corp = corp_df['value'].iloc[-1]
                self.current_state['corp_spread'] = latest_corp
        except:
            pass

        # 決定流動性狀態
        score = max(0, min(100, score))

        if score >= 80:
            state = 'abundant'
            risk_level = 'low'
        elif score >= 60:
            state = 'normal'
            risk_level = 'low'
        elif score >= 40:
            state = 'tight'
            risk_level = 'medium'
        elif score >= 20:
            state = 'very_tight'
            risk_level = 'high'
        else:
            state = 'critical'
            risk_level = 'critical'

        return {
            'state': state,
            'score': score,
            'signals': signals,
            'risk_level': risk_level,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_momentum_adjustment(self, liquidity_state):
        """
        根據流動性狀態調整 Momentum 倉位

        返回: {
            'alpha_allocation': 0-100 (推薦的 Alpha 倉位比例),
            'beta_allocation': 0-100 (推薦的 Beta 倉位比例),
            'risk_warning': str,
            'action': str
        }
        """
        state = liquidity_state['state']
        score = liquidity_state['score']

        adjustments = {
            'abundant': {
                'alpha_allocation': 75,
                'beta_allocation': 25,
                'risk_warning': '無',
                'action': '✅ 增加 Momentum 倉位，聚焦成長股'
            },
            'normal': {
                'alpha_allocation': 60,
                'beta_allocation': 40,
                'risk_warning': '無',
                'action': '✅ 維持中性配置'
            },
            'tight': {
                'alpha_allocation': 40,
                'beta_allocation': 60,
                'risk_warning': '流動性開始緊張',
                'action': '⚠️  降低 Momentum 倉位，增加防守'
            },
            'very_tight': {
                'alpha_allocation': 20,
                'beta_allocation': 80,
                'risk_warning': '流動性嚴重緊張',
                'action': '🚨 激進降低 Alpha，轉向防守'
            },
            'critical': {
                'alpha_allocation': 5,
                'beta_allocation': 95,
                'risk_warning': '流動性危機',
                'action': '🚨 現金為王，停止交易'
            }
        }

        return adjustments.get(state, adjustments['normal'])

    def generate_report(self):
        """生成流動性監測報告"""
        print("\n" + "="*80)
        print("📊 流動性監測報告")
        print("="*80)

        # 評估流動性狀態
        liquidity_state = self.assess_liquidity_state()

        print(f"\n時間: {liquidity_state['timestamp']}")
        print(f"流動性狀態: {liquidity_state['state'].upper()}")
        print(f"流動性分數: {liquidity_state['score']}/100")
        print(f"風險等級: {liquidity_state['risk_level'].upper()}")

        print("\n📋 監測訊號:")
        for signal in liquidity_state['signals']:
            print(f"  {signal}")

        # 獲取 Momentum 調整建議
        adjustment = self.get_momentum_adjustment(liquidity_state)

        print("\n💼 Momentum 策略調整:")
        print(f"  Alpha 倉位: {adjustment['alpha_allocation']}%")
        print(f"  Beta 倉位: {adjustment['beta_allocation']}%")
        print(f"  風險警告: {adjustment['risk_warning']}")
        print(f"  建議行動: {adjustment['action']}")

        # 當前指標值
        print("\n📈 當前指標值:")
        for key, value in self.current_state.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "="*80)

        return liquidity_state, adjustment

    def export_to_csv(self, filename='liquidity_monitor.csv'):
        """匯出監測數據到 CSV"""
        try:
            export_data = {
                'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                **{f"{k}": [v] if isinstance(v, (int, float)) else [v]
                   for k, v in self.current_state.items()}
            }

            df = pd.DataFrame(export_data)
            df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
            print(f"✅ 數據已匯出到: {filename}")
        except Exception as e:
            print(f"❌ 匯出失敗: {e}")


def main():
    """主函數"""
    print("="*80)
    print("🔍 流動性監測系統 (Liquidity Monitor)")
    print("="*80)

    # 初始化監測系統
    monitor = LiquidityMonitor(api_key='')  # 需要設定 FRED API Key

    # 如果沒有 API Key，使用離線模式
    if not monitor.FRED_API_KEY:
        print("\n⚠️  未設定 FRED_API_KEY，以下數據為示例...")
        print("\n獲取 FRED API Key 步驟:")
        print("  1. 訪問 https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  2. 註冊帳戶獲取免費 API Key")
        print("  3. 設定環境變數: export FRED_API_KEY='your_key'")
        print("  或修改本腳本 main() 中的 api_key 參數")

    # 生成報告
    liquidity_state, adjustment = monitor.generate_report()

    # 匯出數據
    monitor.export_to_csv('liquidity_monitor.csv')

    return monitor, liquidity_state, adjustment


if __name__ == '__main__':
    monitor, liquidity_state, adjustment = main()
