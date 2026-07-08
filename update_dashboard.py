"""
自動化儀表板更新系統
將流動性監測結果實時更新到 TradingView 儀表板

用法：
  python update_dashboard.py              # 執行一次更新
  python update_dashboard.py --daemon     # 持續運行，每 5 分鐘更新一次
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import sys
import os

class DashboardUpdater:
    """儀表板更新系統"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.csv_path = self.project_root / 'liquidity_monitor.csv'
        self.decisions_csv = self.project_root / 'momentum_liquidity_decisions.csv'
        self.state_file = self.project_root / 'dashboard_state.json'

    REQUIRED_METRICS = ['net_liquidity', 'net_liquidity_pctl', 'sofr', 'stress_spread', 'hy_spread']

    def get_latest_monitoring_data(self):
        """從 CSV 獲取最新監測數據；缺少必要欄位視同無數據"""
        try:
            df = pd.read_csv(self.csv_path)
            if len(df) == 0:
                return None

            missing = [c for c in self.REQUIRED_METRICS if c not in df.columns]
            if missing:
                print(f"❌ CSV 缺少必要欄位: {missing}（請先執行 liquidity_monitor.py）")
                return None

            latest = df.iloc[-1].to_dict()

            # 從前一筆計算變化量
            if len(df) >= 2:
                prev = df.iloc[-2]
                latest['net_liquidity_change'] = f"{latest['net_liquidity'] - prev['net_liquidity']:+.2f}"
                latest['sofr_change'] = f"{latest['sofr'] - prev['sofr']:+.2f}"
                latest['stress_spread_change'] = f"{latest['stress_spread'] - prev['stress_spread']:+.0f}"
                latest['hy_spread_change'] = f"{latest['hy_spread'] - prev['hy_spread']:+.0f}"

            return latest

        except FileNotFoundError:
            print(f"⚠️  CSV 檔案不存在: {self.csv_path}")
            return None
        except Exception as e:
            print(f"❌ 讀取 CSV 失敗: {e}")
            return None

    def get_latest_decision(self):
        """從決策 CSV 獲取最新倉位建議"""
        try:
            df = pd.read_csv(self.decisions_csv)
            if len(df) == 0:
                return None

            latest = df.iloc[-1].to_dict()
            return latest

        except FileNotFoundError:
            # 決策 CSV 可能不存在
            return None
        except Exception as e:
            print(f"❌ 讀取決策 CSV 失敗: {e}")
            return None

    def classify_state(self, net_liquidity_pctl, sofr, stress_spread, hy_spread):
        """根據指標分類流動性狀態（淨流動性用一年分位數）"""
        # 淨流動性權重 40%
        if net_liquidity_pctl > 70:
            liq_state = 'abundant'
            liq_score = 100
        elif net_liquidity_pctl > 40:
            liq_state = 'normal'
            liq_score = 70
        elif net_liquidity_pctl > 20:
            liq_state = 'tight'
            liq_score = 40
        else:
            liq_state = 'critical'
            liq_score = 10

        # SOFR 權重 30%
        if sofr < 4.5:
            sofr_score = 100
        elif sofr < 5.5:
            sofr_score = 60
        elif sofr < 6.0:
            sofr_score = 30
        else:
            sofr_score = 10

        # 壓力指標權重 20%
        if stress_spread < 20:
            stress_score = 100
        elif stress_spread < 50:
            stress_score = 70
        else:
            stress_score = 30

        # HY Spread 權重 10%
        if hy_spread < 350:
            hy_score = 100
        elif hy_spread < 450:
            hy_score = 60
        else:
            hy_score = 30

        # 綜合評分
        total_score = (liq_score * 0.4 + sofr_score * 0.3 +
                      stress_score * 0.2 + hy_score * 0.1)

        if total_score >= 80:
            state = 'abundant'
        elif total_score >= 60:
            state = 'normal'
        elif total_score >= 40:
            state = 'tight'
        else:
            state = 'critical'

        return state, total_score, liq_state

    def recommend_allocation(self, state):
        """根據流動性狀態推薦倉位"""
        allocations = {
            'abundant': {'alpha': 70, 'beta': 30, 'risk': 'low'},
            'normal': {'alpha': 60, 'beta': 40, 'risk': 'low'},
            'tight': {'alpha': 40, 'beta': 60, 'risk': 'high'},
            'critical': {'alpha': 20, 'beta': 80, 'risk': 'critical'}
        }
        return allocations.get(state, allocations['normal'])

    def generate_dashboard_json(self):
        """生成儀表板 JSON 數據"""
        # 獲取監測數據
        monitoring = self.get_latest_monitoring_data()
        decision = self.get_latest_decision()

        if monitoring is None:
            # 不偽造數據：明確標示等待狀態，前端會顯示提示
            print("❌ 無有效監測數據，輸出等待狀態（不使用假數據）")
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'waiting',
                'message': '⏳ 等待流動性監測數據，執行: python liquidity_monitor.py',
                'net_liquidity': {'value': '—', 'unit': 'T', 'state': 'normal', 'change': '—'},
                'sofr': {'value': '—', 'unit': '%', 'state': 'normal', 'change': '—'},
                'stress_spread': {'value': '—', 'unit': 'bps', 'state': 'normal', 'change': '—'},
                'hy_spread': {'value': '—', 'unit': 'bps', 'state': 'normal', 'change': '—'},
                'liquidity_state': {'overall': 'normal', 'score': 0, 'label': '⏳ 等待數據'},
                'momentum_allocation': {'alpha': '—', 'beta': '—', 'risk': 'unknown',
                                        'recommendation': '⏳ 等待數據'},
                'decision': None,
            }

        # 分類狀態
        net_liq = float(monitoring['net_liquidity'])
        net_liq_pctl = float(monitoring['net_liquidity_pctl'])
        sofr = float(monitoring['sofr'])
        stress = float(monitoring['stress_spread'])
        hy = float(monitoring['hy_spread'])

        state, score, liq_state = self.classify_state(net_liq_pctl, sofr, stress, hy)
        allocation = self.recommend_allocation(state)

        # 建構 JSON
        dashboard_data = {
            'timestamp': monitoring.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'net_liquidity': {
                'value': round(net_liq, 2),
                'unit': 'T',
                'state': liq_state,
                'percentile': round(net_liq_pctl, 0),
                'change': monitoring.get('net_liquidity_change', '+0.00'),
                'threshold': {
                    'abundant': '1年分位數 > 70%',
                    'normal': '40-70%',
                    'tight': '20-40%',
                    'critical': '< 20%'
                }
            },
            'sofr': {
                'value': round(sofr, 2),
                'unit': '%',
                'state': 'abundant' if sofr < 4.5 else 'normal' if sofr < 5.5 else 'tight' if sofr < 6.0 else 'critical',
                'change': monitoring.get('sofr_change', '-0.00'),
                'threshold': {
                    'abundant': '< 4.5%',
                    'normal': '4.5-5.5%',
                    'tight': '5.5-6.0%',
                    'critical': '> 6.0%'
                }
            },
            'stress_spread': {
                'value': round(stress, 0),
                'unit': 'bps',
                'state': 'normal' if stress < 20 else 'tight' if stress < 50 else 'critical',
                'change': monitoring.get('stress_spread_change', '+0'),
                'threshold': {
                    'normal': '< 20 bps',
                    'tight': '20-50 bps',
                    'critical': '> 50 bps'
                }
            },
            'hy_spread': {
                'value': round(hy, 0),
                'unit': 'bps',
                'state': 'abundant' if hy < 350 else 'normal' if hy < 450 else 'tight',
                'change': monitoring.get('hy_spread_change', '+0'),
                'threshold': {
                    'abundant': '< 350 bps',
                    'normal': '350-450 bps',
                    'tight': '> 450 bps'
                }
            },
            'liquidity_state': {
                'overall': state,
                'score': round(score, 0),
                'label': {
                    'abundant': '✅ 充足',
                    'normal': '✓ 正常',
                    'tight': '⚠️  緊張',
                    'critical': '🚨 危機'
                }[state]
            },
            'momentum_allocation': {
                'alpha': allocation['alpha'],
                'beta': allocation['beta'],
                'risk': allocation['risk'],
                'recommendation': self.get_action_text(state, allocation)
            },
            'decision': decision if decision else None
        }

        return dashboard_data

    def get_action_text(self, state, allocation):
        """根據狀態生成行動建議"""
        actions = {
            'abundant': f"✅ 維持或增加倉位 (Alpha {allocation['alpha']}%)",
            'normal': f"✓ 正常交易 (Alpha {allocation['alpha']}%)",
            'tight': f"⚠️  降低倉位 (Alpha {allocation['alpha']}%)",
            'critical': f"🚨 現金為王 (Alpha {allocation['alpha']}%)"
        }
        return actions.get(state, '等待訊號')

    def save_state(self, data):
        """保存狀態到 JSON 檔案"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 狀態已保存: {self.state_file}")
        except Exception as e:
            print(f"❌ 保存狀態失敗: {e}")

    def generate_html_update(self, data):
        """生成 HTML 更新代碼（可插入儀表板）"""
        html = f"""
        <!-- 自動生成於 {data['timestamp']} -->
        <script>
        const monitoringData = {json.dumps(data, ensure_ascii=False)};

        // 更新監測面板
        function updateDashboard() {{
            // 更新狀態面板
            const statusPanel = document.getElementById('statusPanel');
            if (statusPanel) {{
                statusPanel.innerHTML = `
                    <div class="status-item ${{monitoringData.liquidity_state.overall}}">
                        <span class="status-label">💰 淨流動性</span>
                        <span class="status-value">${{monitoringData.net_liquidity.value}}${{monitoringData.net_liquidity.unit}}</span>
                    </div>
                    <div class="status-item ${{monitoringData.sofr.state}}">
                        <span class="status-label">📈 SOFR</span>
                        <span class="status-value">${{monitoringData.sofr.value}}${{monitoringData.sofr.unit}}</span>
                    </div>
                    <div class="status-item ${{monitoringData.momentum_allocation.risk}}">
                        <span class="status-label">🎯 Alpha</span>
                        <span class="status-value">${{monitoringData.momentum_allocation.alpha}}%</span>
                    </div>
                    <div class="status-item" style="border-left-color: #2962ff; color: #2962ff;">
                        <span class="status-label">⏰ 更新時間</span>
                        <span class="status-value">{data['timestamp']}</span>
                    </div>
                `;
            }}
        }}

        // 頁面加載完成後執行
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', updateDashboard);
        }} else {{
            updateDashboard();
        }}
        </script>
        """
        return html

    def run_once(self):
        """執行一次更新"""
        print("\n" + "="*80)
        print("🔄 儀表板更新系統")
        print("="*80)

        # 生成數據
        data = self.generate_dashboard_json()

        # 保存 JSON
        self.save_state(data)

        # 顯示摘要
        print(f"\n⏰ 時間: {data['timestamp']}")
        print(f"💰 淨流動性: ${data['net_liquidity']['value']}{data['net_liquidity']['unit']} ({data['liquidity_state']['label']})")
        print(f"📈 SOFR: {data['sofr']['value']}{data['sofr']['unit']}")
        print(f"⚠️  壓力指標: {data['stress_spread']['value']}{data['stress_spread']['unit']}")
        print(f"📊 HY Spread: {data['hy_spread']['value']}{data['hy_spread']['unit']}")
        print(f"\n🎯 Momentum 建議:")
        print(f"   Alpha: {data['momentum_allocation']['alpha']}%")
        print(f"   Beta: {data['momentum_allocation']['beta']}%")
        print(f"   風險: {data['momentum_allocation']['risk'].upper()}")
        print(f"   行動: {data['momentum_allocation']['recommendation']}")

        print("\n" + "="*80)
        return data

    def run_daemon(self, interval=300):
        """持續運行，定期更新（間隔秒數，預設 300 秒 = 5 分鐘）"""
        print(f"\n🔄 儀表板守護程序已啟動")
        print(f"📋 更新間隔: {interval} 秒 ({interval/60:.0f} 分鐘)")
        print(f"⏹️  按 Ctrl+C 停止\n")

        try:
            while True:
                self.run_once()
                print(f"⏳ 下次更新時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + interval))}")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n✅ 守護程序已停止")


def main():
    updater = DashboardUpdater()

    if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
        # 持續運行模式
        updater.run_daemon()
    else:
        # 單次執行模式
        updater.run_once()


if __name__ == '__main__':
    main()
