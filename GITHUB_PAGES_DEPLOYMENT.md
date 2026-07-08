# TradingView Dashboard 部署到 GitHub Pages

---

## 📋 檔案清單

### 核心檔案

| 檔案名 | 用途 | 狀態 |
|------|------|------|
| `liquidity.html` | 原始 TradingView 儀表板 (v10) | ✅ 已有 |
| `liquidity_dashboard_enhanced.html` | 增強版儀表板 (v11 + 監測面板) | ✅ 新增 |
| `putcall.html` | Put/Call 比率儀表板 | ✅ 已有 |
| `USD.html` | 美元指數儀表板 | ✅ 已有 |

### 相關檔案

- `liquidity_correlation.py` - 流動性相關性分析
- `putcall_research_notes.md` - Put/Call 研究筆記
- `工作日誌.md` - 工作日誌

---

## 🚀 部署步驟

### Step 1️⃣ 確認 GitHub 倉庫狀態

```bash
cd ~/GITHUB/tradingview-dashboard

# 檢查遠程倉庫
git remote -v

# 應該顯示類似:
# origin  https://github.com/t0mst0ne/t0mst0ne.github.io.git (fetch)
# origin  https://github.com/t0mst0ne/t0mst0ne.github.io.git (push)
```

### Step 2️⃣ 檢查 GitHub Pages 配置

#### 方法 A: 在 docs/ 文件夾中（推薦）

```bash
# 如果使用 docs/ 文件夾
ls -la docs/

# 應該看到:
# - index.html (主頁)
# - liquidity.html
# - liquidity_dashboard_enhanced.html (新增)
# - etc.
```

#### 方法 B: 在根目錄（當前方式）

```bash
# 如果用根目錄，檔案就在 ~/GITHUB/tradingview-dashboard/ 中
ls -la liquidity_dashboard_enhanced.html

# 應該顯示:
# -rw-r--r--  1 tomstone  staff  24K  7  8 12:17 liquidity_dashboard_enhanced.html
```

### Step 3️⃣ 提交變更

```bash
cd ~/GITHUB/tradingview-dashboard

# 查看未追蹤的檔案
git status

# 應該看到:
# Untracked files:
#   liquidity_dashboard_enhanced.html

# 添加新檔案
git add liquidity_dashboard_enhanced.html

# 提交
git commit -m "feat: Add enhanced liquidity dashboard with real-time monitoring panel

- Adds liquidity_dashboard_enhanced.html (v11)
- 12 TradingView charts (3x4 grid)
- Real-time monitoring panel on the right
- Status indicator bar at the top
- Automatic updates every 5 minutes
- Integration with liquidity_monitor.py and momentum_liquidity_integration.py"

# 推送到 GitHub
git push origin main
# 或
git push origin master
```

### Step 4️⃣ 驗證 GitHub Pages 部署

```bash
# 檢查 GitHub 倉庫設定
# 訪問: https://github.com/t0mst0ne/t0mst0ne.github.io/settings/pages

# 應該看到:
# ✓ GitHub Pages is currently enabled
# ✓ Publishing from: main branch
# 或: Publishing from: main branch / root
# 或: Publishing from: main branch / docs folder
```

---

## 🌐 訪問 URL

### 原始儀表板（v10）
```
https://t0mst0ne.github.io/tradingview-dashboard/liquidity.html
```

### 增強版儀表板（v11）
```
https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html
```

### 其他儀表板
```
https://t0mst0ne.github.io/tradingview-dashboard/putcall.html
https://t0mst0ne.github.io/tradingview-dashboard/USD.html
```

---

## 🔄 後續自動化

### 使用 GitHub Actions 自動更新

建立 `.github/workflows/update-dashboard.yml`：

```yaml
name: Update Dashboard Data

on:
  schedule:
    # 每天 9:30 AM UTC+8 (1:30 AM UTC)
    - cron: '30 1 * * *'
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install pandas requests
      
      - name: Run liquidity monitor
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
        run: |
          python ../Momentum/Backtest/FED_Policy_Momentum_Analysis/liquidity_monitor.py
          python ../Momentum/Backtest/FED_Policy_Momentum_Analysis/update_dashboard.py
      
      - name: Commit and push
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add dashboard_state.json liquidity_monitor.csv
          git commit -m "chore: Update monitoring data" || true
          git push
```

### 本地自動推送指令碼

```bash
#!/bin/bash
# update_and_push.sh

cd ~/GITHUB/Momentum
python Backtest/FED_Policy_Momentum_Analysis/liquidity_monitor.py
python Backtest/FED_Policy_Momentum_Analysis/update_dashboard.py

cd ~/GITHUB/tradingview-dashboard
git add dashboard_state.json liquidity_monitor.csv liquidity_dashboard_enhanced.html
git commit -m "chore: Auto-update monitoring data" || true
git push origin main

echo "✅ Dashboard updated and pushed to GitHub"
```

---

## 📊 儀表板訪問統計

使用 GitHub 自帶的分析工具查看訪問統計：

```
Settings → Pages → View traffic
```

---

## 🔐 安全性設定

### 設定 GitHub Pages 為 HTTPS Only

```
Settings → Pages → HTTPS enforcement: ON
```

### 配置自訂域名（可選）

如果你有自訂域名：

```
Settings → Pages → Custom domain: your-domain.com
```

---

## 📱 分享儀表板

### 分享完整 URL

```
💾 增強版儀表板（推薦）:
https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html

📊 原始版本:
https://t0mst0ne.github.io/tradingview-dashboard/liquidity.html
```

### 嵌入到其他網站（iframe）

```html
<!-- 嵌入增強版儀表板 -->
<iframe 
  src="https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html"
  width="100%"
  height="900px"
  frameborder="0"
  allowfullscreen>
</iframe>
```

---

## ⚠️ 注意事項

### 1. TradingView Widget 限制

- TradingView Widget 在某些國家或企業防火牆下可能無法加載
- 確保有穩定的網際網路連接
- 某些指標可能需要 TradingView Pro 帳戶（但基本圖表無需)

### 2. CORS 跨域限制

- 監測面板的自動更新可能受到 CORS 限制
- 目前使用本地 JSON 數據 (dashboard_state.json)
- 若需實時更新，考慮使用 GitHub API 或 Webhook

### 3. GitHub Pages 限制

- 每個倉庫最大 1GB 存儲空間
- 每月 100GB 流量限制
- 構建時間限制 10 分鐘

### 4. 監測數據存儲

- CSV 數據會隨著時間增長
- 建議定期歸檔舊數據
- 考慮使用 GitHub Releases 存儲歷史數據

---

## 🛠️ 常見問題

### Q1: 為什麼儀表板顯示為空白？

**A**: 可能是 TradingView Widget 未加載，檢查：
1. 網際網路連接是否正常
2. 防火牆是否阻止 TradingView
3. 瀏覽器控制台是否有錯誤 (F12)

### Q2: 右側監測面板如何自動更新？

**A**: 目前有三種方式：
1. **自動計算**：頁面加載時根據配置自動計算
2. **手動刷新**：點擊「🔄 刷新監測」按鈕
3. **自動推送**：建立 GitHub Actions 工作流自動更新 JSON

### Q3: 如何在本地測試儀表板？

**A**:
```bash
# 方法 1: 用 Python 簡易伺服器
cd ~/GITHUB/tradingview-dashboard
python -m http.server 8000

# 訪問: http://localhost:8000/liquidity_dashboard_enhanced.html

# 方法 2: 用 VS Code Live Server
# 在 VS Code 中右鍵點擊檔案 → "Open with Live Server"
```

### Q4: 如何更新儀表板版本？

**A**: 直接編輯 `liquidity_dashboard_enhanced.html`，然後：
```bash
git add liquidity_dashboard_enhanced.html
git commit -m "update: Dashboard v11.x improvements"
git push origin main
```

---

## 📈 部署檢查清單

- [ ] ✅ 複製 `liquidity_dashboard_enhanced.html` 到 `~/GITHUB/tradingview-dashboard/`
- [ ] ✅ 在本地瀏覽器測試
- [ ] [ ] `git add liquidity_dashboard_enhanced.html`
- [ ] [ ] `git commit -m "feat: Add enhanced dashboard v11"`
- [ ] [ ] `git push origin main`
- [ ] [ ] 驗證 GitHub Pages 已部署 (5-10 秒後)
- [ ] [ ] 訪問 https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html
- [ ] [ ] 測試右側監測面板功能
- [ ] [ ] 測試 TradingView 圖表加載
- [ ] [ ] 考慮設定 GitHub Actions 自動化

---

## 🔗 相關連結

- **倉庫**: https://github.com/t0mst0ne/t0mst0ne.github.io
- **GitHub Pages 設定**: https://github.com/t0mst0ne/t0mst0ne.github.io/settings/pages
- **TradingView**: https://www.tradingview.com/
- **FRED API**: https://fred.stlouisfed.org/docs/api

---

## 📝 提交訊息模板

```
feat: Add enhanced liquidity dashboard v11 with real-time monitoring

Adds liquidity_dashboard_enhanced.html featuring:
- 12 TradingView charts in 3x4 grid
- Real-time monitoring panel with live liquidity metrics
- Status indicator bar with key indicators
- Momentum allocation recommendations
- Auto-refresh every 5 minutes
- Integration with liquidity_monitor.py output

Files added:
- liquidity_dashboard_enhanced.html (24KB)
- GITHUB_PAGES_DEPLOYMENT.md (deployment guide)

Deployed to: https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html
```

---

**版本**：1.0  
**最後更新**：2026-07-08  
**狀態**：✅ 準備推送

