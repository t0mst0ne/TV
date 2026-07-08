# 🤖 GitHub Pages 自動化更新 - 快速啟用

讓儀表板自動每天更新，無需本地執行任何命令

---

## ⚡ 3 分鐘快速啟用

### Step 1️⃣: 申請 FRED API Key (1 分鐘)

```bash
# 訪問 FRED 官網
https://fred.stlouisfed.org/docs/api/api_key.html

# 免費註冊 → 取得 API Key
# 複製 API Key 文本
```

### Step 2️⃣: 設定 GitHub Secret (1 分鐘)

```bash
# 進入倉庫設定
https://github.com/t0mst0ne/t0mst0ne.github.io/settings/secrets/actions

# 點擊 "New repository secret"
# 名稱: FRED_API_KEY
# 值: 粘貼你的 API Key
# 點擊 "Add secret"
```

### Step 3️⃣: 提交 Workflow 文件 (1 分鐘)

```bash
cd ~/GITHUB/tradingview-dashboard

# 檢查工作流文件
ls -la .github/workflows/update-dashboard.yml

# 如果存在，直接推送
git add .github/workflows/update-dashboard.yml GITHUB_ACTIONS_SETUP.md
git commit -m "add: GitHub Actions auto-update workflow"
git push origin main
```

---

## ✅ 驗證設定

### 第 1 次手動執行

```bash
# 用 GitHub CLI 手動觸發
gh workflow run update-dashboard.yml

# 或在 GitHub UI:
# 進入 Actions → "Auto Update Liquidity Dashboard" → "Run workflow"
```

### 查看執行狀態

```bash
# 查看最新執行
gh run list --workflow update-dashboard.yml --limit 1

# 查看詳細日誌
gh run view <run_id> --log
```

### 驗證儀表板更新

訪問儀表板：
```
https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html
```

右邊面板應該顯示：
```
✅ 淨流動性: 最新數據
✅ SOFR 利率: 最新數據
✅ 壓力指標: 最新數據
✅ HY Spread: 最新數據
✅ Alpha 倉位: 最新建議
⏰ 最後更新: [時間戳]
```

---

## 📅 自動執行時間表

工作流會在以下時間自動執行：

| 時間 | 時區 |
|------|------|
| 🕐 9:30 AM | 台灣時間 (UTC+8) |
| 1:30 AM | UTC |
| 每天執行 | 24h 週期 |

### 修改執行時間

編輯 `.github/workflows/update-dashboard.yml`：

```yaml
on:
  schedule:
    - cron: '30 1 * * *'  # 修改這行
```

例如改為 12:00 PM：
```yaml
    - cron: '4 * * * *'   # 12:00 PM UTC+8
```

---

## 🔄 工作流執行邏輯

每次執行時 Workflow 會：

```
1. 檢出倉庫 (dashboard + Momentum)
   ↓
2. 安裝 Python 3.9 + 依賴
   ↓
3. 執行 liquidity_monitor.py
   └─→ 從 FRED API 抓取最新流動性數據
   └─→ 生成 liquidity_monitor.csv
   ↓
4. 執行 update_dashboard.py
   └─→ 評估流動性狀態
   └─→ 生成 Momentum 倉位建議
   └─→ 生成 dashboard_state.json
   ↓
5. 複製生成的文件到儀表板倉庫
   ├─ dashboard_state.json ← 儀表板讀取
   ├─ liquidity_monitor.csv ← 歷史記錄
   └─ momentum_liquidity_decisions.csv ← 決策歷史
   ↓
6. 自動提交並推送到 GitHub
   ↓
7. GitHub Pages 自動更新
   ↓
8. 儀表板自動讀取最新 JSON
```

---

## 📊 監控執行

### GitHub Actions 頁面

https://github.com/t0mst0ne/t0mst0ne.github.io/actions

看到：
- ✅ **Workflows**: 執行狀態
- 📋 **Run history**: 歷史紀錄
- ⏱️ **Duration**: 執行時間

### 命令列

```bash
# 監控最新執行
gh run list --workflow update-dashboard.yml --limit 5

# 實時跟蹤
gh run watch $(gh run list --workflow update-dashboard.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

---

## ⚠️ 常見問題

### Q: Workflow 沒有執行

**A**: 
1. 確認 `.github/workflows/update-dashboard.yml` 已推送到 main 分支
2. 檢查 Actions 是否在倉庫設定中啟用

```bash
# 重新推送確保文件存在
git add .github/workflows/update-dashboard.yml
git commit -m "ensure workflow file"
git push origin main
```

### Q: FRED API Key 錯誤

**A**:
1. 重新驗證 API Key 是否正確
2. 檢查 GitHub Secret 是否設定

```bash
# 驗證 Secret 存在 (但看不到值)
gh secret list --repo t0mst0ne/t0mst0ne.github.io
```

### Q: 儀表板沒有更新

**A**:
1. 清除瀏覽器緩存 (Ctrl+Shift+R)
2. 檢查 dashboard_state.json 是否存在

```bash
# 檢查最新提交
gh api repos/t0mst0ne/t0mst0ne.github.io/commits \
  --jq '.[0] | {message: .commit.message, date: .commit.author.date}'
```

---

## 💾 檔案清單

新增的自動化檔案：

| 檔案 | 用途 |
|------|------|
| `.github/workflows/update-dashboard.yml` | GitHub Actions 工作流定義 |
| `GITHUB_ACTIONS_SETUP.md` | 詳細設定指南 |
| `AUTOMATED_SETUP.md` | 本檔案 (快速啟用) |
| `dashboard_state.json` | 自動生成的監測數據 |
| `liquidity_monitor.csv` | 自動生成的歷史記錄 |

---

## 🎯 完成清單

- [ ] ✅ 申請 FRED API Key
- [ ] ✅ 在 GitHub Secrets 設定 FRED_API_KEY
- [ ] ✅ 推送 Workflow 文件到 main 分支
- [ ] ✅ 手動執行 Workflow 驗證
- [ ] ✅ 檢查儀表板右邊面板顯示最新數據
- [ ] ✅ 設定完成！自動更新已啟用 🚀

---

## 📈 下一步

1. **監控自動執行**
   - 進入 GitHub Actions 查看執行歷史

2. **自訂執行時間** (可選)
   - 編輯 `.github/workflows/update-dashboard.yml` 中的 cron 表達式

3. **分享儀表板**
   - URL: https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html

4. **檢查執行日誌** (需要時)
   - 每次執行失敗時，GitHub 會發送郵件通知
   - 進入 Actions 頁面查看詳細日誌

---

## 🎉 設定完成

現在你的儀表板會 **每天自動更新** ⏰

```
📅 每天 9:30 AM
    ↓
🤖 自動執行監測
    ↓
📊 自動更新儀表板
    ↓
🌐 GitHub Pages 自動部署
    ↓
✅ 完成！無需任何手動操作
```

**祝你使用愉快！** 🚀
