# GitHub Actions 自動化設定指南

自動化儀表板更新，無需本地執行腳本

---

## 📋 架構

```
GitHub Actions (每天 9:30 AM 自動執行)
    ↓
1. 檢出 tradingview-dashboard 倉庫
2. 檢出 Momentum 倉庫
3. 安裝 Python 依賴
4. 執行 liquidity_monitor.py (從 FRED API 抓取數據)
5. 執行 update_dashboard.py (生成 dashboard_state.json)
6. 複製 JSON 到 tradingview-dashboard
7. 自動提交並推送到 GitHub
    ↓
GitHub Pages 自動更新
    ↓
儀表板從 dashboard_state.json 讀取最新數據
```

---

## 🔐 第 1 步：設定 FRED API Key

### 1.1 取得 FRED API Key

```bash
# 訪問 FRED 官網
https://fred.stlouisfed.org/docs/api/api_key.html

# 免費註冊帳戶
# 複製你的 API Key
```

### 1.2 添加到 GitHub Secrets

1. 打開倉庫: https://github.com/t0mst0ne/t0mst0ne.github.io
2. 進入 **Settings** → **Secrets and variables** → **Actions**
3. 點擊 **New repository secret**
4. 名稱: `FRED_API_KEY`
5. 值: 粘貼你的 FRED API Key
6. 點擊 **Add secret**

**驗證**：
```bash
# 在 GitHub Actions 中，密鑰會顯示為:
FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
```

---

## 📅 第 2 步：驗證 Workflow 設定

### 2.1 檢查工作流文件

工作流文件位置：
```
~/.github/workflows/update-dashboard.yml
```

該文件已自動建立，配置內容：
- **觸發時間**：每天 9:30 AM UTC+8 (1:30 AM UTC)
- **手動觸發**：可在 GitHub UI 手動執行
- **Python 版本**：3.9
- **依賴**：pandas, requests, yfinance

### 2.2 查看 Workflow 執行紀錄

```bash
# 方式 1: 在 GitHub UI 中
Settings → Actions → All workflows → Auto Update Liquidity Dashboard

# 方式 2: 用命令列
gh workflow list
gh run list --workflow update-dashboard.yml
```

---

## 🚀 第 3 步：手動觸發第一次執行

```bash
# 方式 1: 用 GitHub CLI
gh workflow run update-dashboard.yml

# 方式 2: 在 GitHub UI 中
1. 進入 Actions 標籤
2. 選擇 "Auto Update Liquidity Dashboard"
3. 點擊 "Run workflow"
```

---

## 📊 第 4 步：驗證執行結果

### 4.1 查看 Workflow 執行詳情

```bash
# 取得最新執行紀錄
gh run list --workflow update-dashboard.yml --limit 5

# 查看詳細日誌
gh run view <run_id> --log
```

### 4.2 檢查生成的文件

執行完成後，以下文件應該會在倉庫中更新：

```
dashboard_state.json          ← 最新的監測數據
liquidity_monitor.csv         ← 歷史監測記錄
momentum_liquidity_decisions.csv ← 倉位建議歷史
```

### 4.3 驗證儀表板更新

訪問儀表板，應該看到：
```
https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html
```

右邊面板應顯示最新的監測數據（不再是「等待數據」）

---

## ⏰ 自訂執行時間

編輯 `.github/workflows/update-dashboard.yml`：

### 更改執行時間

```yaml
on:
  schedule:
    - cron: '30 1 * * *'  # 每天 9:30 AM UTC+8 (1:30 AM UTC)
```

#### 常見時間設定

| 時間 | Cron 表達式 | 說明 |
|------|-----------|------|
| 每天 9:30 AM | `30 1 * * *` | 1:30 AM UTC |
| 每天 12:00 PM | `4 * * * *` | 4:00 AM UTC |
| 每小時執行 | `0 * * * *` | 每小時整點 |
| 每 6 小時 | `0 */6 * * *` | 每 6 小時 |

**轉換時間**: 台灣時間 = UTC + 8 小時

---

## 🔄 自動化流程

### 工作流執行流程

```
┌─────────────────────────────────────────────────────┐
│ 觸發時間: 每天 9:30 AM                              │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 1. 檢出倉庫                                          │
│    - tradingview-dashboard (這個倉庫)               │
│    - Momentum (監測腳本)                            │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 2. 設定 Python 環境                                 │
│    - Python 3.9                                     │
│    - pandas, requests, yfinance                     │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 3. 執行監測腳本                                      │
│    - liquidity_monitor.py (從 FRED API 抓取)        │
│    - update_dashboard.py (生成 JSON)                │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 4. 複製生成的數據                                    │
│    - dashboard_state.json                          │
│    - liquidity_monitor.csv                         │
│    - momentum_liquidity_decisions.csv               │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 5. 提交並推送到 GitHub                              │
│    git add *.json *.csv                             │
│    git commit -m "chore: Auto-update..."            │
│    git push origin main                            │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 6. GitHub Pages 自動更新                           │
│    https://t0mst0ne.github.io/...                   │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 7. 儀表板自動讀取最新數據                             │
│    右邊面板顯示最新監測數據                          │
└─────────────────────────────────────────────────────┘
```

---

## 📈 監控執行狀態

### GitHub Actions 頁面

進入: https://github.com/t0mst0ne/t0mst0ne.github.io/actions

看到：
- ✅ **Latest Runs**: 執行歷史
- 📊 **Workflow Runs**: 詳細日誌
- ⏱️ **Duration**: 執行時間

### 命令列監控

```bash
# 查看最新 5 次執行
gh run list --workflow update-dashboard.yml --limit 5

# 實時監控最新執行
gh run watch $(gh run list --workflow update-dashboard.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

---

## ⚠️ 故障排查

### 問題 1: Workflow 沒有執行

**原因**：
- GitHub Actions 未啟用
- Workflow 文件未提交到 main 分支

**解決**：
```bash
cd ~/GITHUB/tradingview-dashboard
git add .github/workflows/update-dashboard.yml
git commit -m "add: GitHub Actions workflow for auto-update"
git push origin main
```

### 問題 2: FRED API Key 未設定

**症狀**：
```
Error: FRED API key not set
```

**解決**：
1. 進入 Settings → Secrets
2. 驗證 `FRED_API_KEY` 已設定
3. 手動觸發 workflow 重試

### 問題 3: Workflow 執行失敗

**檢查日誌**：
```bash
gh run list --workflow update-dashboard.yml --limit 1 -x

# 查看詳細錯誤
gh run view <run_id> --log
```

常見錯誤：
- `ModuleNotFoundError: No module named 'pandas'` → 依賴未安裝
- `403 Forbidden FRED API` → API Key 無效
- `Permission denied` → GitHub Token 權限不足

### 問題 4: 數據未更新到 GitHub Pages

**原因**：提交失敗或 Pages 緩存

**解決**：
```bash
# 手動強制刷新
1. 進入儀表板 URL
2. 按 Ctrl+Shift+R (清除緩存刷新)

# 檢查倉庫是否收到更新
gh api repos/t0mst0ne/t0mst0ne.github.io/commits \
  --jq '.[] | select(.commit.message | contains("Auto-update")) | .commit.author.date' \
  | head -1
```

---

## 📋 設定檢查清單

- [ ] ✅ FRED API Key 已申請
- [ ] ✅ FRED API Key 已添加到 GitHub Secrets
- [ ] ✅ Workflow 文件 `.github/workflows/update-dashboard.yml` 已提交
- [ ] ✅ 倉庫有分支保護規則允許 Actions 推送 (optional)
- [ ] ✅ 手動執行一次 Workflow 驗證
- [ ] ✅ 檢查 dashboard_state.json 已更新
- [ ] ✅ 儀表板顯示最新數據
- [ ] ✅ 定時自動執行已驗證

---

## 🎯 自動化完成

現在儀表板會每天 **自動更新**：

```
每天 9:30 AM
    ↓
自動執行監測 (liquidity_monitor.py)
    ↓
自動生成數據 (dashboard_state.json)
    ↓
自動推送到 GitHub
    ↓
GitHub Pages 自動部署
    ↓
https://t0mst0ne.github.io/tradingview-dashboard/liquidity_dashboard_enhanced.html
    ↓
右邊面板自動顯示最新流動性數據
```

**無需手動執行任何命令！** 🚀

---

## 📞 常用命令

```bash
# 查看所有 Workflow
gh workflow list

# 手動觸發 Workflow
gh workflow run update-dashboard.yml

# 查看執行歷史
gh run list --workflow update-dashboard.yml

# 查看詳細日誌
gh run view <run_id> --log

# 檢查最新數據時間戳
curl -s https://raw.githubusercontent.com/t0mst0ne/t0mst0ne.github.io/main/dashboard_state.json | jq '.timestamp'
```

---

**版本**: 1.0  
**狀態**: ✅ 自動化已啟用  
**下次執行**: 明天 9:30 AM
