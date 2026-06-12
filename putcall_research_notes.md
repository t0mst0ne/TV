# Put/Call Ratio 研究筆記：哪一種序列配 BB 帶最能抓 SPX 高低點

> 2026-06-12，搭配 `putcall.html` dashboard 使用。
> 結論先講：**股票型 PCCE（第 1 格）是文獻上唯一站得住的選擇，且只對「抓低點」有效。**

## 問題

面板上有六種 put/call 序列（股票型 PCCE、總量 PCC、指數型 PCCI、SPX 專屬 PCSPX、OEX、PCCE/PCCI 比值），研究文獻上哪一種配合 Bollinger Band（20MA ± 2σ）極值，對 SPX 轉折點的捕捉力最好？

## 各候選主張與反駁

### 主張一：「總量 PCC 才對」
Simon & Wiggins (2001) 實證用的就是總量比率，有統計顯著性。

**反駁**：總量 = 股票型 + 指數型的混合。Dennis & Mayhew (2002)、Lakonishok et al. (2007) 的微觀結構研究指出，**指數選擇權由機構避險流量主導**——put 買盤是常態性保險（所以 PCCI 常態 >1），不反映方向性情緒，混進來只會稀釋訊號。Simon & Wiggins 用總量是受當年資料限制，不是因為總量更好。

### 主張二：「PCSPX 才對，直接對口 SPX」
**反駁**：純機構避險流量，且 2022 後 SPX 成交量被 0DTE 策略佔據（Cboe 自家研究估超過四成），當日沖銷的 put/call 與「對未來行情的看法」幾乎無關。PCSPX 作為情緒指標的訊噪比是六者中退化最嚴重的。

### 主張三：「OEX 是聰明錢，最準」
**反駁**：「聰明錢」地位是 1990 年代的遺產。OEX 成交量如今極小，面板第 8 格噴到 40+ 的尖峰就是低成交量雜訊。BB 套在這種序列上會不斷誤報。

## 結論：股票型 PCCE

- **理論依據**：股票型選擇權由散戶方向性交易主導（Pan & Poteshman 2006 區分交易者類型後，散戶情緒成分最濃），是六者中唯一「純情緒」序列——逆向指標要的就是這個。
- **配 BB 的理由**：PCCE 均值有結構性漂移（2021 散戶選擇權熱潮壓到歷史低位），固定閾值會失效；自適應 ±2σ 帶是 McMillan「相對極值法」的標準實作。

### 不對稱性（文獻相當一致）

| 方向 | 有效性 | 原因 |
|------|--------|------|
| **抓低點**（PCCE 突破 BB 上軌） | 有效 | 恐慌是尖峰式的，通常與投降式賣壓同步；對 SPX 低點是同時指標、偶爾領先。Simon & Wiggins 的統計顯著性也集中在恐慌端 |
| **抓高點**（PCCE 觸 BB 下軌） | 很弱 | 自滿可持續數月，貼下軌走是多頭常態；當賣出訊號會過早且誤報率高。頭部是過程不是事件，沒有任何 put/call 序列能靠 BB 可靠抓頂 |

### 實務操作（對應 dashboard 格位）

1. **主訊號**：第 1 格 PCCE 觸 BB 上軌 → 找低點。
2. **確認訊號**：第 6 格 VIX 期限結構（3M/Spot）< 1（倒掛）→ 兩者同時出現是文獻上最有把握的「可能反轉」組合。
3. **輔助判讀**：第 7 格 PCCE/PCCI 比值 → 區分恐慌來自散戶端（可反向解讀）或機構端（避險流量，不宜反向解讀）。

## 參考文獻

- Simon, D. & Wiggins, R. (2001). "S&P futures returns and contrary sentiment indicators." *Journal of Futures Markets* 21(5). — 公開 CBOE put/call 極值對後續 S&P 期貨報酬有統計顯著的逆向預測力。
- Pan, J. & Poteshman, A. (2006). "The Information in Option Volume for Future Stock Prices." *Review of Financial Studies* 19(3). — put/call 預測力主要來自知情者開倉資料；公開總量訊號較弱，適合當警示而非機械訊號。
- Lakonishok, J., Lee, I., Pearson, N. & Poteshman, A. (2007). "Option Market Activity." *Review of Financial Studies* 20(3). — 散戶 vs 機構的選擇權使用行為差異。
- Dennis, P. & Mayhew, S. (2002). "Risk-Neutral Skewness: Evidence from Stock Options." *JFQA* 37(3). — 指數選擇權的機構避險主導結構。
- Zweig 逆向指標傳統：股票型 P/C > 1.0 過度悲觀（偏多反轉）/ < 0.5 過度樂觀；總量 > 1.2 / < 0.8。
- McMillan, L. *Options as a Strategic Investment* — 相對極值法（移動平均 ± 標準差帶取代固定閾值）；OEX 聰明錢解讀（高 = 機構買 put = 偏空，非逆向）。

## 限制

本筆記是文獻結論，**不是近期資料的回測**。Cboe 2019 年後停止免費公開 put/call 歷史檔，實證驗證需另找資料源（或從 TradingView 手動匯出）；可用 2003–2019 舊檔先做歷史回測。
