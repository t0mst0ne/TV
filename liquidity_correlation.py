import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def fetch_data():
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365*10)
    
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")

    # 1. Fetch Yahoo Finance Data
    yf_symbols = ['SPY', 'BTC-USD', 'JPY=X']
    print(f"Fetching YF symbols: {yf_symbols}")
    df_yf = yf.download(yf_symbols, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    
    # Rename YF columns for clarity
    df_yf.rename(columns={'JPY=X': 'USD/JPY (YF)', 'BTC-USD': 'BTC (YF)', 'SPY': 'SPY'}, inplace=True)

    # 2. Fetch FRED Data
    # Mapping based on liquidity.html analysis
    # WALCL: Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level
    # WTREGEN: Liabilities and Capital: Liabilities: Deposits with F.R. Banks, Other Than Reserve Balances: U.S. Treasury, General Account: Week Average
    # RRPONTSYD: Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations
    # SOFR: Secured Overnight Financing Rate
    # IORB: Interest Rate on Reserve Balances
    # DGS10: Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
    # DGS30: Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity
    # BAMLH0A0HYM2: ICE BofA US High Yield Index Option-Adjusted Spread
    # BAMLC0A0CM: ICE BofA US Corporate Index Option-Adjusted Spread
    
    fred_symbols = [
        'WALCL', 'WTREGEN', 'RRPONTSYD', 
        'SOFR', 'IORB', 
        'DGS10', 'DGS30', 
        'BAMLH0A0HYM2', 'BAMLC0A0CM'
    ]
    
    print(f"Fetching FRED symbols: {fred_symbols}")
    try:
        df_fred = web.DataReader(fred_symbols, 'fred', start_date, end_date)
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return None

    # 3. Data Processing & Derived Indicators based on liquidity.html
    
    # Net Liquidity = Fed Balance Sheet (WALCL) - TGA (WTREGEN) - RRP (RRPONTSYD)
    # Note: WALCL is in Millions, WTREGEN and RRPONTSYD are in Billions.
    # We must convert WALCL to Billions => WALCL / 1000
    
    # Handle missing data (FREQ mismatch: WALCL is Weekly, others daily)
    # We forward fill to align weekly data to daily for calculation
    df_fred = df_fred.fillna(method='ffill')
    
    df_fred['Net Liquidity'] = (df_fred['WALCL'] / 1000) - df_fred['WTREGEN'] - df_fred['RRPONTSYD']
    
    # Liquidity Stress Spread = SOFR - IORB
    df_fred['Liquidity Stress'] = df_fred['SOFR'] - df_fred['IORB']
    
    # Rename helpful columns
    df_fred.rename(columns={
        'DGS10': 'US 10Y Yield',
        'DGS30': 'US 30Y Yield',
        'BAMLH0A0HYM2': 'HY Spread',
        'BAMLC0A0CM': 'Corp Spread',
        'RRPONTSYD': 'Reverse Repo',
        'WTREGEN': 'TGA'
    }, inplace=True)
    
    # Combine DataFrames
    # Merge on index (Date)
    df_combined = pd.merge(df_yf, df_fred, left_index=True, right_index=True, how='inner')
    
    # Fill remaining gaps
    df_combined.fillna(method='ffill', inplace=True)
    
    return df_combined

def analyze_correlation(df):
    print("\nRunning Correlation Analysis (Weekly Returns)...")
    
    # Resample to weekly to reduce noise and align better with macro trends
    df_weekly = df.resample('W-FRI').last()
    
    # Calculate Weekly Percent Changes (Returns)
    # For Yields and Spreads, changes in basis points or just raw changes are often more correlated 
    # than percent changes (e.g. yield going from 1% to 1.1% is 10% change, but 4% to 4.1% is 2.5%).
    # However, for Price assets (SPY, BTC), we MUST use % change.
    # For consistency in "Directional Correlation", using differencing for rates and % change for prices 
    # is theoretically best, but simple pct_change is often used for a quick look.
    # Let's use pct_change for prices and diff for rates/spreads?
    # To keep it simple and standard as per "correlation with SPY returns":
    
    # Prices: SPY, BTC (YF), USD/JPY (YF), Net Liquidity (It's a volume/level, roughly treating as asset for corr)
    # Rates/Spreads: US 10Y, US 30Y, HY Spread, Corp Spread, Liquidity Stress, Reverse Repo, TGA
    
    df_returns = pd.DataFrame()
    
    # Prices / Levels -> % Change
    cols_pct = ['SPY', 'BTC (YF)', 'USD/JPY (YF)', 'Net Liquidity', 'Reverse Repo', 'TGA']
    for col in cols_pct:
        if col in df_weekly.columns:
            df_returns[col] = df_weekly[col].pct_change()
            
    # Rates / Spreads -> Difference (Change in value)
    # If yield goes up, bond price down => negative correlation with stocks usually?
    cols_diff = ['US 10Y Yield', 'US 30Y Yield', 'HY Spread', 'Corp Spread', 'Liquidity Stress']
    for col in cols_diff:
        if col in df_weekly.columns:
            df_returns[col] = df_weekly[col].diff()

    # Drop NaN
    df_returns.dropna(inplace=True)
    
    # Calculate Correlation Matrix
    corr_matrix = df_returns.corr()
    
    # Extract SPY correlations
    spy_corr = corr_matrix['SPY'].sort_values(ascending=False)
    
    print("\n10-Year Correlation with SPY (Weekly Changes):")
    print("-" * 50)
    print(spy_corr)
    print("-" * 50)
    
    return corr_matrix

def plot_heatmap(corr_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('10-Year Correlation Matrix (Weekly Changes)')
    plt.tight_layout()
    plt.savefig('liquidity_correlation_heatmap.png')
    print("\nCorrelation heatmap saved to 'liquidity_correlation_heatmap.png'")

import matplotlib.dates as mdates
import numpy as np

def generate_comprehensive_analysis(df):
    print("\nGenerating Comprehensive Analysis Charts...")
    
    # Ensure output directory exists
    import os
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')

    # Weekly Resample for Analysis
    df_weekly = df.resample('W-FRI').last()
    
    # Calculate Weekly Returns/Diffs for Rolling/Scatter
    df_changes = pd.DataFrame()
    
    # Define what is percent change and what is diff
    cols_pct = ['SPY', 'BTC (YF)', 'USD/JPY (YF)', 'Net Liquidity', 'Reverse Repo', 'TGA']
    cols_diff = ['US 10Y Yield', 'US 30Y Yield', 'HY Spread', 'Corp Spread', 'Liquidity Stress']
    
    # Metrics to analyze against SPY
    metrics = [c for c in df.columns if c != 'SPY']
    
    for col in metrics:
        if col not in df_weekly.columns:
            continue
            
        print(f"Processing {col}...")
        
        # Prepare Data
        # 1. Price/Level Data (Weekly)
        spy_prices = df_weekly['SPY']
        metric_prices = df_weekly[col]
        
        # 2. Changes Data (Weekly)
        if col in cols_pct:
            spy_chg = df_weekly['SPY'].pct_change()
            metric_chg = df_weekly[col].pct_change()
            metric_label = f"{col} (% Chg)"
        elif col in cols_diff:
            spy_chg = df_weekly['SPY'].pct_change()
            metric_chg = df_weekly[col].diff()
            metric_label = f"{col} (Diff)"
        else:
            # Fallback
            spy_chg = df_weekly['SPY'].pct_change()
            metric_chg = df_weekly[col].pct_change()
            metric_label = f"{col} (% Chg)"
            
        # Drop initial NaNs for correlation calc
        valid_data = pd.DataFrame({'SPY': spy_chg, 'Metric': metric_chg}).dropna()
        
        # --- PLOTTING ---
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'{col} vs SPY (10-Year Comprehensive Analysis)', fontsize=16)
        
        # Grid layout: 
        # Row 1: Price Overlay (Full width)
        # Row 2: Rolling Correlation (Left), Scatter (Right)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        # 1. Price Overlay (Dual Y-Axis)
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('SPY Price', color=color, fontweight='bold')
        ax1.plot(df_weekly.index, spy_prices, color=color, label='SPY', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax1_dup = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:orange'
        ax1_dup.set_ylabel(col, color=color, fontweight='bold')
        ax1_dup.plot(df_weekly.index, metric_prices, color=color, label=col, linewidth=1.5, linestyle='--')
        ax1_dup.tick_params(axis='y', labelcolor=color)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_dup.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.set_title("Price / Level History (Weekly)")

        # 2. Rolling Correlation
        window_6m = 26
        window_1y = 52
        
        roll_6m = valid_data['SPY'].rolling(window=window_6m).corr(valid_data['Metric'])
        roll_1y = valid_data['SPY'].rolling(window=window_1y).corr(valid_data['Metric'])
        
        ax2.plot(roll_6m.index, roll_6m, label='6-Month Rolling Corr', color='purple', alpha=0.7)
        ax2.plot(roll_1y.index, roll_1y, label='1-Year Rolling Corr', color='green', linewidth=2)
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_title("Rolling Correlation (Weekly Changes)")
        ax2.set_ylabel("Correlation Coefficient")
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.0, 1.0) 

        # 3. Scatter Plot with Regression
        sns.regplot(x=valid_data['SPY']*100, y=valid_data['Metric']*100 if col in cols_pct else valid_data['Metric'], 
                    ax=ax3, scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
        
        # Calc overall correlation for title
        overall_corr = valid_data['SPY'].corr(valid_data['Metric'])
        
        ax3.set_title(f"Weekly Returns Scatter (Correlation: {overall_corr:.2f})")
        ax3.set_xlabel("SPY Weekly Return (%)")
        ax3.set_ylabel(f"{metric_label} {'(%)' if col in cols_pct else '(bps/val)'}")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Sanitize filename
        safe_col_name = col.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"analysis_results/SPY_vs_{safe_col_name}.png"
        plt.savefig(filename)
        plt.close() # Close figure to free memory
        
    print("Charts generated in 'analysis_results/' folder.")

if __name__ == "__main__":
    df = fetch_data()
    if df is not None:
        corr = analyze_correlation(df)
        plot_heatmap(corr)
        generate_comprehensive_analysis(df)
