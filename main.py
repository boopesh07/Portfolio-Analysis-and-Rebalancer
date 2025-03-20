from dotenv import load_dotenv
from config import Config
import requests
import pandas as pd
import io
import aiohttp
import asyncio
from typing import Dict

load_dotenv()

config = Config.get_config()
API_KEY = config.FINNHUB_API_KEY

# Simple dictionary to cache stock prices
price_cache: Dict[str, float] = {}

# Example target allocation
target_allocation = {"Technology": 60, "Automotive": 20, "Finance": 20}

async def fetch_stock_price_async(symbol: str, session: aiohttp.ClientSession) -> float:
    # Check if price is in cache
    if symbol in price_cache:
        print(f"Cache hit for {symbol}")
        return price_cache[symbol]
        
    base_url = "https://www.finnhub.io/api/v1/quote"
    params = {
        "symbol": symbol,
        "token": API_KEY
    }
    
    try:
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                current_price = data["c"]
                # Store in cache
                price_cache[symbol] = current_price
                print(f"Fetched and cached price for {symbol}")
                return current_price
            else:
                print(f"Error fetching {symbol}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

async def update_stock_current_prices_in_portfolio(stocks_df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the DataFrame to avoid modifying the original
    updated_df = stocks_df.copy()
    
    # Create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # Create tasks for all stock symbols
        tasks = []
        for symbol in updated_df['Stock Symbol']:
            tasks.append(fetch_stock_price_async(symbol, session))
        
        # Wait for all requests to complete
        prices = await asyncio.gather(*tasks)
        
        # Update the current prices in the DataFrame
        updated_df['Current Price (USD)'] = prices
        
        # Remove rows where price fetch failed (price is None)
        updated_df = updated_df.dropna(subset=['Current Price (USD)'])
        
        return updated_df

def clear_price_cache():
    """Clear the price cache"""
    price_cache.clear()
    print("Price cache cleared")

# Keep the existing fetch_stock_price for backward compatibility
def fetch_stock_price(symbol):
    base_url = "https://www.finnhub.io/api/v1/quote"
    params = {
        "symbol": symbol,
        "token": API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return data["c"]  # Return the current price
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stock price: {e}")
        return None
    

stock_data = """Stock Symbol,Number of Shares,Purchase Price (USD),Current Price (USD),Sector
AAPL,10,150.00,170.00,Technology
GOOGL,5,1200.00,1250.00,Technology
TSLA,3,700.00,650.00,Automotive
JPM,20,100.00,110.00,Finance"""

def parse_stock_data(data: str):
    # Convert the string data into a DataFrame using pandas read_csv
    # Use StringIO to create a file-like object from the string
    df = pd.read_csv(io.StringIO(data))
    
    # Convert numeric columns to appropriate types
    df['Number of Shares'] = df['Number of Shares'].astype(int)
    df['Purchase Price (USD)'] = df['Purchase Price (USD)'].astype(float)
    df['Current Price (USD)'] = df['Current Price (USD)'].astype(float)
    
    return df

def calculate_total_portfolio(stocks_df):
    # Calculate portfolio value for each stock (current price * number of shares)
    stocks_df['Portfolio Value'] = stocks_df['Current Price (USD)'] * stocks_df['Number of Shares']
    
    # Calculate total portfolio value
    total_portfolio_value = stocks_df['Portfolio Value'].sum()
    
    return total_portfolio_value

def calculate_portfolio_returns(stocks_df):
    """
    Calculate the total portfolio return percentage.
    Return = (Current Value - Purchase Value) / Purchase Value * 100
    """
    # Calculate total gain/loss for each position
    stocks_df['Position Return'] = stocks_df['Number of Shares'] * (stocks_df['Current Price (USD)'] - stocks_df['Purchase Price (USD)'])
    
    # Calculate total initial investment
    total_investment = (stocks_df['Number of Shares'] * stocks_df['Purchase Price (USD)']).sum()
    
    # Calculate total return
    total_return = stocks_df['Position Return'].sum()
    
    # Calculate return percentage
    return_percentage = (total_return / total_investment) * 100
    
    return return_percentage

def detect_portfolio_based_allocation_drift(portfolio_data: pd.DataFrame, target_allocation: Dict[str, float]) -> Dict[str, dict]:
    """
    Detect allocation drift in the portfolio by comparing current sector allocations with target allocations.
    Also analyzes individual stock drifts within each sector.
    
    Args:
        portfolio_data: DataFrame containing portfolio data
        target_allocation: Dictionary of target allocations by sector (in percentages)
    
    Returns:
        Dictionary containing current allocations, drifts, and rebalancing needs by sector,
        including individual stock breakdowns
    """
    # Calculate current value of each position
    portfolio_data['Current Value'] = portfolio_data['Number of Shares'] * portfolio_data['Current Price (USD)']
    
    # Calculate total portfolio value
    total_portfolio_value = portfolio_data['Current Value'].sum()
    
    # Calculate current allocation percentages by sector
    sector_values = portfolio_data.groupby('Sector')['Current Value'].sum()
    current_allocations = (sector_values / total_portfolio_value * 100).to_dict()
    
    # Initialize results dictionary
    results = {}
    
    # Calculate drift for each sector and its stocks
    for sector in target_allocation.keys():
        current_alloc = current_allocations.get(sector, 0)
        target_alloc = target_allocation.get(sector, 0)
        drift = current_alloc - target_alloc
        
        # Get stocks in this sector
        sector_stocks = portfolio_data[portfolio_data['Sector'] == sector]
        sector_total = sector_values.get(sector, 0)
        
        # Calculate individual stock allocations within the sector
        stock_breakdown = []
        if not sector_stocks.empty and sector_total > 0:
            for _, stock in sector_stocks.iterrows():
                stock_value = stock['Current Value']
                stock_sector_percentage = (stock_value / sector_total * 100)
                stock_portfolio_percentage = (stock_value / total_portfolio_value * 100)
                
                # Calculate individual stock's contribution to sector drift
                stock_drift_contribution = stock_portfolio_percentage - (target_alloc * stock_sector_percentage / 100)
                
                stock_breakdown.append({
                    'symbol': stock['Stock Symbol'],
                    'value': stock_value,
                    'sector_percentage': round(stock_sector_percentage, 2),
                    'portfolio_percentage': round(stock_portfolio_percentage, 2),
                    'drift_contribution': round(stock_drift_contribution, 2)
                })
        
        results[sector] = {
            'current_allocation': round(current_alloc, 2),
            'target_allocation': target_alloc,
            'drift': round(drift, 2),
            'needs_rebalancing': abs(drift) >= 5,  # Flag if drift is 5% or more
            'stocks': stock_breakdown
        }
    
    return results

def rebalance_portfolio_threshold_based(portfolio_data: pd.DataFrame, target_allocation: Dict[str, float]) -> Dict[str, list]:
    """
    Provides rebalancing recommendations for stocks that have drifted beyond the threshold (±5%).
    
    Args:
        portfolio_data: DataFrame containing portfolio data
        target_allocation: Dictionary of target allocations by sector (in percentages)
    
    Returns:
        Dictionary containing rebalancing recommendations by sector
    """
    # Get current portfolio state and drift analysis
    drift_analysis = detect_portfolio_based_allocation_drift(portfolio_data, target_allocation)
    total_portfolio_value = portfolio_data['Current Value'].sum()
    
    rebalancing_recommendations = {}
    
    for sector, data in drift_analysis.items():
        sector_recommendations = []
        target_alloc = target_allocation[sector]
        
        # Calculate target sector value
        target_sector_value = (target_alloc / 100) * total_portfolio_value
        
        for stock in data['stocks']:
            symbol = stock['symbol']
            current_value = stock['value']
            stock_info = portfolio_data[portfolio_data['Stock Symbol'] == symbol].iloc[0]
            current_shares = stock_info['Number of Shares']
            current_price = stock_info['Current Price (USD)']
            
            # Calculate target stock value based on its weight within the sector
            target_stock_percentage = stock['sector_percentage']
            target_stock_value = (target_stock_percentage / 100) * target_sector_value
            
            # Calculate value difference and required action
            value_difference = target_stock_value - current_value
            
            # If drift contribution is more than ±5%, add to recommendations
            if abs(stock['drift_contribution']) >= 5:
                shares_difference = value_difference / current_price
                action = "BUY" if shares_difference > 0 else "SELL"
                shares_to_trade = abs(round(shares_difference))
                
                if shares_to_trade > 0:  # Only include if there's a meaningful change needed
                    recommendation = {
                        'symbol': symbol,
                        'action': action,
                        'shares': shares_to_trade,
                        'current_shares': current_shares,
                        'current_price': current_price,
                        'current_allocation': stock['portfolio_percentage'],
                        'target_allocation': target_stock_percentage * (target_alloc / 100),
                        'drift': stock['drift_contribution'],
                        'estimated_value_change': round(shares_to_trade * current_price, 2)
                    }
                    sector_recommendations.append(recommendation)
        
        if sector_recommendations:
            rebalancing_recommendations[sector] = sector_recommendations
    
    return rebalancing_recommendations

parsed_stocks = parse_stock_data(stock_data)

if __name__ == "__main__":
    # Create async function to run our async code
    async def main_async():
        print("Initial Portfolio:")
        print(parsed_stocks)
        print("\nInitial Portfolio Value: $", calculate_total_portfolio(parsed_stocks))
        print("Initial Portfolio Return: {:.2f}%".format(calculate_portfolio_returns(parsed_stocks)))
        
        print("\nFirst update - Fetching all prices...")
        updated_stocks = await update_stock_current_prices_in_portfolio(parsed_stocks)
        print("\nUpdated Portfolio (First Update):")
        print(updated_stocks)
        print("Updated Portfolio Value: $", calculate_total_portfolio(updated_stocks))
        print("Updated Portfolio Return: {:.2f}%".format(calculate_portfolio_returns(updated_stocks)))
        
        print("\nChecking Portfolio Allocation Drift:")
        drift_analysis = detect_portfolio_based_allocation_drift(updated_stocks, target_allocation)
        for sector, data in drift_analysis.items():
            print(f"\n{sector}:")
            print(f"  Current Allocation: {data['current_allocation']}%")
            print(f"  Target Allocation: {data['target_allocation']}%")
            print(f"  Sector Drift: {data['drift']}%")
            if data['needs_rebalancing']:
                print("  ⚠️ Sector Needs Rebalancing!")
            
            print("  Stock Breakdown:")
            for stock in data['stocks']:
                print(f"    {stock['symbol']}:")
                print(f"      Sector Weight: {stock['sector_percentage']}%")
                print(f"      Portfolio Weight: {stock['portfolio_percentage']}%")
                print(f"      Drift Contribution: {stock['drift_contribution']}%")
        
        print("\nRebalancing Recommendations:")
        rebalancing_recommendations = rebalance_portfolio_threshold_based(updated_stocks, target_allocation)
        if rebalancing_recommendations:
            for sector, recommendations in rebalancing_recommendations.items():
                print(f"\n{sector} Rebalancing Actions:")
                for rec in recommendations:
                    print(f"  {rec['symbol']}:")
                    print(f"    Action: {rec['action']} {rec['shares']} shares")
                    print(f"    Current Shares: {rec['current_shares']}")
                    print(f"    Current Price: ${rec['current_price']:.2f}")
                    print(f"    Current Allocation: {rec['current_allocation']:.2f}%")
                    print(f"    Target Allocation: {rec['target_allocation']:.2f}%")
                    print(f"    Drift: {rec['drift']:.2f}%")
                    print(f"    Estimated Value Change: ${rec['estimated_value_change']}")
        else:
            print("No rebalancing needed - all stocks within threshold")

    # Run the async main function
    asyncio.run(main_async())





########## FINNHUB API #############



# Example usage

############ STOCK PARSING ##############



######### MAIN FUNCTION #########

# def main():
    # print(parsed_stocks)
    # print(fetch_stock_price("AAPL"))

# main()
