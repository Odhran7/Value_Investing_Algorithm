import requests
api_key = "api_key" # No longer works :)

headers = {
    'x-rapidapi-host': "morning-star.p.rapidapi.com",
    'x-rapidapi-key': api_key
}

# Map to performance ID in MorningStar
def map_to_morningstar_id(ticker):
    url = "https://morning-star.p.rapidapi.com/market/v2/auto-complete"
    queryString = {"q": ticker}
    res = requests.get(url, headers=headers, params=queryString).json()
    # print(res['results'])
    performanceId = res['results'][0]['performanceId']
    if performanceId == None:
        performanceId = res['results'][1]['performanceId']
    return performanceId

# Obtain fair value as judged by MorningStar
def get_fair_value(ticker):
    url = "https://morning-star.p.rapidapi.com/stock/v2/get-price-fair-value"
    performanceId = map_to_morningstar_id(ticker)
    # print(performanceId)
    queryString = {"performanceId": performanceId}
    res = requests.get(url, headers=headers, params=queryString).json()["chart"]["chartDatums"]["recent"]["latestFairValue"]
    return res

# Now let's join the value data to the data frame
def join_fair_value(sp500_value):
    for i, ticker in enumerate(sp500_value['Ticker']):
        print(f'Obtaining fair value for {ticker} ({i+1}/{len(sp500_value)})')
        try:
            fair_value = get_fair_value(ticker)
            sp500_value.loc[sp500_value['Ticker'] == ticker, 'Fair Value'] = fair_value
        except Exception as e:
            print(f'Failed to obtain fair value for {ticker}')
            print(e)

# We need to generate a portfolio before attempting to retrieve fair values!

# sp500_value = pd.read_csv('./portfolio_value_data.csv')
# join_fair_value(sp500_value)

# # Save the DataFrame to a CSV file
# sp500_value.to_csv('./portfolio_value_fair_value_data.csv', index=False)