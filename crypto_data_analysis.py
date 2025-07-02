import requests
import json
import mysql.connector
from mysql.connector import errorcode
import math
import os
import pandas_datareader as pdr
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import statsmodels.discrete.discrete_model as sm
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import datetime as dt
import time

db_config = {
    'user': 'your_database_user',
    'password': 'your_database_password',
    'host': 'your_database_host',
    'database': 'your_database_name'
}


headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": "YOUR COINGECKO API KEY",
}

class getBinanceIndicators:
    def __init__(self, symbol,interval, limit):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.data = []
        #self.indicator = indicator
        daily_stats = requests.get("https://api.binance.com/api/v3/klines?symbol="+self.symbol+"&interval=" + str(self.interval) + "&limit=" + str(self.limit))
        j_response = json.loads(daily_stats.text)
        #print(j_response)

        for i in j_response:
            kline_historical = {
                'symbol': self.symbol,
                'open_time': str(dt.fromtimestamp(int(i[0])/1000)),
                'open': i[1],
                'high' : i[2],
                'low' : i[3],
                'close' : i[4],
                'volume' : i[5],
                'close_time' : dt.fromtimestamp(int(i[6])/1000),
                'quote_asset_volume' : i[7],
                'number_of_trades' : i[8],
                'taker_buy_base_asset_volume' : i[9],
                'taker_buy_quote_asset_volume' : i[10]
            }
            self.data.append(kline_historical)
        
        #json to dataframe
        self.df = pd.json_normalize(self.data)
        self.df = self.df[['symbol', 'open_time','close_time','open', 'high', 'low', 'close']]
        self.df['open'] = self.df.open.astype(float)
        self.df['high'] = self.df.high.astype(float)
        self.df['close'] = self.df.close.astype(float)
        #print(self.df)
    
    #each indicator returns dataframes
    def rsi(self, window):
        #print(self)
        delta = self.df.close.diff()
        up_days = delta.copy()
        up_days[delta<=0]=0.0
        down_days = abs(delta.copy())
        down_days[delta>0]=0.0
        RS_up = up_days.rolling(window).mean()
        RS_down = down_days.rolling(window).mean()
        rsi= 100-100/(1+RS_up/RS_down)
        self.df['RSI' + str(window)] = rsi
        return self.df
    
    def sma(self, window):
        self.df['SMA' + str(window)] = self.df.iloc[:,6].rolling(window=window).mean()
        return self.df
    
    def garch(self, window):
        #TODO
        print("GARCH")

    def df_tbl(self):
        return self.df


class getGeckoIndicators:
    def __init__(self, symbol, currency, interval, days):
        self.symbol = symbol
        self.currency = currency
        self.interval = interval
        self.days = days
        #self.interval = interval
        #self.limit = limit
        self.data = []
        #self.indicator = indicator
        self.url = "https://api.coingecko.com/api/v3/coins/"+ self.symbol +"/market_chart?vs_currency=" + self.currency + "&days=" + self.days + "&interval=" + self.interval
        daily_stats = requests.get(self.url,headers=headers)
        j_response = json.loads(daily_stats.text)
        try:
            #print(j_response)
            self.j_response = j_response
            for i in j_response['prices']:
                kline_historical = {
                    'symbol': self.symbol,
                    'close' : i[1],
                    'close_time' : dt.fromtimestamp(int(i[0])/1000),
                }
                self.data.append(kline_historical)
        except Exception as err:
            print(j_response['status']['error_message'])
                
        #json to dataframe
        self.df = pd.json_normalize(self.data)
        self.df = self.df[['symbol', 'close_time','close']]
        #self.df['open'] = self.df.open.astype(float)
        #self.df['high'] = self.df.high.astype(float)
        self.df['close'] = self.df.close.astype(float)
        #print(self.df)
    
    #each indicator returns dataframes
    def rsi(self, window):
        #print(self)
        delta = self.df.close.diff()
        up_days = delta.copy()
        up_days[delta<=0]=0.0
        down_days = abs(delta.copy())
        down_days[delta>0]=0.0
        RS_up = up_days.rolling(window).mean()
        RS_down = down_days.rolling(window).mean()
        rsi= 100-100/(1+RS_up/RS_down)
        self.df['RSI' + str(window)] = rsi
        return self.df
    
    def sma(self, window):
        self.df['sma'] = self.df['close'].rolling(window=20).mean()
        return self.df
    
    def garch(self, window):
        #TODO
        print("GARCH")

    def df_tbl(self):
        return self.df


today = dt.datetime.today().strftime("%Y-%m-%d %H_%M_%S")
timestamp = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")

rfr = 0.0689

#get exchange data

r_exchange = requests.get('https://api.coingecko.com/api/v3/exchanges/kucoin')
def get_exchange(r_exchange, exchange_id, base):
    exchange = 0
    j_exchange = json.loads(r_exchange.text)
    for i in j_exchange['tickers']:
        if  i['base'] == base.upper():
            exchange = 1
            break
    return exchange

def supervised_classfication(df):
    df['Close'] = df['close']
    df = df[['Close']]

    #get percentage return
    lags = [1,2,3,4,5,6,7]
    df['Today'] = (df.Close - df.Close.shift(1))/df.Close.shift(1)
    for lag in lags:
        #df[f"Lag{lag}"] = (df.Close - df.Close.shift(lag+1))/df.Close.shift(lag+1)
        df[f"Lag{lag}"] = df.Today.shift(lag)
        #df[f"BTC Lag{lag}"] = df['BTC Today'].shift(lag)

    df['Direction'] = np.where(df['Today']>0, 1, 0)
    df['Random'] = np.random.choice([0,1],df.shape[0])
    df = df.iloc[8:]


    features = ['Lag1', 'Lag2',  'Lag3', 'Lag4', 'Lag5',  'Lag6', 'Lag7']#, 'Previous Volume/Billion']
    X = df[features]
    y = df['Direction']
    #print(X[:-1])

    #random split
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
    # X_test = X_test.sort_index(ascending=True)
    # y_test = y_test.sort_index(ascending=True)

    #split by time series
    df_n = df.shape[0]
    split_size = 0.25
    #df_X_sorted = X.sort_index(ascending=True)
    #df_y_sorted = y.sort_index(ascending=True)
    X_test, y_test = X.iloc[-int(df_n*split_size):], y.iloc[-int(df_n*split_size):]
    X_train, y_train = X.iloc[:int(df_n*(1-split_size))], y.iloc[:int(df_n*(1-split_size))]

    #Logistic Regression
    logreg = LogisticRegression().fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)
    logit_cv = LogisticRegressionCV(cv=5, random_state=0).fit(X_train, y_train)

    #summaries
    sm_logreg = sm.Logit(y.astype(float), X.astype(float)).fit()
    #print(sm_logreg.summary())
    #TODO:
    #use a previous day's close or past close values as explanatory variables for prediction



    #LDA
    #print("*****LDA*****")
    lda_test = LinearDiscriminantAnalysis().fit(X_train, y_train)
    lda_pred = lda_test.predict(X_test)
    lda_pred_proba = lda_test.predict_proba(X_test)

    #QDA
    #print("*****QDA*****")
    qda_test = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    qda_pred = qda_test.predict(X_test)
    qda_pred_proba = qda_test.predict_proba(X_test)

    #KNN
    n_neighbors = np.sqrt(len(X_train.index)).astype(int)
    #n_neighbors = 9
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    #merged output for LogReg, LDA, QDA
    recent_val = len(X_test.index)
    df_recent = df[-recent_val:]
    df_recent['Logistic'] = logreg.predict(df[features][-recent_val:])
    df_recent['Logit Proba'] = logreg.predict_proba(df[features][-recent_val:])[:,1].round(2)
    df_recent['LDA'] = lda_test.predict(df[features][-recent_val:])
    df_recent['LDA Proba'] = lda_test.predict_proba(df[features][-recent_val:])[:,1].round(2)
    df_recent['QDA'] = qda_test.predict(df[features][-recent_val:])
    df_recent['QDA Proba'] = qda_test.predict_proba(df[features][-recent_val:])[:,1].round(2)
    df_recent['KNN'] = knn.predict(df[features][-recent_val:])
    df_recent['KNN Proba'] = knn.predict_proba(df[features][-recent_val:])[:,1].round(2)
    #df_recent['Random'] = np.random.choice([0,1],recent_val)

    cv = 4
    scoring = ['accuracy', 'balanced_accuracy', 'average_precision']    
    
    return_list = {
        'logistic_regression': logreg.score(X_test,y_test),
        'lda': lda_test.score(X_test,y_test),
        'qda': qda_test.score(X_test,y_test),
        'knn': knn.score(X_test, y_test),
    }
    
    return return_list

def mysql_truncate (table):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        sql = "TRUNCATE TABLE " + table
        cursor.execute(sql)
        conn.commit()
        conn.close()
        print("Table Truncation Success")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

    # else:
    #     conn.close()

    #return False as default
    return False    


def mysql_store (dict_obj):
    try:

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        placeholders = ', '.join(['%s'] * len(dict_obj))
        columns = ', '.join(dict_obj.keys())
        sql = "INSERT INTO market_data (%s) VALUES ('%s', '%s', '%s',  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '%s', '%s', '%s', %s,  %s,  %s,  %s, %s , %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '%s', %s, '%s', %s, %s, %s, %s, '%s', '%s', '%s', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (columns, dict_obj['id'],dict_obj['name'],dict_obj['symbol'], dict_obj['current_price'],dict_obj['market_cap'],dict_obj['market_cap_rank'],dict_obj['total_volume'],dict_obj['circulating_supply'],dict_obj['total_supply'],dict_obj['max_supply'],dict_obj['ath'],dict_obj['atl'],dict_obj['roi'],dict_obj['roi_pct'],dict_obj['last_updated'],dict_obj['atl_date'],dict_obj['ath_date'],dict_obj['price_change_24h'],dict_obj['price_change_percentage_24h'],dict_obj['market_cap_change_24h'],dict_obj['market_cap_change_percentage_24h'], dict_obj['sma20'], dict_obj['sma50'], dict_obj['rsi20'], dict_obj['rsi50'], dict_obj['normalized_std'], dict_obj['normalized_std_100'], dict_obj['correlation_btc'], dict_obj['corr_30'],dict_obj['sr'],dict_obj['sr_100'],dict_obj['regression'],dict_obj['lower_ci'],dict_obj['upper_ci'], dict_obj['normal_hypothesis'],dict_obj['lower_pi'], dict_obj['upper_pi'], dict_obj['skewness'], dict_obj['r_sq'], dict_obj['regression_type'],dict_obj['sample_size'],dict_obj['date_processed'],dict_obj['percentile_positive'], dict_obj['percentile_negative'], dict_obj['percentile_half'], dict_obj['mean_return'],dict_obj['timeframe'],dict_obj['tags'], dict_obj['denomination'], dict_obj['logistic_regression'], dict_obj['lda'], dict_obj['qda'], dict_obj['knn'], dict_obj['sr_15'],dict_obj['exchange'], dict_obj['sr_7'], dict_obj['percentile_positive_weekly'], dict_obj['percentile_negative_weekly'], dict_obj['percentile_half_weekly'], dict_obj['mean_return_weekly'], dict_obj['sr_3'])
        #print(sql)
        cursor.execute(sql, dict_obj.values())
        #cursor.execute("""INSERT INTO historical_klines (symbol) values ('test') """)
        conn.commit()
        conn.close()
        print(str(dict_obj['market_cap_rank']) + ".", dict_obj['id'] + ":", "DB Insert Success")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(str(dict_obj['market_cap_rank']) + ".", dict_obj['id'] + ":", err)

    # else:
    #     conn.close()

    #return False as default
    return False


def intraday_insert (intraday_insert_obj):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        placeholders = ', '.join(['%s'] * len(intraday_insert_obj))
        columns = ', '.join(intraday_insert_obj.keys())
        sql = "INSERT INTO intraday_price (%s) VALUES ('%s', '%s', %s, '%s')" % (columns, intraday_insert_obj['symbol'],intraday_insert_obj['close_time'],intraday_insert_obj['close'], intraday_insert_obj['time'])
        cursor.execute(sql, intraday_insert_obj.values())
        #cursor.execute("""INSERT INTO historical_klines (symbol) values ('test') """)
        conn.commit()
        conn.close()
        

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    # else:
    #     conn.close()

    #return False as default
    return False


#TODO:
def regression_plot (X_test, y_test):

    return

def main():

    #BTC Data
    btc = getGeckoIndicators('bitcoin','usd','daily', '365')
    btc.df['Date'] = pd.to_datetime(btc.df['close_time']).dt.date.astype('str')
    btc.df['Close'] = btc.df['close']
    df_btc = btc.df.set_index('Date')
    df_btc = df_btc[['Close']]

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "YOUR COINGECKO API KEY"
    }

    per_page = 100
    for page in range(1, 5):
        
        crypto_list_url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=' + str(per_page) + '&page=' + str(page) + '&sparkline=false'

        r = requests.get(crypto_list_url, headers=headers)
        j_response = json.loads(r.text)

        for entry in j_response:
            time.sleep(20)
            try: 
                #initialize
                timeframe = '1d'
                coin_id = entry['id']
                currency = 'usd'
                data_interval = 'daily'
                days = '365'

                if coin_id =='bitcoin' and currency == 'btc':
                    continue
                
                indicators = getGeckoIndicators(coin_id, currency, data_interval, days)
                symbol = entry['symbol'] + currency.upper()
                
                #intrady_insert
                for row in indicators.df.values.tolist():
                    intraday_insert_obj = {
                        'symbol': row[0],
                        'close_time': row[1],
                        'close': row[2],
                        'time': timestamp
                    }
                    intraday_insert(intraday_insert_obj)

                r_categories = requests.get('https://api.coingecko.com/api/v3/coins/'+coin_id)
                r_categories_json = json.loads(r_categories.text)
                tags = json.dumps(r_categories_json['categories'])
                #categories = str(r_categories_json['categories'])

                try:
                    sma20 = indicators.sma(20).iloc[:,-1:].tail(1).values[0][0]
                    if math.isnan(sma20) == True:
                        sma20 = 'NULL'
                except:
                    sma20 = 'NULL'

                try:
                    sma50 = indicators.sma(50).iloc[:,-1:].tail(1).values[0][0]
                    if math.isnan(sma50) == True:
                        sma50 = 'NULL'
                except:
                    sma50 = 'NULL'

                try:
                    rsi20 = indicators.rsi(20).iloc[:,-1:].tail(1).values[0][0]
                    if math.isnan(rsi20) == True:
                        rsi20 = 'NULL'
                except:
                    rsi20 = 'NULL'
                
                try:
                    rsi50 = indicators.rsi(50).iloc[:,-1:].tail(1).values[0][0]
                    if math.isnan(rsi50) == True:
                        rsi50 = 'NULL'
                except:
                    rsi20 = 'NULL'
                
                # Normalized measures of spread = s/x_bar
                try:
                    normalized_std = indicators.df_tbl()['close'].std()/indicators.df_tbl()['close'].mean()
                    if math.isnan(normalized_std) == True:
                        normalized_std = 'NULL'
                except:
                    normalized_std = 'NULL'
                
                #Normalized spread: 100 days
                try:
                    normalized_std_100 = indicators.df_tbl()['close'].tail(100).std()/indicators.df_tbl()['close'].tail(100).mean()
                    if math.isnan(normalized_std_100) == True:
                        normalized_std_100 = 'NULL'
                except:
                    normalized_std_100 = 'NULL'    
                
                df = indicators.df_tbl()
                df['Date'] =  pd.to_datetime(df['close_time']).dt.date.astype('str')
                df = df.set_index('Date')

                logistic_regression = supervised_classfication(df)['logistic_regression']
                lda = supervised_classfication(df)['lda']
                if (lda > 0.9):
                    print('pause')
                qda = supervised_classfication(df)['qda']
                knn = supervised_classfication(df)['knn']          

                df = df.merge(df_btc, on='Date')

                #correlation
                try:
                    correlation_btc = df[['Close_x','Close_y']].corr()['Close_y']['Close_x']
                    if math.isnan(correlation_btc) == True:
                        correlation_btc = 'NULL'
                except:
                    correlation_btc = 'NULL'

                #last 30 days correlation
                try:
                    corr_30 = df.tail(30)[['Close_x','Close_y']].corr()['Close_y']['Close_x']
                    if math.isnan(corr_30) == True:
                        corr_30 = 'NULL'
                except:
                    corr_30 = 'NULL'

                #Sharpe ratio (historical)

                try:
                    r = (df['close'] - df['close'].shift(1))/df['close'].shift(1)
                    sr = (r.mean()-rfr)/r.std() * np.sqrt(365)
                    if math.isnan(sr) == True:
                        sr = 'NULL'            
                except:
                    sr = 'NULL'
                
                #Sharpe ratio (100 days)

                try:
                    r = (df['close'].tail(100) - df['close'].tail(100).shift(1))/df['close'].tail(100).shift(1)
                    sr_100 = (r.mean()-rfr)/r.std() * np.sqrt(365)
                    if math.isnan(sr_100) == True:
                        sr_100 = 'NULL'
                except:
                    sr_100 = 'NULL'

                try:
                    r = (df['close'].tail(15) - df['close'].tail(15).shift(1))/df['close'].tail(15).shift(1)
                    sr_15 = (r.mean()-rfr)/r.std() * np.sqrt(365)
                    if math.isnan(sr_15) == True:
                        sr_15 = 'NULL'
                except:
                    sr_15 = 'NULL'           

                try:
                    r = (df['close'].tail(7) - df['close'].tail(7).shift(1))/df['close'].tail(7).shift(1)
                    sr_7 = (r.mean()-rfr)/r.std() * np.sqrt(365)
                    if math.isnan(sr_7) == True:
                        sr_7 = 'NULL'
                except:
                    sr_7 = 'NULL'     

                try:
                    r = (df['close'].tail(3) - df['close'].tail(3).shift(1))/df['close'].tail(3).shift(1)
                    sr_3 = (r.mean()-rfr)/r.std() * np.sqrt(365)
                    if math.isnan(sr_3) == True:
                        sr_3 = 'NULL'
                except:
                    sr_3 = 'NULL'

                #train-test split
                df['ix'] = np.arange(df.shape[0])
                X = df['ix']+1
                #X = df[['ix']].shift(-1)
                y = df['close']
                df_n = df.shape[0]

                tcsv = TimeSeriesSplit(n_splits=10)
                for train_index, test_index in tcsv.split(X):
                    #print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                #regression
                try:
                    def log_func(a,b,x):
                        return a*np.log(x)+b

                    def my_sin(x, freq, amplitude, phase, offset):
                        return np.sin(x * freq + phase) * amplitude + offset

                    y_train = np.log(y_train)
                    logarithmic_reg = curve_fit(lambda t,a,b: log_func(a,b,t),  X_train,  y_train)# p0=(3.0,-10))#, bounds=(0, [np.inf, np.inf]))#,  p0=(4, 0.1))
                    log_b0 = logarithmic_reg[0][0]
                    log_b1 = logarithmic_reg[0][1]
                    x_current = X_test.tail(1)[0]
                    log_reg = np.exp(log_func(log_b0,log_b1,x_current))
                    log_pred_arr = log_func(log_b0,log_b1,X_test)
                    #log_r_sq = r2_score(y_test, log_pred_arr)
                    log_r_sq = r2_score(y, np.exp(log_func(log_b0,log_b1,X)))
                    
                    y_fitted = log_func(log_b0,log_b1,X)

                    r_sq = log_r_sq
                    y_pred = log_reg
                    y_trained = log_func(log_b0,log_b1,X_train)
                    y_pred_arr = log_pred_arr            
                except Exception as e:
                    print(e)
                    y_pred = 'NULL'
                    r_sq = 'NULL'
                
                try:
                    lower_ci = np.log(y_pred) - 1.96*(np.log(y_train).std())/np.sqrt(len(X_train.index))
                    lower_ci = np.exp(lower_ci)
                    upper_ci = np.log(y_pred) + 1.96*(np.log(y_train).std())/np.sqrt(len(X_train.index))
                    upper_ci = np.exp(upper_ci)
                except Exception as e:
                    print(e)
                    lower_ci = 'NULL'
                    upper_ci = 'NULL'
                
                #PI 95
                try:

                    t_value = stats.t.ppf(q=1-.05/2,df=len(X_train.index)-2)
                    sample_variance = np.var(X_train)
                    se = y_train.std()

                    se_pred = se*np.sqrt(1+(1/len(X_train.index))+((x_current-X_train.mean())**2)/((len(X_train.index)-1)*sample_variance))
                    
                    lower_pi = np.log(y_pred) - t_value * se_pred
                    lower_pi = np.exp(lower_pi)
                    upper_pi = np.log(y_pred) + t_value * se_pred
                    upper_pi = np.exp(upper_pi)

                except Exception as e:
                    print(e)
                    lower_pi = 'NULL'
                    upper_pi = 'NULL'

                #normal-test
                try:
                    #null hypothesis: X_train comes from a normal distribution
                    #alpha = 1e-3
                    alpha = 0.05
                    p = stats.normaltest(y_train)
                    if p.pvalue < alpha:
                        #null rejected
                        normal_hypothesis = 1
                    else:
                        #null not rejected
                        normal_hypothesis = 0
                except Exception as e:
                    print(e)
                    normal_hypothesis = 'NULL'

                #skewness
                print(symbol.upper())
                print('page: ', page)
                try:
                    skewness = (y_train.mean() - y_train.median())/y_train.std()
                except Exception as e:
                    skewness = 'NULL'

                try:
                    #df['lag1'] = df['close'].shift(1)
                    a = 95
                    b = 100-a   
                    df['return'] = (df['close'] - df['close'].shift(1))/df['close'].shift(1)
                    percentile_positive = np.percentile(df['return'].dropna(), a)
                    percentile_negative = np.percentile(df['return'].dropna(), b) 
                    percentile_half = np.percentile(df['return'].dropna(), 50)
                    mean_return = round(np.mean(df['return'].dropna()),5)
                    if mean_return == float("inf"):
                        mean_return = 'NULL'
                except Exception as e:
                    percentile_positive = 'NULL'
                    percentile_negative = 'NULL'
                    percentile_half = 'NULL'
                    mean_return = 'NULL'

                try:
                    #df['lag1'] = df['close'].shift(1)
                    a = 95
                    b = 100-a   
                    df['return'] = (df['close'] - df['close'].shift(7))/df['close'].shift(7)
                    percentile_positive_weekly = np.percentile(df['return'].dropna(), a)
                    percentile_negative_weekly = np.percentile(df['return'].dropna(), b) 
                    percentile_half_weekly = np.percentile(df['return'].dropna(), 50)
                    mean_return_weekly = round(np.mean(df['return'].dropna()),5)
                    if mean_return_weekly == float("inf"):
                        mean_return_weekly = 'NULL'
                except Exception as e:
                    percentile_positive_weekly = 'NULL'
                    percentile_negative_weekly = 'NULL'
                    percentile_half_weekly = 'NULL'
                    mean_return_weekly = 'NULL'

                plt.style.use("dark_background")
                fig, ax = plt.subplots()
                plt.title(symbol.upper() + " Prediction (r^2: " + str(round(r_sq,5)) +","+ timeframe +"), $" + str(round(y_test.tail(1)[0],3)))

                num_bars = 100
                plt.semilogy(pd.to_datetime(df.index), y, color="white")

                plt.plot(pd.to_datetime(df.index)[-len(y_pred_arr):],np.exp(y_pred_arr), color="gold", label='prediction')
                plt.plot(pd.to_datetime(df.index)[:len(y_trained)],np.exp(y_trained), color="orange", label='train set')

                #plotting ema1
                ema1_n = 20
                df_ema1 = df.close.ewm(span=ema1_n, adjust=False).mean()
                plt.plot(pd.to_datetime(df.index),df_ema1, color='midnightblue', label='ema'+str(ema1_n))

                #plotting ema2
                ema2_n = 50
                df_ema2 = df.close.ewm(span=ema2_n, adjust=False).mean()
                plt.plot(pd.to_datetime(df.index),df_ema2, color='mediumblue', label='ema'+str(ema2_n))

                #plotting ema3
                ema3_n = 100
                df_ema3 = df.close.ewm(span=ema3_n, adjust=False).mean()
                plt.plot(pd.to_datetime(df.index),df_ema3, color='blue', label='ema'+str(ema3_n)) 

                #plotting ema4
                ema4_n = 200
                df_ema4 = df.close.ewm(span=ema4_n, adjust=False).mean()
                plt.plot(pd.to_datetime(df.index),df_ema4, color='slateblue', label='ema'+str(ema4_n))    

                #plt.plot(X_test, np.exp(y_pred_arr)) #orange line
                plt.text(pd.to_datetime(df.index)[-1], np.exp(y_pred_arr.tail(1)[0]),np.exp(y_pred_arr.tail(1)[0]))

                #plt.ylim(bottom = 1)
                #plt.show()
                x_coordinates = [pd.to_datetime(df.index)[0], pd.to_datetime(df.index)[-1]]
                upper_pi_y = [upper_pi,upper_pi]
                lower_pi_y = [lower_pi,lower_pi]
                plt.plot(x_coordinates, upper_pi_y, color='red') #red line
                plt.text(pd.to_datetime(df.index)[0], upper_pi, str(upper_pi))
                plt.plot(x_coordinates, lower_pi_y, color='green') #purple line
                plt.text(pd.to_datetime(df.index)[0], lower_pi, str(lower_pi))

                plt.legend(loc='best')
                fig.autofmt_xdate()
                #plt.legend(['model', 'prediction', 'reality', 'upper prediction', 'lower prediction'], loc = "lower right")
                #plt.show()

                save_path = os.path.join(os.path.dirname(__file__), 'graphs')

                if not os.path.isdir(save_path + str(today) + '_' + timeframe):
                    os.makedirs(save_path + str(today) + '_' + timeframe)
                plt.savefig(save_path + str(today) + '_' + timeframe +'/' + symbol.upper() + '_'+ timeframe +'.png')
                plt.close('all')

            except Exception as e: 
                print(e)
                sma20 = 'NULL'
                sma50 = 'NULL'
                rsi20 = 'NULL'
                rsi50 = 'NULL'
                normalized_std = 'NULL'
                normalized_std_100 = 'NULL'
                correlation_btc = 'NULL'
                corr_30 = 'NULL'
                sr = 'NULL'
                sr_100 = 'NULL'
                y_pred = 'NULL'
                lower_ci = 'NULL'
                upper_ci = 'NULL'
                normal_hypothesis = 'NULL'
                lower_pi = 'NULL'
                upper_pi = 'NULL'
                skewness = 'NULL'
                r_sq = 'NULL'
            
            market_data = {
                'id': entry['id'],
                'name': entry['name'],
                'symbol': entry['symbol'],
                'current_price': ('NULL' if entry['current_price'] == None else entry['current_price']),
                'market_cap': ('NULL' if entry['market_cap'] == None else entry['market_cap']),
                'market_cap_rank': ('NULL' if entry['market_cap_rank'] == None else entry['market_cap_rank']),
                'total_volume': ('NULL' if entry['total_volume'] == None else entry['total_volume']),
                'circulating_supply': ('NULL' if entry['circulating_supply'] == None else entry['circulating_supply']),
                'total_supply':('NULL' if entry['total_supply'] == None else entry['total_supply']),
                'max_supply': ('NULL' if entry['max_supply'] == None else entry['max_supply']),
                'ath': ('NULL' if entry['ath'] == None else entry['ath']),
                'atl': ('NULL' if entry['atl'] == None else entry['atl']),
                'roi': ('NULL' if entry['roi'] == None else entry['roi']['times']),
                'roi_pct': ('NULL' if entry['roi'] == None else entry['roi']['percentage']),
                'last_updated': str(entry['last_updated']),
                'atl_date': ('NULL' if entry['atl_date'] == None else str(entry['atl_date'])),            
                'ath_date': ('NULL' if entry['ath_date'] == None else str(entry['ath_date'])),
                'price_change_24h': ('NULL' if entry['price_change_24h'] == None else entry['price_change_24h']),
                'price_change_percentage_24h': ('NULL' if entry['price_change_percentage_24h'] == None else entry['price_change_percentage_24h']),
                'market_cap_change_24h': ('NULL' if entry['market_cap_change_24h'] == None else entry['market_cap_change_24h']),
                'market_cap_change_percentage_24h': ('NULL' if entry['market_cap_change_percentage_24h'] == None else entry['market_cap_change_percentage_24h']),
                'sma20': sma20,
                'sma50': sma50,
                'rsi20': rsi20,
                'rsi50': rsi50,
                'normalized_std': normalized_std,
                'normalized_std_100': normalized_std_100,
                'correlation_btc': correlation_btc,
                'corr_30': corr_30,
                'sr': sr,
                'sr_100': sr_100,
                'regression': y_pred,
                'lower_ci': 'NULL',
                'upper_ci': 'NULL',
                'normal_hypothesis': normal_hypothesis,
                'lower_pi': lower_pi,
                'upper_pi': upper_pi,
                'skewness': skewness,
                'r_sq': r_sq,
                'regression_type': 'log_r_sq',
                'sample_size': len(y),
                'date_processed': timestamp,
                'percentile_positive': percentile_positive,
                'percentile_negative': percentile_negative,
                'percentile_half': percentile_half,
                'mean_return': mean_return,
                'timeframe': timeframe,
                'tags': tags,
                'denomination': 'usd',
                'logistic_regression': logistic_regression,
                'lda': lda,
                'qda': qda,
                'knn': knn,
                'sr_15': sr_15,
                'exchange': get_exchange(r_exchange,'kraken', entry['symbol']),
                'sr_7': sr_7,
                'percentile_positive_weekly': percentile_positive_weekly,
                'percentile_negative_weekly': percentile_negative_weekly,
                'percentile_half_weekly': percentile_half_weekly,
                'mean_return_weekly': mean_return_weekly,
                'sr_3': sr_3,
            }  
            print('page ', str(page))
            mysql_store(market_data)


if __name__ == "__main__":
    main()