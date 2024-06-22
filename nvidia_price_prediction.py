# Let's import selenium libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Let's import other necessary libraries
import re
import requests
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import lightgbm as lgb
from bs4 import BeautifulSoup
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from selenium.webdriver.chrome.options import Options

# Let's import warning library
import warnings as wg
wg.filterwarnings('ignore')

# Let's enter path to WebDriver executable
DRIVER_PATH = r"C:\Users\HP\OneDrive\İş masası\chromedriver-win64\chromedriver.exe"

# Let's enter URL of the webpage for scrape
URL = 'https://finance.yahoo.com/quote/NVDA/history/?guccounter=1'

# Initialize Chrome options
chrome_options = Options()
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--allow-running-insecure-content')
chrome_options.add_argument('--disable-web-security')
chrome_options.add_argument('--headless')  # Run Chrome in headless mode
chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration
chrome_options.add_argument('--log-level=3')  # Suppress logs
chrome_options.add_argument('--disable-extensions')
chrome_options.add_argument('--disable-logging')
chrome_options.add_argument('--silent')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Let's initialize the WebDriver
service = Service(DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Maximize the browser window
driver.maximize_window()

# Let's adjust the waiting time as needed based on page load times
wait = WebDriverWait(driver, 10)

# Let's navigate to the webpage
driver.get(URL)

try:
    table_container = wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@class = 'table-container svelte-ewueuo']")))

    # Let's extract the table headers (thead)
    thead = table_container.find_element(By.TAG_NAME, 'thead')
    columns = [header.text for header in thead.find_elements(By.TAG_NAME, 'th')]

    data = []

    # Let's extract the table body (tbody)
    tbody = table_container.find_element(By.TAG_NAME, 'tbody')
    rows = tbody.find_elements(By.TAG_NAME, 'tr')

    # Let's extract data from each row and append to the data list
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, 'td')
        row_data = [cell.text for cell in cells]
        data.append(row_data)

    # Let's create a Pandas DataFrame from the extracted data
    df = pd.DataFrame(data, columns=columns)

    # Let's do preprocessing over data frame
    df.drop_duplicates(subset='Date', keep='last', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    
except Exception as e:
    print(f"Error: {e}")

finally:
    driver.quit()

print('\n| -------------------------- NVIDIA Real Time Price -------------------------- |\n')
print(df.head())

price = input('\n• Which price do you want to see? (Open,High,Low,Close,Adj Close): ').capitalize()

df.to_excel(r'C:\Users\HP\OneDrive\İş masası\nvidia_project\nvidia_price.xlsx', index = False)

df = df[['Date',price]]

# Let's sorted dataset by date column
df_sorted = df.sort_values(by='Date')
df_sorted.set_index('Date', inplace = True)
df_sorted[price] = df_sorted[price].astype('float') 

print('| -------------------------- NVIDIA Real Time Dataset Visualization -------------------------- |')

print('\n• The first plot shows the general tendency, and the second plot shows the trend line.')
# Let's look at in plot
x = np.arange(len(df_sorted)) 
y = df_sorted[price].values  

# Let's visualize df_sorted trend
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# First subplot - General Tendency
axs[0].plot(df_sorted.index, df_sorted[price], label='General Tendency', marker='o', ms = 4)
axs[0].set_xlabel('Date')
axs[0].set_ylabel(f'NVIDIA {price} Price')
axs[0].set_title('General Tendency')
axs[0].legend()

# Second subplot - Trend Line
axs[1].plot(df_sorted.index, df_sorted[price], label='General Tendency', marker='o' , ms = 2)
axs[1].plot(df_sorted.index, p(x), linestyle='--', color='red', label='Trend Line')
axs[1].set_xlabel('Date')
axs[1].set_ylabel(f'NVIDIA {price} Price')
axs[1].set_title('Trend Line')
axs[1].legend()
plt.tight_layout()
plt.show()

print('\n| -------------------------- NVIDIA Real Time Prediction -------------------------- |\n')

# Let's divide train and test dataset
train = df_sorted[:int(len(df_sorted)*0.95)]
test = df_sorted[int(len(df_sorted)*0.95):]

# ### Let's create plot function for visualize plots ###
# def plot_prediction(y_pred, step):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     fig.canvas.manager.window.showMaximized()
#     train[price].plot(ax=ax, legend=True, label="TRAIN")
#     test[price].plot(ax=ax, legend=True, label="TEST")
#     y_pred.plot(ax=ax, legend=True, label="PREDICTION")
#     plt.title(f'NVIDIA Prediction next {step} days')
#     plt.show()

### Let's create function for find the best parameters ###
def tes_optimizer(train, test, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

### Let's get the best parameters ###
alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))

### Let's use ses method and predict test values ###
best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, test , abg, step= len(test))
# tes_model = ExponentialSmoothing(train, trend="mul", seasonal="mul", seasonal_periods=12).\
#             fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

tes_model = ExponentialSmoothing(df_sorted, trend="mul", seasonal="mul", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)


# advance_step = int(input('\n\n• Please enter how many days are you want to see for predict?: '))
# step = len(test) + advance_step
# y_pred = tes_model.forecast(steps=step)
# future_dates = pd.date_range(start=test.index[0] + pd.Timedelta(days=1), periods=step, freq='D')
# y_pred.index = future_dates


advance_step = int(input('\n\n• Please enter how many days are you want to see for predict?: '))
# step = len(test) + advance_step
y_pred = tes_model.forecast(steps=advance_step)
future_dates = pd.date_range(start=df_sorted.index[-1] + pd.Timedelta(days=1), periods=advance_step, freq='D')
y_pred.index = future_dates

print('\n| -------------------------- NVIDIA Prediction Visualization -------------------------- |')
print('\n• The this plot shows train test and predict value.')

### Let's create plot function for visualize plots ###
def plot_prediction(y_pred, step):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.window.showMaximized()
    df_sorted[price].plot(ax=ax, legend=True, label="CURRENT DATA")
    y_pred.plot(ax=ax, legend=True, label="PREDICTION")
    plt.title(f'NVIDIA Prediction next {step} days')
    plt.show()

### Let's create plot function for visualize plots ###
plot_prediction(y_pred, advance_step)

