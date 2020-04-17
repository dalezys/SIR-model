import pandas as pd
import numpy as np
import io
from datetime import timedelta, date

# get the dates in between two dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'
start_date = date(2020,1,22)
end_date = date.today()

# read the csv files and create a time index
raw_data = []
time_index = []

for day in daterange(start_date,end_date):
    url = base_url + day.strftime("%m-%d-%Y")+".csv"
    raw_data.append(pd.read_csv(url))
    time_index.append(day)

# create three dataframe
columns = ['Confirmed','Deaths','Recovered']

US_df = pd.DataFrame(index=time_index, columns=columns)
NY_df = pd.DataFrame(index=time_index, columns=columns)
NYC_df = pd.DataFrame(index=time_index, columns=columns)

for i in range(len(raw_data)):
  
  # get US data
  try:
    US_temp = raw_data[i][raw_data[i]['Country/Region']=='US']
  except:
    US_temp = raw_data[i][raw_data[i]['Country_Region']=='US']
  US_df.iloc[i]['Confirmed'] = US_temp['Confirmed'].sum()
  US_df.iloc[i]['Deaths'] = US_temp['Deaths'].sum()
  US_df.iloc[i]['Recovered'] = US_temp['Recovered'].sum()

  NY_Confirmed = 0
  NY_Deaths = 0
  NY_Recovered = 0

  # get NY state data
  for j in range(len(US_temp)):
    try:
      if 'NY' in US_temp.iloc[j]['Province/State'] or (US_temp.iloc[j]['Province/State'] == 'New York'):
        NY_Confirmed += US_temp.iloc[j]['Confirmed']
        NY_Deaths += US_temp.iloc[j]['Deaths']
        NY_Recovered += US_temp.iloc[j]['Recovered']
    except:
      NY_temp = US_temp[US_temp['Province_State'] == 'New York']
      NY_Confirmed = NY_temp['Confirmed'].sum()
      NY_Deaths = NY_temp['Deaths'].sum()
      NY_Recovered = NY_temp['Recovered'].sum()
      
  NY_df.iloc[i]['Confirmed'] = NY_Confirmed
  NY_df.iloc[i]['Deaths'] = NY_Deaths
  NY_df.iloc[i]['Recovered'] = NY_Recovered

  # get NYC data (data incomplete)
  try:
    NYC_df.iloc[i]['Confirmed'] = US_temp[US_temp['Admin2']=='New York City']['Confirmed'].sum()
    NYC_df.iloc[i]['Deaths'] = US_temp[US_temp['Admin2']=='New York City']['Deaths'].sum()
    NYC_df.iloc[i]['Recovered'] = US_temp[US_temp['Admin2']=='New York City']['Recovered'].sum()
  except:
    NYC_df.iloc[i]['Confirmed'] = 0
    NYC_df.iloc[i]['Deaths'] = 0
    NYC_df.iloc[i]['Recovered'] = 0

import seaborn as sns
import matplotlib.pyplot as plt

US_df['New Confirmed'] = US_df['Confirmed'].diff()
US_df['New Deaths'] = US_df['Deaths'].diff()
US_df['New Recovered'] = US_df['Recovered'].diff()

NY_df['New Deaths'] = NY_df['Deaths'].diff()
NY_df['New Confirmed'] = NY_df['Confirmed'].diff()
NY_df['New Recovered'] = NY_df['Recovered'].diff()

t = time_index

# to solve an initial value problem for ODEs
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def Loss(rates, infected,recovered, S_0, I_0,R_0):
  size = len(infected)
  beta, gamma = rates
  def model(t, y):
    S = y[0]
    I = y[1]
    R = y[2]
    dSdt = -beta*S*I
    dIdt = beta*S*I-gamma*I
    dRdt = gamma*I
    return [dSdt, dIdt, dRdt]
  # solve the system
  rez = solve_ivp(model, [0,size], [S_0, I_0, R_0], t_eval=np.arange(0,size,1), vectorized = True)
  # print('1', rez)
  # calculate the loss
  loss_i = np.sqrt(np.mean((rez.y[1]-infected)**2))
  # print('2',loss_i)
  loss_r = np.sqrt(np.mean((rez.y[2]-recovered)**2))
  # print('3',loss_r)

  # since infected is much larger than reduced so does its loss
  # hence their weights are slightly adjusted
  return 0.2*loss_i + 0.8*loss_r

S_US = 327200000
I_US = 10
R_US = 0

pred_lenth = 90
recovered = US_df['Recovered']
deaths = US_df['Deaths']
infected = US_df['Confirmed'] - recovered - deaths

class SIR(object):
  def __init__(self, data, Loss, pred_lenth, S_0, I_0,R_0):
    self.data = data
    self.Loss = Loss
    self.pred_lenth = pred_lenth
    self.S_0 = S_0
    self.I_0 = I_0
    self.R_0 = R_0

  def predict(self, pred_lenth, beta, gamma, infected, deaths, recovered):
    T = infected.index.values
    current = infected.index[-1]

    # extend the index
    for i in range(pred_lenth):
      current += timedelta(days=1)
      T = np.append(T, current)
    size = len(infected) + pred_lenth

    def model(t, y):
      S = y[0]
      I = y[1]
      R = y[2]
      dSdt = -beta*S*I
      dIdt = beta*S*I-gamma*I
      dRdt = gamma*I
      return [dSdt, dIdt, dRdt]

    pred = solve_ivp(model, [0, size], [self.S_0,self.I_0,self.R_0], t_eval=np.arange(0, size, 1))
    extended_infected = np.concatenate((infected.values, [None]*pred_lenth))
    extended_deaths = np.concatenate((deaths.values, [None]*pred_lenth))
    extended_recovered = np.concatenate((recovered.values, [None]*pred_lenth))
    return T, extended_infected, extended_deaths, extended_recovered, pred
    
  def train(self):
    data= self.data
    recovered = data['Recovered']
    deaths = data['Deaths']
    infected = data['Confirmed'] - recovered - deaths
    optimization = minimize(Loss, [0.001,0.001], args=(infected, recovered, self.S_0, self.I_0, self.R_0),
                            method='L-BFGS-B',bounds=[(0.0000001,0.4),(0.0000001, 0.4)])
    beta, gamma = optimization.x
    print(optimization)
    
    T, extended_infected, extended_deaths, extended_recovered, pred= self.predict(pred_lenth, beta, gamma, infected, deaths, recovered)
    
    result_df = pd.DataFrame({'Real Infected':extended_infected, 
                              'Real Deaths': extended_deaths, 
                              'Real Recovered': extended_recovered, 
                              'Predict Susceptible': pred.y[0], 
                              'Predict Infected': pred.y[1], 
                              'Predict Recovered': pred.y[2]}, 
                             index = T)
    fig, ax = plt.subplots(figsize=(15, 10))
    result_df.plot(ax=ax)
    print(f"beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
    plt.savefig('result.png')
    return beta, gamma, beta/gamma, pred.y[1][1:]-pred.y[1][:-1]

us_sir = SIR(US_df,Loss,120,S_US,I_US,R_US)
beta, gamma, r_0, I_pred = us_sir.train()
print("Your results of beat, gamma, r_0 and I_pred is:")
print(beta)
print(gamma)
print(r_0)
print(I_pred)



