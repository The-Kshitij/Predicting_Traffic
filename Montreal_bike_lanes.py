import numpy as np;
import pandas as pd;

from sklearn.impute import SimpleImputer;

dataset = pd.read_csv("comptagesvelo2015.csv");

dataset = dataset.drop(dataset.columns[1], axis = 1);

x = dataset.iloc[:, 0];
y = dataset.iloc[:, 1:];

imputer = SimpleImputer(missing_values=np.nan,strategy="mean");
imputer.fit(y);
y = imputer.transform(y);

imputer = SimpleImputer(missing_values=np.inf,strategy="constant", fill_value=0);
imputer.fit(y);
y = imputer.transform(y);

import holidays;

from sklearn.preprocessing import StandardScaler;

y_scaler = StandardScaler();
y = y_scaler.fit_transform(y);

"""**Getting the holidays in canada**"""

canadian_holidays = holidays.CA();

x = x.to_numpy();

x.reshape(len(x),1);

holiday_ans = np.zeros((len(x),1));
for i in range(0, len(x)):
  if x[i] in canadian_holidays:
    holiday_ans[i] = 1;

"""# **Checking if it rained or not for dates present in x**

**Weather Condition of canada since 1940**
"""

raining_cond = pd.read_csv("Canadian_climate_history.csv");

raining_cond.columns

def ReturnMonth(m):
  months = {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04", "May":"05", "Jun":"06", "Jul":"07", "Aug":"08", "Sep":"09", "Oct":"10", "Nov":"11", "Dec":"12"};
  return months[m];
    
def RemoveTimeAndChangeSlash(d):
  cntr = 0;
  d = d.split(" ");
  d = d[0];
  s = d.split('-');
  return s[0]+"/"+ReturnMonth(s[1])+"/"+s[2];

weather_conds = [];
prc = 0;
for i in x:
  if prc==1:
    break;
  else:
    for j in range(0, len(raining_cond['LOCAL_DATE'])):
      k = RemoveTimeAndChangeSlash(raining_cond['LOCAL_DATE'][j]);
      if i==k:
        temp = raining_cond.iloc[j, :6];
        temp['LOCAL_DATE'] = k;
        weather_conds.append(temp);
        #print(weather_conds);
        #print(k);        
        break;

weather_conds_np = np.array(weather_conds);


new_x = weather_conds_np;


x_final = np.append(new_x, holiday_ans, axis = 1);


from sklearn.linear_model import LinearRegression;

from sklearn.model_selection import train_test_split;

x_final = x_final[:,1:];

x_scaler = StandardScaler();

x_final_scaled = x_scaler.fit_transform(x_final[:,:-1]);

imputer = SimpleImputer(missing_values=np.nan, strategy="mean");
x_final_scaled = imputer.fit_transform(x_final_scaled);

x_train, x_test, y_train, y_test = train_test_split(x_final_scaled, y, test_size=0.3, random_state = 1);

"""**Creating the models for individual values**"""

models = [];
print(len(y[0, :]));
for i in range(0, len(y[0, :])):
  model = LinearRegression();
  model.fit(x_train, y_train[:,i]);
  models.append(model);

outputs = [];
for i in range(0, len(y[0, :])):
  y_pred = models[i].predict(x_test);
  outputs.append(y_pred);


import matplotlib.pyplot as plt;

x_range = range(0, len(x_test));

#checking the model
plt.plot(x_range, y_test[:,6], label="Actual", color = "red");
plt.plot(x_range, outputs[6], label="Predicted", color = "blue")
plt.legend();
