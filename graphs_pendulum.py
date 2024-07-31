import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
 


df1 = pd.read_csv('pendulum_td3.csv')
#df1 = pd.DataFrame(columns=['Episode','Expected_return','Runtime'])

df2 = pd.read_csv('pendulum_ppo.csv')
#df2 = pd.DataFrame(columns=['Episode','Expected_return','Runtime'])

fig, ax = plt.subplots(2,2)

df1.plot(x ='episode', y='expected_return', kind = 'line',ax=ax[0,0])
ax[0][0].set_title("TD3 Expected Return")
df1.plot(x ='episode', y='runtime', kind = 'line',ax=ax[0,1])
ax[0][1].set_title("Runtime")

df2.plot(x ='episode', y='expected_return', kind = 'line',ax=ax[1,0])
ax[1][0].set_title("PPO Expected Return")
df2.plot(x ='episode', y='runtime', kind = 'line',ax=ax[1,1])
ax[1][1].set_title("Runtime")

plt.show()