import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
 


df = pd.read_csv('iiwa_td3.csv')
#df1 = pd.DataFrame(columns=['Episode','Expected_return','Runtime'])


fig, ax = plt.subplots(2)

df.plot(x ='episode', y='expected_return', kind = 'line',ax=ax[0])
ax[0].set_title("TD3 Expected Return")
df.plot(x ='episode', y='runtime', kind = 'line',ax=ax[1])
ax[1].set_title("Runtime")

plt.show()