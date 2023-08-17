c=dataset.groupby(187,group_keys=False).apply(lambda dataset : dataset.sample(1))
c

plt.plot(c.iloc[0,:186])
plt.title('Normal Beat')

plt.plot(c.iloc[1,:186])
plt.title('Supraventricular Beat')


plt.plot(c.iloc[2,:186])
plt.title('Ventricular Beat')

plt.plot(c.iloc[3,:186])
plt.title('Fusion Beat')

plt.plot(c.iloc[4,:186])
plt.title('Unkonown Beat')

import numpy as np
from scipy.signal import iirfilter, freqz


tempo=c
bruiter=iirfilter(tempo, [2*np.pi*50, 2*np.pi*200], rs=50,
                        btype='band', analog=True, ftype='cheby2')
bruiter

c1=bruiter.groupby(187,group_keys=False).apply(lambda bruiter : bruiter.sample(1))

plt.plot(c1.iloc[0,:186])
plt.title('Normal Beat')

plt.plot(c1.iloc[1,:186])
plt.title('Supraventricular Beat')

plt.plot(c1.iloc[2,:186])
plt.title('Ventricular Beat')


plt.plot(c1.iloc[3,:186])
plt.title('Fusion Beat')

plt.plot(c1.iloc[4,:186])
plt.title('Unkonown Beat')
