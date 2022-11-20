import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# sns.set_style('whitegrid')

T = 21
XRANGE = 200

def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


gac_data = np.load('results/T={}_off_gac.npy'.format(T))
gpg_data = np.load('results/T={}_off_gpg.npy'.format(T))
wis_data = np.load('results/T={}_off_gac_wis_cv.npy'.format(T))

X = range(XRANGE)
gac_mean = np.mean(gac_data, axis=0)
gac_std = np.std(gac_data, axis=0)
gpg_mean = np.mean(gpg_data, axis=0)
gpg_std = np.std(gpg_data, axis=0)
wis_mean = np.mean(wis_data, axis=0)
wis_std = np.std(wis_data, axis=0)


fig, axe = plot.subplots(1, 1, figsize=(13, 10))
# axe.set_xlim([-50, 1100])
axe.tick_params(axis='both', which='major', labelsize=46)
font1 = {  # 'family': 'Times New Roman',
    'weight': 'bold',
    'size': 36,
    # 'alpha': 0.2,
}
font2 = {  # 'family': 'Times New Roman',
    'weight': 'bold',
    'size': 42,
    # 'alpha': 0.2,
}

plot.plot(X, gpg_mean[:XRANGE], label='off-policy GPG', linewidth=8.0)
plot.plot(X, gac_mean[:XRANGE], label='SEGAC', linewidth=8.0)
plot.plot(X, wis_mean[:XRANGE], label='SEGAC+WIS+CV', linewidth=8.0)
#
# plot.fill_between(X, gpg_mean + 0.5 * gpg_std, gpg_mean - 0.5 * gpg_std, alpha=0.5)
# plot.fill_between(X, gac_mean + 0.5 * gac_std, gac_mean - 0.5 * gac_std, alpha=0.5)
# plot.fill_between(X, wis_mean + 0.5 * wis_std, wis_mean - 0.5 * wis_std, alpha=0.5)

# plot.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.13)
plot.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.13)
axe.set_ylabel("On Time Arrival Probability", font2)
axe.set_xlabel("Episodes", font2)
axe.yaxis.set_major_formatter(FuncFormatter(to_percent))
plot.legend(prop=font1)
plot.savefig("off-policy-T{}.pdf".format(T))
plot.show()

