import xija
import numpy as np
import matplotlib.pyplot as plt
from Ska.Matplotlib import pointpair

start = '2012:001'
stop = '2012:344'

msid = '1pdeaat'
model_spec = 'psmc_model_spec.json'

model = xija.ThermalModel('psmc', start=start, stop=stop,
                          model_spec=model_spec)
model.make()
model.calc()

psmc = model.get_comp(msid)
resid = psmc.dvals - psmc.mvals

xscatter = np.random.uniform(-0.2, 0.2, size=len(psmc.dvals))
yscatter = np.random.uniform(-0.2, 0.2, size=len(psmc.dvals))
plt.figure(2)
plt.clf()
plt.plot(psmc.dvals + xscatter, resid + yscatter, '.', ms=1.0, alpha=1)
plt.xlabel('{} telemetry (degC)'.format(msid.upper()))
plt.ylabel('Data - Model (degC)')
plt.title('Production fit Residual vs. Data ({} - {})'.format(start, stop))

bins = np.arange(6, 57, 5.0)
r1 = []
r99 = []
ns = []
xs = []
for x0, x1 in zip(bins[:-1], bins[1:]):
    ok = (psmc.dvals >= x0) & (psmc.dvals < x1)
    val1, val99 = np.percentile(resid[ok], [1, 99])
    xs.append((x0 + x1) / 2)
    r1.append(val1)
    r99.append(val99)
    ns.append(sum(ok))

xspp = pointpair(bins[:-1], bins[1:])
r1pp = pointpair(r1)
r99pp = pointpair(r99)

plt.plot(xspp, r1pp, '-r')
plt.plot(xspp, r99pp, '-r', label='1% and 99% limits')
plt.grid()
plt.ylim(-8, 14)
plt.xlim(5, 60)

plt.plot([5, 60], [3.5, 3.5], 'g--', alpha=1, label='+/- 3.5 degC')
plt.plot([5, 60], [-3.5, -3.5], 'g--', alpha=1)
for x, n, y in zip(xs, ns, r99):
    plt.text(x, max(y + 1, 5), 'N={}'.format(n),
         rotation='vertical', va='bottom', ha='center')

plt.legend(loc='upper right')

savefig('psmc_resid_prod_fs2_{}_{}.pdf'.format(start, stop))
