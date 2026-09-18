import numpy as np

PARENT = "lhs_joint_ecc_chain.npz"
OUT = "lhs_slice_chain.npz"
DT_LO = 2.17
DT_HI = 2.23

z = np.load(PARENT)
dt = z["dt_h"].ravel()
m = (dt >= DT_LO) & (dt <= DT_HI)
np.savez(OUT, dep=z["dep"].ravel()[m], secosw=z["secosw"].ravel()[m],
         sesinw=z["sesinw"].ravel()[m], flat=z["flat"][m], logprob=z["logprob"].ravel()[m])
print("n=%d median=%.2f sd=%.2f" % (m.sum(), np.median(z["dep"].ravel()[m]), z["dep"].ravel()[m].std(ddof=1)))
