from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("qtagg")

from utils.paths import GRAPHICS_DIR

#######################################
"""
Cross-Decoding: Stimulus Probability Timecourse Simulation
"""

X = 0.86 * np.random.rand(3, 100)
X[0, 40:45] = 1.4 * np.sin(np.linspace(np.pi / 4, 3 * np.pi / 4, 5))
X[1, 51:57] = 1.1 * np.sin(np.linspace(np.pi / 4, 3 * np.pi / 4, 6))
X[2, 61:65] = 1.2 * np.sin(np.linspace(np.pi / 4, 3 * np.pi / 4, 4))

vmin = stats.norm.ppf(0.75)
vmax = stats.norm.ppf(0.95)

fig, ax = plt.subplots(figsize=(5, 2))
im = ax.imshow(X, vmin=vmin, vmax=vmax, cmap="hot", aspect="auto")
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
ax.tick_params(axis="both", length=0)
plt.tight_layout(rect=[0, 0, 0.93, 1], pad=0.01)
cbar_ax = fig.add_axes([0.96, 0.01, 0.03, 0.99])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_ticklabels("")
cbar.set_ticks([])
fig.savefig(Path(GRAPHICS_DIR, "Paradigm", "CrossDecoding_Sim.svg"), dpi=300)


#######################################
"""
Cross-Correlation Sequenceness: Simulation for Methods Figure

Plot 1 – forward & backward cross-correlation: damped oscillatory decay
          (~8 Hz autocorrelation structure); forward carries an extra Gaussian
          bump at ~80 ms reflecting fast sequence replay.
Plot 2 – net sequenceness (fwd − bkw), isolating the replay bump.
"""

n_lags = 40
lags_ms = np.arange(1, n_lags + 1) * 10  # 10 … 400 ms

# Decaying envelope and 16 Hz oscillation phase
env = np.exp(-lags_ms / 130.0)
theta = 0.7 * (2 * np.pi * 12.0 * lags_ms / 1000.0)

# Gaussian bump on the forward curve (replay signal ~120 ms)
bump = 0.04 * np.exp(-0.5 * ((lags_ms - 100) / 20.0) ** 2)

# fwd and bkw share the same decaying oscillatory base with a 36° phase offset.
# The DC term (0.45*env) keeps each curve positive.
# Their difference cancels the DC and oscillates around zero from the start;
# the bump creates a clear positive peak at ~120 ms on top of that oscillation.
fwd = 0.45 * env + 0.18 * env * np.cos(theta) + bump
bkw = 0.45 * env + 0.18 * env * np.cos(theta - np.pi / 8)
net = fwd - bkw  # = -0.11*env*sin(θ−π/10) + bump  →  oscillates ≈ 0 with bump

# Colors — ColorBrewer-derived, colorblind-safe
CLR_FWD = "#1a9641"  # medium blue
CLR_BKW = "#d6604d"  # warm red-orange
CLR_NET = "#4393c3"  # green
LW = 2.5


# ── Plot 1: forward + backward ────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(3.2, 2.2))

ax1.plot(lags_ms, fwd, color=CLR_FWD, lw=LW)
ax1.plot(lags_ms, bkw, color=CLR_BKW, lw=LW)
ax1.fill_between(
    lags_ms,
    fwd,
    bkw,
    where=(fwd >= bkw).tolist(),
    alpha=0.18,
    color=CLR_FWD,
    linewidth=0,
)
ax1.set_xlim(0, 410)
ax1.set_ylim(0.0, None)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_linewidth(2.5)
ax1.set(xlabel=None, ylabel=None, xticks=[], yticks=[], xticklabels=[], yticklabels=[])
fig1.tight_layout(pad=0.01)
fig1.savefig(Path(GRAPHICS_DIR, "Paradigm", "XCorr_Sim_FwdBkw.svg"), dpi=300)


# ── Plot 2: net sequenceness (fwd − bkw) ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(3.2, 2.2))

ax2.plot(lags_ms, net, color=CLR_NET, lw=LW)
ax2.axhline(0, color="#000000", lw=2.0, ls="-")
ax2.set_xlim(0, 410)
ax2.spines[["top", "right", "bottom"]].set_visible(False)
ax2.spines[["left"]].set_linewidth(2.5)
ax2.set(xlabel=None, ylabel=None, xticks=[], yticks=[], xticklabels=[], yticklabels=[])

fig2.tight_layout(pad=0.01)
fig2.savefig(Path(GRAPHICS_DIR, "Paradigm", "XCorr_Sim_Net.svg"), dpi=300)
