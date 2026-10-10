import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure pvo_code is importable
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PVO_PATH = os.path.join(BASE, 'pvo_code', 'pvo_code')
if PVO_PATH not in sys.path:
    sys.path.insert(0, PVO_PATH)

from pvo_network import PVONetwork
from sim import SIM_ENV
from ext_pvo import UncertaintyPVONetwork, train_network

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)
MODELS_DIR = os.path.join(OUT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET = os.path.join(PVO_PATH, 'pvo_dataset.npz')

# 1) Train small Uncertainty PVO
print('Training Uncertainty PVO (small)...')
unc_model = UncertaintyPVONetwork()
unc_path = os.path.join(MODELS_DIR, 'unc_pvo.pth')
train_network(unc_model, DATASET, epochs=5, batch_size=256, out_path=unc_path)

# 2) Prepare baseline PVO path (if existing weights available, use them)
baseline_weights = os.path.join(PVO_PATH, 'pvo_model_weights.pth')
if not os.path.exists(baseline_weights):
    print('Baseline weights not found; training PVONetwork quick baseline...')
    pvom = PVONetwork()
    pvom_path = os.path.join(MODELS_DIR, 'pvobase.pth')
    # reuse train_network but adapt to PVONetwork signature
    from ext_pvo import train_network as _train
    _train(pvom, DATASET, epochs=5, batch_size=256, out_path=pvom_path)
    baseline_weights = pvom_path

# 3) Evaluation: run short episodes for three methods
NUM_EPISODES = 12
METHODS = ['no_pvo', 'pvo_baseline', 'pvo_unc']
results = {m: {'success':0, 'collision':0} for m in METHODS}

# helper to get pvo features
import torch.nn.functional as F

def get_pvo_features_from_model(model, scan_seq, action, mc_samples=8):
    model.eval()
    device = next(model.parameters()).device
    xs = torch.tensor(scan_seq, dtype=torch.float32).unsqueeze(0).to(device) # (1, seq, 42)
    xa = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
    # MC dropout: enable train mode for dropout layers
    model.train()
    preds = []
    for _ in range(mc_samples):
        with torch.no_grad():
            p = model(xs, xa).cpu().numpy().ravel()[0]
        preds.append(p)
    mean_p = float(np.mean(preds))
    std_p = float(np.std(preds))
    model.eval()
    return mean_p, std_p

# create sim instances (no rendering)
sim = SIM_ENV(render=False)

for method in METHODS:
    print('Evaluating method', method)
    for ep in range(NUM_EPISODES):
        sim.reset()
        done = False
        steps = 0
        max_steps = 200
        # simple random or heuristic controller: sample candidate actions and pick the one with lowest predicted collision prob
        while not done and steps < max_steps:
            # sample N candidate actions
            candidates = []
            for _ in range(8):
                lin = np.random.uniform(-0.6, 0.6)
                ang = np.random.uniform(-1.2, 1.2)
                candidates.append([lin, ang])
            # get current scan history approximation: use repeated current scan as seq
            scan, *_ = sim.step(0.0,0.0)
            # build a seq_len x 42 array by repeating
            seq = np.tile(np.array(scan)[:42]/8.0, (5,1))
            # evaluate candidates
            scores = []
            for a in candidates:
                if method == 'no_pvo':
                    score = 0.5 # neutral
                elif method == 'pvo_baseline':
                    # load baseline model on demand
                    if 'pvomodel' not in globals():
                        pvomodel = PVONetwork()
                        pvomodel.load_state_dict(torch.load(baseline_weights, map_location='cpu'))
                        pvomodel.eval()
                        globals()['pvomodel'] = pvomodel
                    # forward
                    with torch.no_grad():
                        p = globals()['pvomodel'](torch.tensor(seq,dtype=torch.float32).unsqueeze(0), torch.tensor(a,dtype=torch.float32).unsqueeze(0))
                        score = float(p.cpu().numpy().ravel()[0])
                else: # pvo_unc
                    if 'uncmodel' not in globals():
                        uncmodel = UncertaintyPVONetwork()
                        uncmodel.load_state_dict(torch.load(unc_path, map_location='cpu'))
                        uncmodel.eval()
                        globals()['uncmodel'] = uncmodel
                    mean_p, std_p = get_pvo_features_from_model(globals()['uncmodel'], seq, a, mc_samples=6)
                    # combine mean and uncertainty into a risk score (mean + k*std)
                    score = mean_p + 1.5 * std_p
                scores.append(score)
            # pick action with lowest score
            best_idx = int(np.argmin(scores))
            act = candidates[best_idx]
            obs = sim.step(act[0], act[1])
            # step returns tuple last element is reward; collision & goal detection from return
            _, _, collision, goal = None, None, None, None
            # sim.step returns (scan, distance, cos, sin, collision, goal, diff_rad, action, reward)
            ret = obs
            collision_flag = ret[4]
            goal_flag = ret[5]
            if collision_flag:
                results[method]['collision'] += 1
                done = True
            if goal_flag:
                results[method]['success'] += 1
                done = True
            steps += 1
    print('Method', method, 'done')

# compute rates
rates = {}
for m in METHODS:
    rates[m] = {'success_rate': results[m]['success']/NUM_EPISODES, 'collision_rate': results[m]['collision']/NUM_EPISODES}

print('Rates:', rates)

# plot
labels = ['No-PVO','PVO','PVO-UQ']
success = [rates['no_pvo']['success_rate'], rates['pvo_baseline']['success_rate'], rates['pvo_unc']['success_rate']]
collision = [rates['no_pvo']['collision_rate'], rates['pvo_baseline']['collision_rate'], rates['pvo_unc']['collision_rate']]

x = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(6,4))
plt.bar(x - width/2, success, width, label='success_rate')
plt.bar(x + width/2, collision, width, label='collision_rate')
plt.xticks(x, labels)
plt.ylim(0,1)
plt.ylabel('Rate')
plt.title('PVO method comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pvo_comparison_rates.png'))
print('Saved comparison plot to', os.path.join(OUT_DIR, 'pvo_comparison_rates.png'))

# save numeric results
np.savez(os.path.join(OUT_DIR, 'pvo_rates.npz'), rates=rates)
print('Saved numeric results')
