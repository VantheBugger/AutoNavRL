import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
from shapely.geometry import box
from stable_baselines3 import TD3

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from fourwis import FourWISActionMapper
from sim import SIM_ENV
from train import RobotNavEnv


RESULT_DIR = os.path.join("experiments", "paper_results")


@dataclass(frozen=True)
class PVOConfig:
    risk_weight: float = 2.8
    uncertainty_weight: float = 0.0
    switch_weight: float = 0.16
    trigger_risk: float = 0.45
    trigger_uncertainty: float = 0.55
    conservative: bool = False


METHOD_LABELS = {
    "td3_diff": "DRL-VO",
    "4wis_mm": "PVO",
    "4wis_pvo": "AVO",
    "4wis_pvo_uq": "AVO-UQ",
}


METHOD_ORDER = ["td3_diff", "4wis_mm", "4wis_pvo", "4wis_pvo_uq"]
BASE_4WIS_CONFIG = PVOConfig(risk_weight=0.0, switch_weight=0.08, trigger_risk=1.1)
PVO_CONFIG = PVOConfig(risk_weight=2.6, switch_weight=0.10, trigger_risk=0.65)
PVO_UQ_CONFIG = PVOConfig(
    risk_weight=3.6,
    uncertainty_weight=0.45,
    switch_weight=0.16,
    trigger_risk=0.55,
    trigger_uncertainty=0.55,
    conservative=True,
)


def angle_wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def scan_clearance(scan, direction, width=0.35):
    scan = np.asarray(scan, dtype=float).reshape(-1)
    scan[np.isinf(scan)] = 7.0
    n = scan.size
    center = int(round((angle_wrap(direction) + np.pi) / (2 * np.pi) * (n - 1)))
    half = max(1, int(round(width / (2 * np.pi) * n)))
    ids = [(center + i) % n for i in range(-half, half + 1)]
    return float(np.min(scan[ids]))


def goal_features(env):
    robot = np.asarray(env.env.get_robot_state(), dtype=float).reshape(-1)
    goal = np.asarray(env.robot_goal, dtype=float).reshape(-1)
    delta = goal[:2] - robot[:2]
    dist = float(np.linalg.norm(delta))
    goal_angle = angle_wrap(np.arctan2(delta[1], delta[0]) - robot[2])
    return robot, goal, dist, goal_angle


def local_risk(scan, command, conservative=False):
    vx, vy, wz = command.forward, command.lateral, command.yaw_rate
    speed = float(np.hypot(vx, vy))
    if speed < 1e-3:
        return 0.05, 0.2

    direction = float(np.arctan2(vy, vx))
    clearance = min(
        scan_clearance(scan, direction, width=0.75),
        scan_clearance(scan, direction + 0.45, width=0.45),
        scan_clearance(scan, direction - 0.45, width=0.45),
    )
    side_clearance = min(
        scan_clearance(scan, direction + np.pi / 4, width=0.55),
        scan_clearance(scan, direction - np.pi / 4, width=0.55),
    )
    required = 0.82 + 1.05 * speed + 0.25 * abs(wz)
    margin = clearance - required
    risk = 1.0 / (1.0 + np.exp(4.0 * margin))
    uncertainty = 1.0 / (1.0 + np.exp(5.0 * (side_clearance - 1.05)))
    if conservative:
        risk = min(1.0, risk + 0.35 * uncertainty)
    return float(risk), float(uncertainty)


def robot_footprint(x, y, theta, length=0.78, width=0.58):
    geom = box(-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)
    geom = affinity.rotate(geom, theta, origin=(0.0, 0.0), use_radians=True)
    return affinity.translate(geom, xoff=x, yoff=y)


def rollout_geometry_risk(env, robot, command, horizon_steps=2, dt=0.3):
    x, y, theta = float(robot[0]), float(robot[1]), float(robot[2])
    min_clearance = float("inf")
    for i in range(1, horizon_steps + 1):
        t = i * dt
        heading = theta + command.yaw_rate * t * 0.5
        wx = np.cos(heading) * command.forward - np.sin(heading) * command.lateral
        wy = np.sin(heading) * command.forward + np.cos(heading) * command.lateral
        pred_x = x + wx * t
        pred_y = y + wy * t
        pred_theta = theta + command.yaw_rate * t
        footprint = robot_footprint(pred_x, pred_y, pred_theta)
        for obj in env.env.obstacle_list:
            geom = getattr(obj, "_geometry", None)
            if geom is None:
                continue
            clearance = float(footprint.distance(geom))
            min_clearance = min(min_clearance, clearance)
            if footprint.intersects(geom):
                return 1.0
    if not np.isfinite(min_clearance):
        return 0.0
    return float(1.0 / (1.0 + np.exp(8.0 * (min_clearance - 0.12))))


def candidate_actions(goal_angle):
    actions = []
    for speed in (0.35, 0.65):
        for steer in (-0.8, 0.0, 0.8):
            actions.append(np.array([speed, steer, -0.8], dtype=np.float32))
    for speed in (-0.55, 0.55):
        actions.append(np.array([speed, 0.0, 0.0], dtype=np.float32))
    for offset in (-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2):
        direction = np.clip(angle_wrap(goal_angle + offset) / np.pi, -1.0, 1.0)
        actions.append(np.array([0.65, direction, 0.8], dtype=np.float32))
        if abs(offset) <= np.pi / 4:
            actions.append(np.array([0.25, direction, 0.8], dtype=np.float32))
    actions.append(np.array([0.0, 0.0, -0.8], dtype=np.float32))
    return actions


def select_4wis_action(env, mapper, mode, config=None):
    scan = np.asarray(env.env.get_lidar_scan()["ranges"], dtype=float)
    robot, goal, dist, goal_angle = goal_features(env)

    if config is None:
        if mode == "4wis_pvo_uq":
            config = PVO_UQ_CONFIG
        elif mode == "4wis_pvo":
            config = PVO_CONFIG
        else:
            config = BASE_4WIS_CONFIG

    base_best = (float("inf"), None, 0.0, 0.0)
    method_best = (float("inf"), None, 0.0, 0.0)
    candidate_records = []
    goal_world = (goal[:2] - robot[:2]) / max(dist, 1e-6)

    previous_mode = mapper.current_mode
    for action in candidate_actions(goal_angle):
        trial = FourWISActionMapper(
            wheelbase=mapper.wheelbase,
            track_width=mapper.track_width,
            max_speed=mapper.max_speed,
            max_lateral_speed=mapper.max_lateral_speed,
            max_yaw_rate=mapper.max_yaw_rate,
            max_steer=mapper.max_steer,
            switch_penalty=mapper.switch_penalty,
        )
        trial.current_mode = previous_mode
        command = trial.map(action)
        risk, uncertainty = local_risk(scan, command, conservative=config.conservative)
        risk = max(risk, rollout_geometry_risk(env, robot, command))
        heading = np.array([
            np.cos(robot[2]) * command.forward - np.sin(robot[2]) * command.lateral,
            np.sin(robot[2]) * command.forward + np.cos(robot[2]) * command.lateral,
        ])
        progress = float(np.dot(heading, goal_world))
        speed = float(np.hypot(command.forward, command.lateral))
        switch = 1.0 if command.mode != previous_mode else 0.0
        near_goal_bonus = -0.25 if dist < 0.7 and command.mode == previous_mode else 0.0
        low_speed_penalty = 0.6 if speed < 0.05 else max(0.0, 0.25 - speed) * 1.2
        candidate_records.append({
            "action": action,
            "risk": risk,
            "uncertainty": uncertainty,
            "progress": progress,
            "switch": switch,
            "yaw_rate": abs(command.yaw_rate),
            "low_speed_penalty": low_speed_penalty,
            "speed": speed,
        })
        for use_base, current_best in ((True, base_best), (False, method_best)):
            if use_base:
                risk_weight = BASE_4WIS_CONFIG.risk_weight
                uncertainty_weight = BASE_4WIS_CONFIG.uncertainty_weight
                switch_weight = BASE_4WIS_CONFIG.switch_weight
            else:
                risk_weight = config.risk_weight
                uncertainty_weight = config.uncertainty_weight
                switch_weight = config.switch_weight
            score = (
                -1.35 * progress
                + risk_weight * risk
                + uncertainty_weight * uncertainty
                + switch_weight * switch
                + low_speed_penalty
                + 0.08 * abs(command.yaw_rate)
                + near_goal_bonus
            )
            if score < current_best[0]:
                if use_base:
                    base_best = (score, action, risk, uncertainty)
                else:
                    method_best = (score, action, risk, uncertainty)

    if mode == "4wis_mm":
        _, action, risk, uncertainty = base_best
    else:
        safe_records = [
            r for r in candidate_records
            if r["risk"] <= config.trigger_risk
            and r["uncertainty"] <= config.trigger_uncertainty
            and r["speed"] > 0.05
        ]
        if safe_records:
            best = min(
                safe_records,
                key=lambda r: (
                    -1.75 * r["progress"]
                    + 0.35 * r["risk"]
                    + 0.20 * r["uncertainty"]
                    + config.switch_weight * r["switch"]
                    + 0.05 * r["yaw_rate"]
                    + r["low_speed_penalty"]
                ),
            )
        else:
            best = min(
                candidate_records,
                key=lambda r: (
                    -0.9 * r["progress"]
                    + 5.5 * r["risk"]
                    + 0.8 * r["uncertainty"]
                    + config.switch_weight * r["switch"]
                    + 0.08 * r["yaw_rate"]
                    + r["low_speed_penalty"]
                ),
            )
        action, risk, uncertainty = best["action"], best["risk"], best["uncertainty"]

    return action, risk, uncertainty


def evaluate_td3_diff(model_path, episodes, seed, max_steps):
    random.seed(seed)
    np.random.seed(seed)
    model = TD3.load(model_path)
    env = RobotNavEnv(render=False, vehicle_model="diff")
    rows = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        reward = 0.0
        policy_time = []
        while not done:
            start = time.perf_counter()
            action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
            policy_time.append(time.perf_counter() - start)
            obs, reward, done, info = env.step(action)
        rows.append({
            "seed": seed,
            "episode": ep,
            "success": float(info["success"]),
            "collision": float(info["collision"]),
            "timeout": float(info["time_limit_reached"]),
            "steps": float(info["steps"]),
            "distance": float(info["total_distance"]),
            "velocity": float(info["average_velocity"]),
            "mode_switches": np.nan,
            "side_parallel_ratio": np.nan,
            "ackermann_ratio": np.nan,
            "lateral_ratio": np.nan,
            "parallel_ratio": np.nan,
            "inference_ms": float(np.mean(policy_time) * 1000.0),
            "risk": np.nan,
            "uncertainty": np.nan,
        })
    plt.close("all")
    return rows


def evaluate_4wis(mode, episodes, seed, max_steps, config=None):
    random.seed(seed)
    np.random.seed(seed)
    env = SIM_ENV(world_file="robot_world_4wis.yaml", render=False)
    mapper = FourWISActionMapper(switch_penalty=0.28)
    rows = []

    for ep in range(episodes):
        mapper.reset()
        sim_data = env.reset()
        done = False
        steps = 0
        total_distance = 0.0
        total_velocity = 0.0
        last_position = np.asarray(env.env.get_robot_state(), dtype=float).reshape(-1)[:2]
        reward = 0.0
        risks = []
        uncs = []
        policy_time = []

        while not done and steps < max_steps:
            start = time.perf_counter()
            action, risk, unc = select_4wis_action(env, mapper, mode, config=config)
            command = mapper.map(action)
            policy_time.append(time.perf_counter() - start)
            sim_data = env.step(
                lin_velocity=command.forward,
                lateral_velocity=command.lateral,
                ang_velocity=command.yaw_rate,
            )
            reward = float(sim_data[-1]) - command.switch_penalty
            current_position = np.asarray(env.env.get_robot_state(), dtype=float).reshape(-1)[:2]
            total_distance += float(np.linalg.norm(current_position - last_position))
            total_velocity += float(np.linalg.norm(sim_data[7]))
            last_position = current_position
            risks.append(risk)
            uncs.append(unc)
            steps += 1
            done = bool(sim_data[4] or sim_data[5])

        success = bool(sim_data[5])
        collision = bool(sim_data[4])
        timeout = not success and not collision and steps >= max_steps
        ratios = mapper.mode_ratios()
        rows.append({
            "seed": seed,
            "episode": ep,
            "success": float(success),
            "collision": float(collision),
            "timeout": float(timeout),
            "steps": float(steps),
            "distance": float(total_distance),
            "velocity": float(total_velocity / max(steps, 1)),
            "mode_switches": float(mapper.mode_switches),
            "side_parallel_ratio": float(ratios["lateral_ratio"] + ratios["parallel_ratio"]),
            "ackermann_ratio": float(ratios["ackermann_ratio"]),
            "lateral_ratio": float(ratios["lateral_ratio"]),
            "parallel_ratio": float(ratios["parallel_ratio"]),
            "inference_ms": float(np.mean(policy_time) * 1000.0),
            "risk": float(np.mean(risks)) if risks else 0.0,
            "uncertainty": float(np.mean(uncs)) if uncs else 0.0,
        })
    plt.close("all")
    return rows


def summarize(rows):
    summary = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            summary[key] = np.nan
        else:
            summary[key] = float(np.mean(valid))
    return summary


def metric_mean_sem(per_seed_rows, key):
    values = np.asarray([r[key] for r in per_seed_rows], dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    sem = 0.0 if values.size == 1 else float(np.std(values, ddof=1) / np.sqrt(values.size))
    return float(np.mean(values)), sem


def aggregate_by_seed(rows):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    return [
        {"seed": seed, **summarize(seed_rows)}
        for seed, seed_rows in sorted(by_seed.items())
    ]


def format_rate(mean, sem):
    if np.isnan(mean):
        return "--"
    return f"{mean * 100:.1f} $\\pm$ {sem * 100:.1f}"


def format_value(mean, sem, suffix=""):
    if np.isnan(mean):
        return "--"
    return f"{mean:.2f} $\\pm$ {sem:.2f}{suffix}"


def write_latex_table(path, aggregate):
    metrics = [
        ("success", "成功率 $\\uparrow$", format_rate),
        ("collision", "碰撞率 $\\downarrow$", format_rate),
        ("timeout", "超时率 $\\downarrow$", format_rate),
        ("steps", "步数 $\\downarrow$", format_value),
        ("distance", "路径 $\\downarrow$", format_value),
        ("mode_switches", "切换次数 $\\downarrow$", format_value),
        ("side_parallel_ratio", "侧/平比例", format_rate),
        ("inference_ms", "推理耗时/ms $\\downarrow$", format_value),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        f.write("方法 & " + " & ".join(m[1] for m in metrics) + " \\\\\n")
        f.write("\\midrule\n")
        for method in METHOD_ORDER:
            row = [METHOD_LABELS[method]]
            for key, _, formatter in metrics:
                mean = aggregate[method][key]["mean"]
                sem = aggregate[method][key]["sem"]
                row.append(formatter(mean, sem))
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def save_outputs(results, seeds, episodes, max_steps):
    os.makedirs(RESULT_DIR, exist_ok=True)
    methods = METHOD_ORDER
    metric_keys = [k for k in list(next(iter(results.values()))["summary"].keys()) if k not in ("seed", "episode")]

    raw_path = os.path.join(RESULT_DIR, "paper_pvo_episode_results.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + list(next(iter(results.values()))["episodes"][0].keys()))
        writer.writeheader()
        for method in methods:
            for row in results[method]["episodes"]:
                writer.writerow({"method": method, **row})

    per_seed_path = os.path.join(RESULT_DIR, "paper_pvo_per_seed.csv")
    with open(per_seed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed"] + metric_keys)
        for method in methods:
            for row in results[method]["per_seed"]:
                writer.writerow([method, row["seed"]] + [row[k] for k in metric_keys])

    aggregate = {}
    summary_path = os.path.join(RESULT_DIR, "paper_pvo_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "label", "seeds", "episodes_per_seed", "total_episodes"] + [
            f"{key}_mean" for key in metric_keys
        ] + [
            f"{key}_sem" for key in metric_keys
        ])
        for method in methods:
            aggregate[method] = {}
            means = []
            sems = []
            for key in metric_keys:
                mean, sem = metric_mean_sem(results[method]["per_seed"], key)
                aggregate[method][key] = {"mean": mean, "sem": sem}
                means.append(mean)
                sems.append(sem)
            writer.writerow([
                method,
                METHOD_LABELS[method],
                " ".join(str(s) for s in seeds),
                episodes,
                len(seeds) * episodes,
            ] + means + sems)

    legacy_path = os.path.join(RESULT_DIR, "paper_4wis_summary.csv")
    with open(legacy_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + metric_keys)
        for method in methods:
            writer.writerow([method] + [results[method]["summary"][k] for k in metric_keys])

    np.savez(
        os.path.join(RESULT_DIR, "paper_pvo_results.npz"),
        results=results,
        aggregate=aggregate,
        seeds=np.asarray(seeds),
        episodes=episodes,
        max_steps=max_steps,
    )
    np.savez(os.path.join(RESULT_DIR, "paper_4wis_results.npz"), results=results)

    labels = [METHOD_LABELS[m] for m in methods]
    success = [aggregate[m]["success"]["mean"] * 100 for m in methods]
    collision = [aggregate[m]["collision"]["mean"] * 100 for m in methods]
    timeout = [aggregate[m]["timeout"]["mean"] * 100 for m in methods]
    success_sem = [aggregate[m]["success"]["sem"] * 100 for m in methods]
    collision_sem = [aggregate[m]["collision"]["sem"] * 100 for m in methods]
    timeout_sem = [aggregate[m]["timeout"]["sem"] * 100 for m in methods]
    x = np.arange(len(methods))
    width = 0.26
    plt.figure(figsize=(8.6, 4.4))
    plt.bar(x - width, success, width, yerr=success_sem, capsize=4, label="Success")
    plt.bar(x, collision, width, yerr=collision_sem, capsize=4, label="Collision")
    plt.bar(x + width, timeout, width, yerr=timeout_sem, capsize=4, label="Timeout")
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Rate (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "paper_pvo_main_rates.png"), dpi=200)
    plt.savefig(os.path.join(RESULT_DIR, "paper_4wis_rates.png"), dpi=200)
    plt.close()

    steps = [aggregate[m]["steps"]["mean"] for m in methods]
    distance = [aggregate[m]["distance"]["mean"] for m in methods]
    plt.figure(figsize=(8.6, 4.2))
    plt.plot(labels, steps, marker="o", label="Average steps")
    plt.plot(labels, distance, marker="s", label="Path length")
    plt.xticks(rotation=10)
    plt.ylabel("Episode metric")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "paper_pvo_efficiency.png"), dpi=200)
    plt.close()

    latex_path = os.path.join(RESULT_DIR, "paper_pvo_main_table.tex")
    write_latex_table(latex_path, aggregate)

    return summary_path


def run_threshold_sensitivity(seeds, episodes, max_steps, thresholds):
    rows = []
    for threshold in thresholds:
        config = PVOConfig(
            risk_weight=PVO_CONFIG.risk_weight,
            uncertainty_weight=0.0,
            switch_weight=PVO_CONFIG.switch_weight,
            trigger_risk=float(threshold),
        )
        for seed in seeds:
            print(f"Running 4wis_pvo threshold={threshold} seed={seed}...")
            eps = evaluate_4wis("4wis_pvo", episodes, seed, max_steps, config=config)
            summary = summarize(eps)
            rows.append({"threshold": float(threshold), "seed": seed, **summary})

    out_path = os.path.join(RESULT_DIR, "paper_pvo_threshold_sensitivity.csv")
    metric_keys = [k for k in rows[0].keys() if k not in ("threshold", "seed", "episode")]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "seed"] + metric_keys)
        for row in rows:
            writer.writerow([row["threshold"], row["seed"]] + [row[k] for k in metric_keys])

    thresholds_unique = sorted(set(r["threshold"] for r in rows))
    success_means, collision_means, timeout_means = [], [], []
    success_sem, collision_sem, timeout_sem = [], [], []
    aggregate = {}
    for threshold in thresholds_unique:
        subset = [r for r in rows if r["threshold"] == threshold]
        aggregate[threshold] = {}
        for key, means, sems in (
            ("success", success_means, success_sem),
            ("collision", collision_means, collision_sem),
            ("timeout", timeout_means, timeout_sem),
        ):
            mean, sem = metric_mean_sem(subset, key)
            aggregate[threshold][key] = {"mean": mean, "sem": sem}
            means.append(mean * 100)
            sems.append(sem * 100)

    x = np.arange(len(thresholds_unique))
    width = 0.26
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - width, success_means, width, yerr=success_sem, capsize=4, label="Success")
    plt.bar(x, collision_means, width, yerr=collision_sem, capsize=4, label="Collision")
    plt.bar(x + width, timeout_means, width, yerr=timeout_sem, capsize=4, label="Timeout")
    plt.xticks(x, [f"{t:.2f}" for t in thresholds_unique])
    plt.xlabel("PVO trigger threshold")
    plt.ylabel("Rate (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "paper_pvo_thresholds.png"), dpi=200)
    plt.close()

    latex_path = os.path.join(RESULT_DIR, "paper_pvo_threshold_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("PVO 阈值 & 成功率 $\\uparrow$ & 碰撞率 $\\downarrow$ & 超时率 $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for threshold in thresholds_unique:
            f.write(
                f"{threshold:.2f} & "
                f"{format_rate(aggregate[threshold]['success']['mean'], aggregate[threshold]['success']['sem'])} & "
                f"{format_rate(aggregate[threshold]['collision']['mean'], aggregate[threshold]['collision']['sem'])} & "
                f"{format_rate(aggregate[threshold]['timeout']['mean'], aggregate[threshold]['timeout']['sem'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=None, help="Compatibility alias for one seed.")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--td3-model", default="models/td3.zip")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--sensitivity-episodes", type=int, default=15)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55])
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else args.seeds

    results = {}
    for name in METHOD_ORDER:
        rows = []
        for seed in seeds:
            print(f"Running {name} seed={seed} episodes={args.episodes}...")
            if name == "td3_diff":
                seed_rows = evaluate_td3_diff(args.td3_model, args.episodes, seed, args.max_steps)
            elif name == "4wis_pvo":
                seed_rows = evaluate_4wis(name, args.episodes, seed, args.max_steps, config=PVO_CONFIG)
            elif name == "4wis_pvo_uq":
                seed_rows = evaluate_4wis(name, args.episodes, seed, args.max_steps, config=PVO_UQ_CONFIG)
            else:
                seed_rows = evaluate_4wis(name, args.episodes, seed, args.max_steps, config=BASE_4WIS_CONFIG)
            rows.extend(seed_rows)
        per_seed = aggregate_by_seed(rows)
        summary = summarize(rows)
        results[name] = {"episodes": rows, "per_seed": per_seed, "summary": summary}
        print(name, summary)

    csv_path = save_outputs(results, seeds, args.episodes, args.max_steps)
    print(f"Saved summary to {csv_path}")

    if not args.skip_sensitivity:
        sensitivity_path = run_threshold_sensitivity(
            seeds,
            args.sensitivity_episodes,
            args.max_steps,
            args.thresholds,
        )
        print(f"Saved threshold sensitivity to {sensitivity_path}")


if __name__ == "__main__":
    main()
