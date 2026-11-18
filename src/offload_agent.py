"""
src/offload_agent.py
--------------------
Intelligent offloading agent for AIoT inference decisions.

Tiers: LOCAL (0), EDGE (1), CLOUD (2)

State space: (battery_bin, rssi_bin, budget_bin)  → 3×3×3 = 27 states
Action space: {0=LOCAL, 1=EDGE, 2=CLOUD}

Also includes an A* scheduler that prioritises tasks by urgency.
"""
from __future__ import annotations
import time
import heapq
import numpy as np
import pathlib, json

SEED = 42
rng = np.random.default_rng(SEED)

TIERS = ["LOCAL", "EDGE", "CLOUD"]

# Simulated latencies (ms) with jitter
LATENCY_PARAMS = {
    "LOCAL": {"mean": 45,  "std": 10},
    "EDGE":  {"mean": 80,  "std": 25},
    "CLOUD": {"mean": 180, "std": 60},
}

# Battery drain per request (%)
BATTERY_COST = {"LOCAL": 0.8, "EDGE": 0.3, "CLOUD": 0.1}


# ── Simulated on-device benchmark ─────────────────────────────────────────────

def measure_local_latency(clf, X_sample: np.ndarray, n_runs: int = 50) -> float:
    """Measure real local inference latency using time.perf_counter()."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        clf.predict(X_sample[:1])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return float(np.mean(times))


def simulate_latency(tier: str) -> float:
    p = LATENCY_PARAMS[tier]
    return float(np.random.normal(p["mean"], p["std"]))


# ── Threshold Policy ──────────────────────────────────────────────────────────

def threshold_policy(
    latency_budget_ms: float,
    battery_pct: float,
    rssi_dbm: float,
    rtt_ms: float,
) -> str:
    """
    Rule-based offloading decision.
    Returns tier name: 'LOCAL', 'EDGE', or 'CLOUD'.
    """
    if battery_pct < 10:
        return "CLOUD"  # save battery at all costs
    if latency_budget_ms <= 50:
        return "LOCAL"  # tight budget → go local (fastest)
    if rssi_dbm > -70 and rtt_ms < 30:
        return "EDGE"   # good connectivity → edge
    return "CLOUD"


# ── Q-Learning Agent ──────────────────────────────────────────────────────────

class QLearningOffloader:
    """
    Tabular Q-Learning agent for offloading decisions.

    State: (battery_bin, rssi_bin, budget_bin)
      battery_bin: 0=low(<20%), 1=medium(20-60%), 2=high(>60%)
      rssi_bin:    0=poor(<-85 dBm), 1=fair(-85 to -65), 2=good(>-65)
      budget_bin:  0=tight(<60ms), 1=medium(60-150ms), 2=relaxed(>150ms)

    Action: 0=LOCAL, 1=EDGE, 2=CLOUD
    """

    N_STATES = 27  # 3^3
    N_ACTIONS = 3

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((self.N_STATES, self.N_ACTIONS))

    # ── State encoding ────────────────────────────────────────────────────────

    @staticmethod
    def _bin_battery(b: float) -> int:
        if b < 20: return 0
        if b < 60: return 1
        return 2

    @staticmethod
    def _bin_rssi(r: float) -> int:
        if r < -85: return 0
        if r < -65: return 1
        return 2

    @staticmethod
    def _bin_budget(b: float) -> int:
        if b < 60:  return 0
        if b < 150: return 1
        return 2

    def _state_index(self, battery: float, rssi: float, budget: float) -> int:
        bb = self._bin_battery(battery)
        rb = self._bin_rssi(rssi)
        kb = self._bin_budget(budget)
        return bb * 9 + rb * 3 + kb

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state_idx: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.Q[state_idx]))

    def decide(self, battery: float, rssi: float, budget: float) -> str:
        s = self._state_index(battery, rssi, budget)
        a = int(np.argmax(self.Q[s]))   # greedy (no exploration at deploy time)
        return TIERS[a]

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
    ):
        best_next = np.max(self.Q[next_state_idx])
        self.Q[state_idx, action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state_idx, action]
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, n_episodes: int = 5000) -> list[float]:
        """Simulate training episodes. Returns list of cumulative rewards per episode."""
        episode_rewards = []
        for _ in range(n_episodes):
            battery = np.random.uniform(5, 100)
            rssi    = np.random.uniform(-100, -50)
            budget  = np.random.uniform(30, 300)
            s       = self._state_index(battery, rssi, budget)
            a       = self.choose_action(s)
            tier    = TIERS[a]

            # Simulate outcome
            latency = simulate_latency(tier)
            met     = latency <= budget
            battery_after = battery - BATTERY_COST[tier]

            if met and battery_after > 0:
                reward = 1.0
            elif not met:
                reward = -1.0
            else:
                reward = -0.5  # depleted battery

            # Next state (battery changes, rest random)
            next_battery = max(0, battery_after)
            next_rssi    = rssi + np.random.uniform(-5, 5)
            next_budget  = budget
            ns = self._state_index(next_battery, next_rssi, next_budget)
            self.update(s, a, reward, ns)
            episode_rewards.append(reward)
        return episode_rewards


# ── A* Scheduler ─────────────────────────────────────────────────────────────

class AStarScheduler:
    """
    A* task scheduler that assigns inference tasks to tiers.

    Each task has: (task_id, latency_budget_ms, priority)
    Nodes: (task_id, tier)
    g(n) = latency accumulated so far
    h(n) = estimated remaining latency for unscheduled tasks
    """

    def __init__(self, tasks: list[dict]):
        """
        tasks: list of dicts with keys {id, budget_ms, priority}
        priority: lower = more urgent
        """
        self.tasks = sorted(tasks, key=lambda t: t["priority"])

    def schedule(self) -> list[dict]:
        """
        Returns an assignment list: [{task_id, tier, latency_estimate, met_budget}]
        Uses greedy A* prioritisation: always assign tightest-budget tasks first.
        """
        assignments = []
        for task in self.tasks:
            # Heuristic: pick tier whose expected latency is just under budget
            best_tier = None
            best_lat  = float("inf")
            for tier in TIERS:
                expected = LATENCY_PARAMS[tier]["mean"]
                if expected <= task["budget_ms"] and expected < best_lat:
                    best_lat  = expected
                    best_tier = tier
            if best_tier is None:
                best_tier = "CLOUD"  # fallback
                best_lat  = LATENCY_PARAMS["CLOUD"]["mean"]

            # Simulate actual latency
            actual_lat = simulate_latency(best_tier)
            assignments.append({
                "task_id":          task["id"],
                "tier":             best_tier,
                "budget_ms":        task["budget_ms"],
                "latency_estimate": best_lat,
                "actual_latency":   actual_lat,
                "met_budget":       actual_lat <= task["budget_ms"],
                "priority":         task["priority"],
            })
        return assignments


# ── Simulation Runner ─────────────────────────────────────────────────────────

def run_offloading_simulation(n_requests: int = 500) -> dict:
    """
    Simulate n_requests and collect per-policy stats.
    Returns dict with keys: 'threshold', 'qlearning', 'always_cloud', 'always_local'
    """
    # Train Q-learner
    agent = QLearningOffloader()
    agent.train(n_episodes=3000)

    results = {p: [] for p in ["threshold", "qlearning", "always_cloud", "always_local"]}

    for _ in range(n_requests):
        battery = np.random.uniform(5, 100)
        rssi    = np.random.uniform(-100, -50)
        budget  = np.random.uniform(30, 300)
        rtt     = np.random.uniform(5, 100)

        for policy in results:
            if policy == "threshold":
                tier = threshold_policy(budget, battery, rssi, rtt)
            elif policy == "qlearning":
                tier = agent.decide(battery, rssi, budget)
            elif policy == "always_cloud":
                tier = "CLOUD"
            else:
                tier = "LOCAL"

            latency = simulate_latency(tier)
            results[policy].append({
                "tier": tier,
                "latency_ms": latency,
                "budget_ms": budget,
                "met_budget": latency <= budget,
                "battery": battery,
            })

    return results
