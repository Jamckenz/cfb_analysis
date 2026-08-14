from __future__ import annotations
import numpy as np

import cfb_data as D
from cfb_model import load_schedule, load_results, game_predictions


def monte_carlo(trials: int = 10_000, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    games = load_schedule()
    load_results(games)
    preds = game_predictions(games)

    results = {}
    for team in D.FOCUS_TEAMS:
        rows = [r for r in preds if r["team"] == team]
        actual_wins = sum(r["actual_result"] for r in rows if r["played"])
        remaining_probs = np.array([r["win_prob"] for r in rows if not r["played"]])

        if len(remaining_probs) > 0:
            draws = rng.random((trials, len(remaining_probs))) < remaining_probs
            sim_wins = actual_wins + draws.sum(axis=1)
        else:
            sim_wins = np.full(trials, actual_wins)

        n_games = len(rows)
        dist = {w: float(np.mean(sim_wins == w)) for w in range(n_games + 1)}
        results[team] = {
            "mean_wins": float(sim_wins.mean()),
            "distribution": dist,
            "p_undefeated": dist.get(n_games, 0.0),
            "p_11_plus": sum(p for w, p in dist.items() if w >= 11),
            "p_10_plus": sum(p for w, p in dist.items() if w >= 10),
            "p_bowl_eligible": sum(p for w, p in dist.items() if w >= 6),
            "p_losing_season": sum(p for w, p in dist.items() if w < 6),
        }
    return results


def print_monte_carlo(trials: int = 10_000, seed: int | None = 42):
    results = monte_carlo(trials=trials, seed=seed)
    print("=" * 70)
    print(f"MONTE CARLO SEASON SIMULATION ({trials:,} trials)")
    print("=" * 70)
    for team, r in results.items():
        print(f"\n{team}  (mean {r['mean_wins']:.2f} wins)")
        print(f"  P(undefeated)     {r['p_undefeated']*100:5.1f}%")
        print(f"  P(11+ wins)       {r['p_11_plus']*100:5.1f}%")
        print(f"  P(10+ wins)       {r['p_10_plus']*100:5.1f}%")
        print(f"  P(bowl eligible)  {r['p_bowl_eligible']*100:5.1f}%")
        print(f"  P(losing season)  {r['p_losing_season']*100:5.1f}%")


def pca_clustering(k: int = 5, random_state: int = 42) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    teams = list(D.PCA_STATS.keys())
    X = np.array([
        [D.PCA_STATS[t][1], D.PCA_STATS[t][2], D.PCA_STATS[t][3], D.PCA_STATS[t][4],
         D.SOS_OWP.get(D.NAME_ALIAS.get(t, t), 0.55)]
        for t in teams
    ])
    X_adj = X.copy()
    X_adj[:, 2] = -X_adj[:, 2]
    X_adj[:, 3] = -X_adj[:, 3]

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_adj)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(Xz)

    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(scores[:, :2])
    cluster_pc1 = {c: scores[labels == c, 0].mean() for c in range(k)}
    order = sorted(cluster_pc1, key=lambda c: -cluster_pc1[c])
    relabel = {old: new + 1 for new, old in enumerate(order)}

    results = {}
    for i, t in enumerate(teams):
        results[t] = {
            "conference": D.PCA_STATS[t][0],
            "pc1": float(scores[i, 0]), "pc2": float(scores[i, 1]), "pc3": float(scores[i, 2]),
            "cluster": relabel[labels[i]],
        }
    return {
        "results": results,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "k": k,
        "scaler": scaler,
        "pca": pca,
    }


def project_onto_pca(off_ppg: float, off_ypg: float, def_ppg: float, def_ypg: float,
                      sos: float, scaler, pca) -> tuple[float, float, float]:
    x = np.array([[off_ppg, off_ypg, -def_ppg, -def_ypg, sos]])
    xz = scaler.transform(x)
    scores = pca.transform(xz)
    return float(scores[0, 0]), float(scores[0, 1]), float(scores[0, 2])


def print_pca_summary():
    out = pca_clustering()
    res = out["results"]
    print("=" * 70)
    print("POWER 4 + NOTRE DAME — PCA CLUSTERING")
    print("=" * 70)
    ev = out["explained_variance"]
    print(f"Explained variance: PC1 {ev[0]*100:.1f}%, PC2 {ev[1]*100:.1f}%, PC3 {ev[2]*100:.1f}%\n")
    for c in range(1, out["k"] + 1):
        members = sorted([t for t in res if res[t]["cluster"] == c],
                          key=lambda t: -res[t]["pc1"])
        print(f"Cluster {c} ({len(members)} teams): {', '.join(members)}")

    nd = res["Notre Dame"]
    print(f"\nNotre Dame: cluster {nd['cluster']}, PC1={nd['pc1']:.2f}, PC2={nd['pc2']:.2f}")
    c1 = [t for t in res if res[t]["cluster"] == 1]
    print(f"Cluster 1 (elite tier) PC1 range: "
          f"{min(res[t]['pc1'] for t in c1):.2f} to {max(res[t]['pc1'] for t in c1):.2f}")


if __name__ == "__main__":
    print_pca_summary()
    print()
    print_monte_carlo()
