"""
PG-Match: a DAA-driven engine for PG / co-living discovery.
BCSE204L Design and Analysis of Algorithms - course project
Author: Anubhab Baruah

Every feature of the engine is backed by a classical algorithm, implemented
from scratch (only the standard library is used for the algorithms):

    1. KMP string matching       - amenity keyword search in listing text   O(n + m)
    2. Dijkstra (binary heap)    - commute time from a PG to campus         O((V + E) log V)
    3. Floyd-Warshall            - all-pairs commute table for landmarks    O(V^3)
    4. 0/1 Knapsack (DP)         - best add-on amenities within a budget    O(n * W)
    5. Heap-based top-k          - best k PGs out of N ranked by score      O(N log k)
    6. Edmonds-Karp max flow     - allot students to PGs with bed limits    O(V * E^2)

Run:  python pg_match.py            (demo on the sample city)
      python pg_match.py --bench    (demo + empirical complexity benchmarks)
"""

from __future__ import annotations

import heapq
import random
import sys
import time
from collections import deque

INF = float("inf")


# ---------------------------------------------------------------------------
# 1. KMP string matching
# ---------------------------------------------------------------------------
def kmp_prefix(pattern: str) -> list[int]:
    """Longest proper prefix that is also a suffix, for each prefix of pattern."""
    pi = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = pi[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        pi[i] = k
    return pi


def kmp_search(text: str, pattern: str) -> list[int]:
    """All start indices of pattern in text. O(n + m)."""
    if not pattern:
        return []
    pi, res, q = kmp_prefix(pattern), [], 0
    for i, ch in enumerate(text):
        while q > 0 and ch != pattern[q]:
            q = pi[q - 1]
        if ch == pattern[q]:
            q += 1
        if q == len(pattern):
            res.append(i - len(pattern) + 1)
            q = pi[q - 1]
    return res


def mentions(text: str, word: str) -> bool:
    """True if `word` occurs as a whole word and is not negated by a preceding 'non'.
    KMP finds candidate positions; boundary checks reject 'spACious' and 'Non AC'."""
    t, w = text.lower(), word.lower()
    for i in kmp_search(t, w):
        before = t[i - 1] if i > 0 else " "
        after = t[i + len(w)] if i + len(w) < len(t) else " "
        if before.isalnum() or after.isalnum():
            continue
        if t[max(0, i - 4):i].strip() == "non":
            continue
        return True
    return False


def naive_search(text: str, pattern: str) -> list[int]:
    """Brute-force matcher, O(n * m) worst case - used only as a baseline."""
    n, m, res = len(text), len(pattern), []
    for s in range(n - m + 1):
        j = 0
        while j < m and text[s + j] == pattern[j]:
            j += 1
        if j == m:
            res.append(s)
    return res


# ---------------------------------------------------------------------------
# 2. Dijkstra with a binary heap
# ---------------------------------------------------------------------------
def dijkstra(adj: dict, src) -> tuple[dict, dict]:
    """adj[u] = [(v, w), ...] with w >= 0. Returns (dist, parent)."""
    dist = {u: INF for u in adj}
    parent = {u: None for u in adj}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:          # stale entry (lazy deletion)
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:  # relaxation
                dist[v] = d + w
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, parent


def path_to(parent: dict, target) -> list:
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    return path[::-1]


# ---------------------------------------------------------------------------
# 3. Floyd-Warshall
# ---------------------------------------------------------------------------
def floyd_warshall(n: int, edges: list[tuple[int, int, float]]) -> list[list[float]]:
    d = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for u, v, w in edges:
        d[u][v] = min(d[u][v], w)
    for k in range(n):
        dk = d[k]
        for i in range(n):
            dik = d[i][k]
            if dik == INF:
                continue
            di = d[i]
            for j in range(n):
                if dik + dk[j] < di[j]:
                    di[j] = dik + dk[j]
    return d


# ---------------------------------------------------------------------------
# 4. 0/1 Knapsack (bottom-up DP with reconstruction)
# ---------------------------------------------------------------------------
def knapsack_01(items: list[tuple[str, int, int]], budget: int) -> tuple[int, list[str]]:
    """items = [(name, cost, utility)]. Maximise utility with total cost <= budget."""
    n = len(items)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        _, c, u = items[i - 1]
        prev, cur = dp[i - 1], dp[i]
        for w in range(budget + 1):
            cur[w] = prev[w]
            if c <= w and prev[w - c] + u > cur[w]:
                cur[w] = prev[w - c] + u
    chosen, w = [], budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(items[i - 1][0])
            w -= items[i - 1][1]
    return dp[n][budget], chosen[::-1]


# ---------------------------------------------------------------------------
# 5. Heap-based top-k
# ---------------------------------------------------------------------------
def top_k(scored: list[tuple[float, str]], k: int) -> list[tuple[float, str]]:
    """Keep a min-heap of size k; O(N log k). Returns best-first."""
    heap: list[tuple[float, str]] = []
    for item in scored:
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True)


# ---------------------------------------------------------------------------
# 6. Edmonds-Karp max flow (BFS augmenting paths)
# ---------------------------------------------------------------------------
class MaxFlow:
    def __init__(self, n: int):
        self.n = n
        self.cap: list[dict[int, int]] = [dict() for _ in range(n)]

    def add_edge(self, u: int, v: int, c: int) -> None:
        self.cap[u][v] = self.cap[u].get(v, 0) + c
        self.cap[v].setdefault(u, 0)  # residual edge

    def edmonds_karp(self, s: int, t: int) -> int:
        flow = 0
        while True:
            parent = [-1] * self.n
            parent[s] = s
            q = deque([s])
            while q and parent[t] == -1:
                u = q.popleft()
                for v, c in self.cap[u].items():
                    if c > 0 and parent[v] == -1:
                        parent[v] = u
                        q.append(v)
            if parent[t] == -1:
                return flow
            # bottleneck
            b, v = INF, t
            while v != s:
                u = parent[v]
                b = min(b, self.cap[u][v])
                v = u
            v = t
            while v != s:
                u = parent[v]
                self.cap[u][v] -= b
                self.cap[v][u] += b
                v = u
            flow += b


# ---------------------------------------------------------------------------
# Sample city (synthetic, illustrative travel times in minutes)
# ---------------------------------------------------------------------------
NODES = ["Campus", "Katpadi Jn", "Gandhi Nagar", "Sathuvachari",
         "Bagayam", "Thorapadi", "Old Town", "Bus Stand"]
ROADS = [  # undirected (u, v, minutes)
    ("Campus", "Katpadi Jn", 9), ("Campus", "Gandhi Nagar", 6),
    ("Gandhi Nagar", "Katpadi Jn", 5), ("Gandhi Nagar", "Bus Stand", 12),
    ("Katpadi Jn", "Bus Stand", 14), ("Bus Stand", "Old Town", 4),
    ("Bus Stand", "Sathuvachari", 10), ("Sathuvachari", "Bagayam", 8),
    ("Old Town", "Thorapadi", 9), ("Bagayam", "Thorapadi", 7),
    ("Old Town", "Bagayam", 11),
]

PGS = [  # name, locality, rent (Rs/month), beds free, rating, description
    ("Sri Sai Residency", "Gandhi Nagar", 7500, 3, 4.3,
     "Spacious rooms with AC, wifi and home food. 5 min walk to main road."),
    ("Comfort Stay PG", "Katpadi Jn", 6000, 2, 3.9,
     "Non AC twin sharing, wifi, laundry. Near railway station."),
    ("Nest Co-living", "Gandhi Nagar", 9800, 4, 4.6,
     "Premium co-living, AC rooms, gym, wifi, housekeeping, power backup."),
    ("Green Valley PG", "Sathuvachari", 5200, 5, 3.7,
     "Budget PG with mess, wifi, quiet locality, CCTV."),
    ("Royal Boys Hostel", "Bus Stand", 5800, 3, 3.5,
     "Triple sharing, food included, near bus stand, power backup."),
    ("Lotus Ladies PG", "Bagayam", 6800, 2, 4.4,
     "Ladies only, AC optional, wifi, CCTV, warden on site."),
    ("Urban Den", "Old Town", 7200, 2, 4.0,
     "Single rooms with attached bath, wifi, laundry, AC."),
    ("Thorapadi Homes", "Thorapadi", 4800, 4, 3.6,
     "Affordable shared rooms, mess food, water purifier."),
]

ADDONS = [  # name, monthly cost (Rs), utility score (from user survey, 1-10)
    ("AC upgrade", 1500, 8), ("Gym access", 700, 5), ("Daily housekeeping", 600, 6),
    ("Laundry", 400, 7), ("Unlimited high-speed wifi", 300, 9),
    ("Dinner plan", 1200, 7), ("Parking", 500, 3), ("Power backup", 250, 6),
]


def build_adj(nodes, roads):
    adj = {u: [] for u in nodes}
    for u, v, w in roads:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def score(pg, commute, lo_rent, hi_rent, hi_commute):
    """Weighted score out of 100: rating 40%, affordability 35%, short commute 25%."""
    _, _, rent, _, rating, _ = pg
    return round(40 * rating / 5
                 + 35 * (hi_rent - rent) / (hi_rent - lo_rent)
                 + 25 * (1 - commute / hi_commute), 2)


def demo() -> dict:
    out: dict = {}
    print("=" * 70)
    print("PG-Match demo  (synthetic city, travel times in minutes)")
    print("=" * 70)

    # --- 1. KMP keyword search
    keyword = "wifi"
    hits = [(pg[0], len(kmp_search(pg[5].lower(), keyword))) for pg in PGS]
    ac = [pg[0] for pg in PGS if mentions(pg[5], "AC")]
    print(f"\n[1] KMP: listings mentioning '{keyword}': {sum(1 for _, c in hits if c)} of {len(PGS)}")
    raw_ac = sum(len(kmp_search(pg[5].lower(), "ac")) for pg in PGS)
    print(f"    raw KMP hits for 'ac' across all text: {raw_ac} (includes 'spACious', 'Non AC')")
    print(f"    listings that really offer AC (word boundary + negation check): {ac}")
    print(f"    prefix function of 'abab': {kmp_prefix('abab')}")
    out["kmp_ac"] = ac

    # --- 2. Dijkstra from campus
    adj = build_adj(NODES, ROADS)
    dist, parent = dijkstra(adj, "Campus")
    print("\n[2] Dijkstra from Campus:")
    for node in NODES:
        print(f"    {node:<14} {dist[node]:>3} min  via {' -> '.join(path_to(parent, node))}")
    out["dijkstra"] = {n: (dist[n], path_to(parent, n)) for n in NODES}

    # --- 3. Floyd-Warshall all pairs
    idx = {n: i for i, n in enumerate(NODES)}
    edges = []
    for u, v, w in ROADS:
        edges += [(idx[u], idx[v], w), (idx[v], idx[u], w)]
    apsp = floyd_warshall(len(NODES), edges)
    agree = all(apsp[0][idx[n]] == dist[n] for n in NODES)
    print(f"\n[3] Floyd-Warshall APSP ({len(NODES)}x{len(NODES)}); row 'Campus' matches Dijkstra: {agree}")
    out["apsp"] = apsp

    # --- 5. score + top-k  (commute uses Dijkstra result)
    lo, hi = min(pg[2] for pg in PGS), max(pg[2] for pg in PGS)
    hc = max(dist[pg[1]] for pg in PGS)
    scored = [(score(pg, dist[pg[1]], lo, hi, hc), pg[0]) for pg in PGS]
    best = top_k(scored, 3)
    print("\n[5] Top-3 PGs by score (rating, rent, commute):")
    for s, name in best:
        pg = next(p for p in PGS if p[0] == name)
        print(f"    {s:6.2f}  {name:<18} Rs {pg[2]}  {dist[pg[1]]} min  {pg[4]} stars")
    out["topk"] = best
    out["scored"] = sorted(scored, reverse=True)

    # --- 4. Knapsack add-ons
    budget = 3500
    util, chosen = knapsack_01(ADDONS, budget)
    cost = sum(c for n, c, _ in ADDONS if n in chosen)
    print(f"\n[4] 0/1 Knapsack, add-on budget Rs {budget}: utility {util}, cost Rs {cost}")
    print(f"    chosen: {chosen}")
    # greedy-by-ratio counter-example check
    greedy, g_cost, g_util = [], 0, 0
    for n, c, u in sorted(ADDONS, key=lambda x: x[2] / x[1], reverse=True):
        if g_cost + c <= budget:
            greedy.append(n); g_cost += c; g_util += u
    print(f"    greedy-by-ratio would give utility {g_util} (cost Rs {g_cost})")
    out["knapsack"] = (util, chosen, cost, g_util, g_cost)

    # --- 6. Max-flow allocation
    random.seed(7)
    students = []
    for s in range(20):
        budget_s = random.choice([5000, 6000, 7000, 8000, 10000])
        max_commute = random.choice([15, 20, 25, 30])
        students.append((f"S{s + 1:02d}", budget_s, max_commute))
    S, T = 0, 1
    st_id = {st[0]: 2 + i for i, st in enumerate(students)}
    pg_id = {pg[0]: 2 + len(students) + j for j, pg in enumerate(PGS)}
    mf = MaxFlow(2 + len(students) + len(PGS))
    eligible_edges = 0
    for st in students:
        mf.add_edge(S, st_id[st[0]], 1)
        for pg in PGS:
            if pg[2] <= st[1] and dist[pg[1]] <= st[2]:
                mf.add_edge(st_id[st[0]], pg_id[pg[0]], 1)
                eligible_edges += 1
    for pg in PGS:
        mf.add_edge(pg_id[pg[0]], T, pg[3])
    flow = mf.edmonds_karp(S, T)
    alloc = {}
    for st in students:
        for pg in PGS:
            u, v = st_id[st[0]], pg_id[pg[0]]
            if v in mf.cap[u] and mf.cap[v].get(u, 0) == 1 and mf.cap[u][v] == 0:
                alloc[st[0]] = pg[0]
    total_beds = sum(pg[3] for pg in PGS)
    print(f"\n[6] Edmonds-Karp allocation: {flow} of {len(students)} students placed "
          f"({total_beds} beds, {eligible_edges} eligible student-PG pairs)")
    unplaced = [s[0] for s in students if s[0] not in alloc]
    print(f"    unplaced (no eligible PG with a free bed): {unplaced}")
    # greedy baseline: students in arrival order grab their best-rated eligible PG
    beds = {pg[0]: pg[3] for pg in PGS}
    g_placed = 0
    for st in students:
        opts = [pg for pg in PGS if pg[2] <= st[1] and dist[pg[1]] <= st[2] and beds[pg[0]] > 0]
        if opts:
            pick = max(opts, key=lambda p: p[4])
            beds[pick[0]] -= 1
            g_placed += 1
    print(f"    greedy first-come baseline places only {g_placed} of {len(students)}")
    out["flow"] = (flow, len(students), total_beds, eligible_edges, alloc, unplaced, students, g_placed)
    return out


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def _t(fn, *a, reps=3):
    best = INF
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best * 1000  # ms


def random_graph(n, avg_deg=6, seed=1):
    rnd = random.Random(seed)
    adj = {i: [] for i in range(n)}
    edges = []
    for i in range(1, n):  # spanning tree keeps it connected
        j, w = rnd.randrange(i), rnd.randint(1, 30)
        adj[i].append((j, w)); adj[j].append((i, w)); edges += [(i, j, w), (j, i, w)]
    for _ in range(n * (avg_deg // 2 - 1)):
        i, j, w = rnd.randrange(n), rnd.randrange(n), rnd.randint(1, 30)
        if i != j:
            adj[i].append((j, w)); adj[j].append((i, w)); edges += [(i, j, w), (j, i, w)]
    return adj, edges


def bench() -> dict:
    res: dict = {}
    print("\n" + "=" * 70 + "\nBenchmarks (best of 3, ms)\n" + "=" * 70)

    # APSP: repeated Dijkstra vs Floyd-Warshall
    res["apsp"] = []
    print("\nAPSP on sparse graphs (avg degree 6):  n | V x Dijkstra | Floyd-Warshall")
    for n in [50, 100, 150, 200, 250, 300]:
        adj, edges = random_graph(n)
        td = _t(lambda: [dijkstra(adj, s) for s in adj])
        tf = _t(floyd_warshall, n, edges, reps=1)
        res["apsp"].append((n, round(td, 2), round(tf, 2)))
        print(f"  {n:4d} | {td:10.2f} | {tf:10.2f}")

    # Single-source Dijkstra scaling
    res["dijkstra"] = []
    print("\nDijkstra single source:  V | E | ms")
    for n in [10_000, 20_000, 40_000, 80_000, 160_000]:
        adj, _ = random_graph(n)
        e = sum(len(v) for v in adj.values())
        td = _t(dijkstra, adj, 0)
        res["dijkstra"].append((n, e, round(td, 2)))
        print(f"  {n:7d} | {e:8d} | {td:8.2f}")

    # KMP vs naive on adversarial input
    res["kmp"] = []
    print("\nString matching, text 'a'*n, pattern 'a'*499+'b':  n | naive | KMP")
    pat = "a" * 499 + "b"
    for n in [5_000, 10_000, 20_000, 40_000]:
        text = "a" * n
        tn = _t(naive_search, text, pat, reps=1)
        tk = _t(kmp_search, text, pat)
        res["kmp"].append((n, round(tn, 2), round(tk, 2)))
        print(f"  {n:6d} | {tn:9.2f} | {tk:7.2f}")

    # Knapsack n*W
    res["knap"] = []
    print("\nKnapsack:  n | W | n*W | ms")
    rnd = random.Random(3)
    for n, W in [(50, 2000), (100, 2000), (100, 4000), (200, 4000), (200, 8000)]:
        items = [(f"i{i}", rnd.randint(50, 800), rnd.randint(1, 20)) for i in range(n)]
        tk = _t(knapsack_01, items, W, reps=1)
        res["knap"].append((n, W, n * W, round(tk, 2)))
        print(f"  {n:4d} | {W:5d} | {n * W:8d} | {tk:8.2f}")

    # Top-k vs full sort
    res["topk"] = []
    print("\nTop-10 of N:  N | heap top-k | full sort")
    for N in [100_000, 200_000, 400_000, 800_000]:
        data = [(rnd.random(), str(i)) for i in range(N)]
        th = _t(top_k, data, 10)
        ts = _t(lambda: sorted(data, reverse=True)[:10])
        res["topk"].append((N, round(th, 2), round(ts, 2)))
        print(f"  {N:7d} | {th:8.2f} | {ts:8.2f}")

    # Edmonds-Karp on random bipartite allocation
    res["flow"] = []
    print("\nEdmonds-Karp allocation:  students | PGs | edges | flow | ms")
    for ns in [200, 400, 800, 1600]:
        npg = ns // 10
        r2 = random.Random(ns)

        def run():
            mf = MaxFlow(2 + ns + npg)
            e = 0
            for i in range(ns):
                mf.add_edge(0, 2 + i, 1)
                for j in r2.sample(range(npg), 5):
                    mf.add_edge(2 + i, 2 + ns + j, 1); e += 1
            for j in range(npg):
                mf.add_edge(2 + ns + j, 1, 8)
            return mf.edmonds_karp(0, 1), e
        t0 = time.perf_counter(); f, e = run(); tf = (time.perf_counter() - t0) * 1000
        res["flow"].append((ns, npg, e, f, round(tf, 2)))
        print(f"  {ns:5d} | {npg:4d} | {e:6d} | {f:5d} | {tf:8.2f}")
    return res


if __name__ == "__main__":
    out = demo()
    if "--bench" in sys.argv:
        b = bench()
        import json
        with open("bench_results.json", "w") as fh:
            json.dump(b, fh, indent=1)
