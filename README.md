# DAA Assignment — BCSE204L

**Anubhab Baruah (24BPS1152)**

Six classical algorithms, implemented from scratch, applied to one problem:
finding a paying-guest (PG) room near campus and allotting students to the beds that are left.

| # | Algorithm | Complexity | Used for |
|---|-----------|------------|----------|
| 1 | KMP string matching | O(n + m) | amenity keyword search in listings |
| 2 | Dijkstra (binary heap) | O((V + E) log V) | commute time and route from campus |
| 3 | Floyd–Warshall | O(V³) | all-pairs commute table, cross-checks Dijkstra |
| 4 | 0/1 Knapsack (DP) | O(n · W) | best add-on bundle within a budget |
| 5 | Heap-based top-k | O(N log k) | best k PGs by score |
| 6 | Edmonds–Karp max flow | O(V · E²) | allot students to PGs under bed limits |

## Key results
- 0/1 Knapsack DP finds a bundle worth **40** utility points; greedy-by-ratio stops at 36.
- Max-flow places **10** students (the provable maximum); first-come greedy places 9.
- KMP is ~170× faster than naive matching on adversarial text; heap top-k is up to 17× faster than a full sort.
- Measured growth exponents match theory (Floyd–Warshall: 2.99 vs 3).

## Files
| File | What it is |
|------|-----------|
| `pg_match.py` | all six algorithms, sample data, demo and benchmarks (Python 3, standard library only) |
| `index.html` | interactive web demo — the same algorithms in JavaScript |
| `report.pdf` | project report |
| `sample_output.txt` | output of the demo run |
| `bench_results.json` | benchmark timings used in the report |

## Run
```
python pg_match.py           # demo on the sample city
python pg_match.py --bench   # demo + timing benchmarks (~1 min)
```
Open `index.html` in a browser for the interactive demo.

Sample city data (localities, rents, travel times) is synthetic and for illustration only.
