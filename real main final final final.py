# TINY INSTANCE VALIDATION VERSION (Real instance data)
# Based on Main model. Only data changed, constraints unchanged.
# Rewritten to CONTROL OUTPUT and ALWAYS show FINAL SUMMARY clearly.

from gurobipy import Model, GRB, quicksum
import os, json, glob, sys, atexit, re
import pandas as pd
from math import radians, sin, cos, asin, sqrt
from collections import Counter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# OUTPUT CONTROL FLAGS (تغییر اصلی: کنترل پرینت‌ها)
# =========================================================
VERBOSE = False                 # True => چاپ جزئیات مسیرها/بازدیدها/تحویل‌ها
DEBUG_MODEL = False             # True => چاپ‌های دیباگ مدل (شمارش کانسترینت‌ها و ...)
PRINT_SANITY_PASSES = False     # True => چاپ ✓ ها در sanity checks (در حالت عادی فقط FAILها چاپ می‌شود)

GUROBI_TO_TERMINAL = False      # False => خروجی solver روی ترمینال کم شود
GUROBI_LOGFILE = "gurobi.log"   # None یا مثلا "gurobi.log" برای ذخیره لاگ گورو‌بی در فایل
RUN_REPORT_PREFIX = "run_report"
RUN_REPORT_FILE = None

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

_report_stream = None

def setup_report_logging(report_file: str):
    global _report_stream
    if _report_stream is not None:
        return
    _report_stream = open(report_file, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _report_stream)
    sys.stderr = _Tee(sys.stderr, _report_stream)
    atexit.register(_report_stream.close)

def log(msg: str):
    print(msg)
    sys.stdout.flush()

def vlog(msg: str):
    if VERBOSE:
        print(msg)
        sys.stdout.flush()

def dlog(msg: str):
    if DEBUG_MODEL:
        print(msg)
        sys.stdout.flush()

# =========================================================
# Helper: Haversine
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R*c
# =========================================================
# Answer defenition
# =========================================================
def print_service_summary(model, V0, K, T, W, y, q, p, v, gamma, x=None, A=None, node_to_atm=None):
    """
    Very compact summary:
      - total objective
      - per day: served ATMs with (q,p,chosen cassette)
    """
    model.update()
    if model.SolCount == 0:
        print("No feasible solution -> no summary.")
        return

    # ---- total cost (objective) ----
    print("\n================= SERVICE SUMMARY (COMPACT) =================")
    print(f"Objective (total cost): {model.ObjVal:.2f}")

    # ---- per day served ATMs ----
    for t in T:
        served_rows = []

        for i in V0:
            for k in K:
                if y[i, k, t].X > 0.5:
                    # map to original atm id if provided
                    atm_label = node_to_atm[i] if node_to_atm else str(i)

                    qval = q[i, k, t].X
                    pval = p[i, k, t].X

                    # chosen cassette (should be exactly one if visited)
                    chosen_w = None
                    for w in W:
                        if v[i, w, k, t].X > 0.5:
                            chosen_w = w
                            break
                    gam = gamma[chosen_w] if chosen_w is not None else None

                    served_rows.append((atm_label, k, qval, pval, chosen_w, gam))

        if not served_rows:
            # خیلی خلاصه: اگر هیچ ATM سرویس نشده، یک خط
            print(f"t={t}: (no service)")
            continue

        # چاپ یک خط برای هر ATM سرویس‌شده
        # فرمت: t=.. | ATM=.. k=.. y=1 q=.. p=.. w=.. gamma=..
        for (atm_label, k, qval, pval, chosen_w, gam) in served_rows:
            print(
                f"t={t} | ATM={atm_label} | k={k} | y=1 | "
                f"q={qval:.0f} | p={pval:.0f} | w={chosen_w} | gamma={gam:.0f}"
            )

        # (اختیاری) خلاصه مسیر همان روز
        if x is not None and A is not None:
            used_edges = 0
            for k in K:
                for (i, j) in A:
                    if x[i, j, k, t].X > 1e-6:
                        used_edges += 1
            print(f"t={t} | used_edges={used_edges}")

    print("=============================================================\n")
def print_daily_routes(model, T, K, A, x, dist, node_to_atm, depot, cost_per_km, eps=1e-6):
    """
    For each day t and vehicle k:
      - prints used edges with ATM IDs
      - prints total distance (km) and route cost (€) for that day
      - tries to reconstruct ONE feasible tour order starting/ending at depot (if possible)
        Note: because edges are undirected (A has i<j only), the tour may not be unique.
    """
    model.update()
    if model.SolCount == 0:
        print("\n================= DAILY ROUTES =================")
        print("No feasible solution -> no routes.")
        print("================================================\n")
        return

    def label(n):
        if n == depot:
            return "DEPOT"
        return str(node_to_atm.get(n, n))

    from collections import Counter

    def build_adj(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        return adj

    def reconstruct_tour(edges):
        """
        Attempt to build a closed walk starting/ending at depot using all used edges once.
        Works best when the selected subgraph forms a single cycle (degree 2 for visited nodes).
        """
        if not edges:
            return None
        adj = build_adj(edges)
        if depot not in adj:
            return None

        # store multiedges counts
        cnt = Counter((min(u, v), max(u, v)) for (u, v) in edges)

        tour = [depot]
        cur = depot

        # walk until no unused incident edge is available
        for _ in range(100000):
            nxt = None
            for nb in adj.get(cur, []):
                e = (min(cur, nb), max(cur, nb))
                if cnt[e] > 0:
                    nxt = nb
                    break
            if nxt is None:
                break

            e = (min(cur, nxt), max(cur, nxt))
            cnt[e] -= 1
            tour.append(nxt)
            cur = nxt

            if cur == depot:
                # if all edges consumed, we have a closed tour
                if all(v == 0 for v in cnt.values()):
                    return tour
                # else continue (could be multiple cycles / disconnected parts)
        return None

    print("\n================= DAILY ROUTES =================")
    for t in T:
        for k in K:
            used_edges = []
            day_km = 0.0

            for (i, j) in A:
                xv = float(x[i, j, k, t].X)
                if xv > eps:
                    reps = int(round(xv))  # depot edges can be 2
                    for _ in range(reps):
                        used_edges.append((i, j))
                        day_km += float(dist[(i, j)])

            if not used_edges:
                print(f"\nDay t={t}, k={k}: (no route)")
                continue

            day_cost = day_km * float(cost_per_km)
            print(f"\nDay t={t}, k={k}:")
            print(f"  total_km   = {day_km:.3f} km")
            print(f"  route_cost = {day_cost:.3f} €")

            print("  used edges:")
            edge_counts = Counter((min(u, v), max(u, v)) for (u, v) in used_edges)
            for (u, v), ccount in sorted(edge_counts.items()):
                print(f"    {label(u)} -- {label(v)}   (x={ccount}, dist={float(dist[(u, v)]):.3f} km)")

            tour = reconstruct_tour(used_edges)
            if tour is not None:
                tour_str = " -> ".join(label(n) for n in tour)
                print(f"  reconstructed tour: {tour_str}")
            else:
                print("  reconstructed tour: (not uniquely reconstructable / not a single cycle)")
    print("================================================\n")    
# =========================================================
# Create model
# =========================================================
model = Model("ATM_IRP_Model")
model.ModelSense = GRB.MINIMIZE

# Keep solver logging active for file output, optionally hide console noise.
model.Params.OutputFlag = 1
model.Params.LogToConsole = 1 if GUROBI_TO_TERMINAL else 0
if GUROBI_LOGFILE:
    model.Params.LogFile = GUROBI_LOGFILE

# =========================================================
# PARAMETERS (Your chosen parameters)
# =========================================================
DEPOT_LAT, DEPOT_LON = 41.029455, 28.939143

gamma = {1: 300_000.0, 2:400_000.0, 3:500_000.0}
U_i_value = 750_000.0

I0_depot = 21_400_000.0
kappa_value = 5_000_000.0

h_value = 0.001
s_value = 0.01

mu = 4.0                 # minutes
speed_kmh = 25.0
theta = 8.0 * 60.0       # 8 hours -> minutes

cost_per_km = 2.5        # c_ij = 2.5 * dist_ij

# =========================================================
# Model sets
# =========================================================
depot = 0
K = [1]                      # 1 vehicle
T = list(range(1, 8))        # 7 days
Tp1 = list(range(1, 9))      # 1..8 (t+1)
W = [1, 2, 3]                # 3 cassette sizes

# =========================================================
# DATA LOADER (REAL INSTANCE)
# =========================================================
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")

atm_master_file = sorted(glob.glob(os.path.join(OUT_DIR, "*_atm_master.csv")), key=os.path.getmtime)[-1]
edges_file      = sorted(glob.glob(os.path.join(OUT_DIR, "*_edges_dist_km.csv")), key=os.path.getmtime)[-1]
r_file          = sorted(glob.glob(os.path.join(OUT_DIR, "*_r_nextweek.json")), key=os.path.getmtime)[-1]

import datetime, os

match = re.search(r"_end_(\d{8})_", os.path.basename(r_file))
end_week = match.group(1) if match else "unknown"
RUN_REPORT_FILE = f"{RUN_REPORT_PREFIX}_end_{end_week}.log"
setup_report_logging(RUN_REPORT_FILE)

print("Using atm_master_file:", atm_master_file, datetime.datetime.fromtimestamp(os.path.getmtime(atm_master_file)))
print("Using edges_file     :", edges_file,      datetime.datetime.fromtimestamp(os.path.getmtime(edges_file)))
print("Using r_file (DEMAND):", r_file,          datetime.datetime.fromtimestamp(os.path.getmtime(r_file)))
print("Using run_report_file:", RUN_REPORT_FILE)

atm_master = pd.read_csv(atm_master_file)  # columns: atm_id, lat, lon
edges_df   = pd.read_csv(edges_file)       # columns: i, j, dist_km

with open(r_file, "r", encoding="utf-8") as f:
    r_json = json.load(f)  # keys: "atm_id|t" -> avg_withdrawal

atm_ids = atm_master["atm_id"].astype(str).tolist()
N = len(atm_ids)

atm_to_node = {atm_id: idx+1 for idx, atm_id in enumerate(atm_ids)}
node_to_atm = {idx+1: atm_id for idx, atm_id in enumerate(atm_ids)}

V0 = list(range(1, N+1))
V  = [depot] + V0

# Distances dictionary dist[(i,j)] for i<j including depot edges
dist = {}

# ATM–ATM from file
for row in edges_df.itertuples(index=False):
    ai = str(row.i); aj = str(row.j)
    i = atm_to_node[ai]
    j = atm_to_node[aj]
    if i < j:
        dist[(i, j)] = float(row.dist_km)
    else:
        dist[(j, i)] = float(row.dist_km)

# Depot–ATM distances (computed)
lat_map = dict(zip(atm_master["atm_id"].astype(str), atm_master["lat"]))
lon_map = dict(zip(atm_master["atm_id"].astype(str), atm_master["lon"]))

for atm_id in atm_ids:
    j = atm_to_node[atm_id]
    dkm = haversine_km(DEPOT_LAT, DEPOT_LON, lat_map[atm_id], lon_map[atm_id])
    dist[(depot, j)] = float(dkm)  # depot=0 < j

A = sorted(dist.keys())  # arcs i<j

c = {(i, j): cost_per_km * dist[(i, j)] for (i, j) in A}
tau = {(i, j): dist[(i, j)] * 60.0 / speed_kmh for (i, j) in A}  # minutes

U = {i: U_i_value for i in V0}
h = {i: h_value for i in V0}
s = {i: s_value for i in V0}

I0 = {i: 0.0 for i in V0}
kappa = {k: kappa_value for k in K}

r = {(i, t): 0.0 for i in V0 for t in T}
for key, val in r_json.items():
    atm_id, t_str = key.split("|")
    t = int(t_str)
    if t in T:
        i = atm_to_node[str(atm_id)]
        r[(i, t)] = float(val)

r_depot = {t: 0.0 for t in T}
delta = {(i, t): 1 for i in V0 for t in T}  # allow all

# =========================================================
# Decision variables
# =========================================================
y = model.addVars(V, K, T, vtype=GRB.BINARY, name="y")                 # visit
z = model.addVars(V0, T, vtype=GRB.BINARY, name="z")                   # stockout indicator
q = model.addVars(V0, K, T, lb=0.0, vtype=GRB.CONTINUOUS, name="q")    # delivery
p = model.addVars(V0, K, T, lb=0.0, vtype=GRB.CONTINUOUS, name="p")    # pickup

I = model.addVars(V0, Tp1, lb=0.0, vtype=GRB.CONTINUOUS, name="I")     # inventory
S = model.addVars(V0, Tp1, lb=0.0, vtype=GRB.CONTINUOUS, name="S")     # lost demand
bal = model.addVars(V0, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="bal")

v = model.addVars(V0, W, K, T, vtype=GRB.BINARY, name="v")             # cassette choice

# x: integer with bounds; depot edges allow 0..2, non-depot allow 0..1
x = model.addVars(A, K, T, vtype=GRB.INTEGER, lb=0, name="x")
for (i, j) in A:
    for k in K:
        for t in T:
            if i == depot:
                x[i, j, k, t].ub = 2
            else:
                x[i, j, k, t].ub = 1

I_dep = model.addVars(Tp1, lb=0.0, vtype=GRB.CONTINUOUS, name="Idep")
model.addConstr(I_dep[1] == I0_depot, name="DepotInit")

# =========================================================
# Objective
# =========================================================
inv_cost = quicksum(h[i] * I[i, t] for i in V0 for t in Tp1)
stockout_cost = quicksum(s[i] * S[i, t] for i in V0 for t in Tp1)
route_cost = quicksum(c[(i, j)] * x[i, j, k, t] for (i, j) in A for k in K for t in T)

model.setObjective(inv_cost + stockout_cost + route_cost, GRB.MINIMIZE)

# =========================================================
# Constraints - Block 1
# =========================================================
model.addConstrs(
    (quicksum(y[i, k, t] for k in K) <= 1 for i in V0 for t in T),
    name="ServeOnce"
)

model.addConstrs(
    (q[i, k, t] <= U[i] * y[i, k, t] for i in V0 for k in K for t in T),
    name="DeliverIfVisit"
)

model.addConstrs(
    (y[i, k, t] <= delta[i, t] for i in V0 for k in K for t in T),
    name="VisitWindow"
)

# =========================================================
# Constraints - Block 2 (link x to y)
# =========================================================
model.addConstrs(
    (x[i, j, k, t] <= y[i, k, t] for (i, j) in A if i != depot for k in K for t in T),
    name="EdgeImpliesVisit_i"
)

model.addConstrs(
    (x[i, j, k, t] <= y[j, k, t] for (i, j) in A if i != depot for k in K for t in T),
    name="EdgeImpliesVisit_j"
)

model.addConstrs(
    (x[depot, j, k, t] <= 2 * y[j, k, t] for (i, j) in A if i == depot for k in K for t in T),
    name="DepotEdgeImpliesVisit"
)

# =========================================================
# Constraints - Block 3 (capacity + time)
# =========================================================
model.addConstrs(
    (quicksum(q[i, k, t] for i in V0) <= kappa[k] * y[depot, k, t] for k in K for t in T),
    name="VehicleCapacity"
)

model.addConstrs(
    (
        quicksum(tau[(i, j)] * x[i, j, k, t] for (i, j) in A)
        + quicksum(mu * y[i, k, t] for i in V)
        <= theta
        for k in K for t in T
    ),
    name="RouteDuration"
)

# =========================================================
# Constraints - Block 4 (degree constraints)
# =========================================================
model.addConstrs(
    (
        quicksum(x[i, j, k, t] for (i, j) in A if i == node or j == node)
        ==
        2 * y[node, k, t]
        for node in V for k in K for t in T
    ),
    name="Degree"
)

# =========================================================
# Subtour elimination callback (SEC)
# =========================================================
def _connected_components(nodes, edges):
    adj = {n: set() for n in nodes}
    for u, w in edges:
        adj[u].add(w)
        adj[w].add(u)

    seen = set()
    comps = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp = []
        seen.add(n)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    return comps

def subtour_cb(cb_model, where):
    if where != GRB.Callback.MIPSOL:
        return

    x_sol = cb_model.cbGetSolution(cb_model._x)

    V0_cb = cb_model._V0
    K_cb  = cb_model._K
    T_cb  = cb_model._T
    A_cb  = cb_model._A
    depot_cb = cb_model._depot

    for k in K_cb:
        for t in T_cb:
            chosen = []
            for (i, j) in A_cb:
                if i == depot_cb:
                    continue
                if x_sol[i, j, k, t] > 0.5:
                    chosen.append((i, j))

            comps = _connected_components(V0_cb, chosen)

            for C in comps:
                if len(C) <= 1:
                    continue
                if len(C) < len(V0_cb):
                    m = C[0]
                    lhs = quicksum(
                        cb_model._x[i, j, k, t]
                        for (i, j) in A_cb
                        if (i in C and j in C and i != depot_cb)
                    )
                    rhs = quicksum(cb_model._y[i, k, t] for i in C) - cb_model._y[m, k, t]
                    cb_model.cbLazy(lhs <= rhs)

# =========================================================
# Depot balance
# =========================================================
model.addConstrs(
    (
        I_dep[t + 1]
        ==
        I_dep[t]
        + r_depot[t]
        - quicksum(q[i, k, t] for i in V0 for k in K)
        + quicksum(p[i, k, t] for i in V0 for k in K)
        for t in T
    ),
    name="DepotBalance"
)

# =========================================================
# Stockout indicator with z
# =========================================================
model.addConstrs(
    (
        I[i, t] - r[i, t] + quicksum(q[i, k, t] for k in K) - quicksum(p[i, k, t] for k in K)
        <= U[i] * (1 - z[i, t])
        for i in V0 for t in T
    ),
    name="StockoutLogic1"
)

model.addConstrs(
    (
        -I[i, t] + r[i, t] - quicksum(q[i, k, t] for k in K) + quicksum(p[i, k, t] for k in K)
        <= U[i] * z[i, t]
        for i in V0 for t in T
    ),
    name="StockoutLogic2"
)

# =========================================================
# Inventory dynamics via balance
# =========================================================
model.addConstrs((I[i, 1] == I0[i] for i in V0), name="ATMInit")

model.addConstrs(
    (
        bal[i, t]
        ==
        I[i, t]
        - r[i, t]
        + quicksum(q[i, k, t] for k in K)
        - quicksum(p[i, k, t] for k in K)
        for i in V0 for t in T
    ),
    name="BalanceDef"
)

model.addConstrs((I[i, t + 1] >= bal[i, t] for i in V0 for t in T), name="InvGEbal")
model.addConstrs((S[i, t + 1] >= -bal[i, t] for i in V0 for t in T), name="ShortGEnegBal")

# =========================================================
# Linearize p = y * I
# =========================================================
model.addConstrs((p[i, k, t] <= I[i, t] for i in V0 for k in K for t in T), name="P_le_I")
model.addConstrs((p[i, k, t] <= U[i] * y[i, k, t] for i in V0 for k in K for t in T), name="P_le_Uy")
model.addConstrs(
    (p[i, k, t] >= I[i, t] - U[i] * (1 - y[i, k, t]) for i in V0 for k in K for t in T),
    name="P_ge_IminusU1y"
)

# =========================================================
# Link q to cassette choice v
# =========================================================
model.addConstrs(
    (quicksum(v[i, w, k, t] for w in W) == y[i, k, t] for i in V0 for k in K for t in T),
    name="OneCassetteIfVisit"
)

model.addConstrs(
    (q[i, k, t] == quicksum(gamma[w] * v[i, w, k, t] for w in W) for i in V0 for k in K for t in T),
    name="DeliveryFromCassette"
)

# =========================================================
# ATM capacity bounds
# =========================================================
model.addConstrs((I[i, t] <= U[i] for i in V0 for t in Tp1), name="InvCap")

# Tighten max: I - S = balance
model.addConstrs((I[i, t + 1] - S[i, t + 1] == bal[i, t] for i in V0 for t in T), name="InvShortBalance")

# ATM capacity after service
model.addConstrs(
    (
        I[i, t]
        + quicksum(q[i, k, t] for k in K)
        - quicksum(p[i, k, t] for k in K)
        <= U[i]
        for i in V0 for t in T
    ),
    name="ATMCapaAfterService"
)

model.update()

# =========================================================
# Debug prints (کنترل‌شده با DEBUG_MODEL)
# =========================================================
def print_constraint_prefix_counts(m):
    m.update()
    constrs = m.getConstrs()
    dlog("\n========== CONSTRAINT PREFIX COUNTS ==========")
    dlog(f"Total constraints: {len(constrs)}\n")

    prefixes = []
    for cst in constrs:
        name = cst.ConstrName
        prefix = name.split('[')[0]
        prefixes.append(prefix)

    counts = Counter(prefixes)
    for prefix, cnt in sorted(counts.items()):
        dlog(f"{prefix:30s} : {cnt}")

    dlog("==============================================\n")

def print_obj_coeffs_for_prefix(m, prefix, max_show=15):
    m.update()
    hits = []
    for var in m.getVars():
        if var.VarName.startswith(prefix):
            hits.append((var.VarName, var.Obj))
    hits.sort(key=lambda x: abs(x[1]), reverse=True)

    dlog(f"\n=== Objective coefficients for variables starting with '{prefix}' ===")
    shown = 0
    for name, obj in hits:
        if abs(obj) > 1e-12:
            dlog(f"{name:30s}  ObjCoeff={obj}")
            shown += 1
            if shown >= max_show:
                break
    if shown == 0:
        dlog(f"None of the '{prefix}*' variables have non-zero objective coefficients.")
    dlog("==============================================================\n")

if DEBUG_MODEL:
    print_constraint_prefix_counts(model)
    print_obj_coeffs_for_prefix(model, "S", max_show=15)
    print_obj_coeffs_for_prefix(model, "z", max_show=15)
    print_obj_coeffs_for_prefix(model, "y", max_show=15)
    print_obj_coeffs_for_prefix(model, "x", max_show=15)

# =========================================================
# Attach for callback + enable LazyConstraints
# =========================================================
model._x = x
model._y = y
model._V0 = V0
model._K = K
model._T = T
model._A = A
model._depot = depot

model.Params.LazyConstraints = 1

# ---- solver controls ----
model.Params.TimeLimit = 3600
model.Params.MIPGap = 0.10
model.Params.MIPFocus = 2
model.Params.Heuristics = 0.15
model.Params.Cuts = 2

log("=== START OPTIMIZE ===")
model.optimize(subtour_cb)
log("=== END OPTIMIZE ===")

# =========================================================
# POST-SOLVE ANALYSIS (کم‌حجم + فلگ‌دار)
# =========================================================
log("\n================= POST-SOLVE ANALYSIS =================")

status = model.Status
status_map = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.SUBOPTIMAL: "SUBOPTIMAL (feasible solution found)",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
}
log("Status: " + status_map.get(status, f"Status code: {status}"))

has_incumbent = model.SolCount > 0

if not has_incumbent:
    log("No feasible incumbent available. Skipping solution analysis.")
    log("====================================================\n")
    passed = False  # sanity checks not applicable
else:
    log(f"Objective value: {model.ObjVal}")

    # ---- Compact summary counts (always) ----
    total_visits = 0
    for t in T:
        for i in V0:
            for k in K:
                if y[i, k, t].X > 0.5:
                    total_visits += 1

    total_used_edges = 0
    for t in T:
        for k in K:
            for (i, j) in A:
                if x[i, j, k, t].X > 1e-6:
                    total_used_edges += 1

    log(f"Summary: total visits={total_visits}, total used-edges={total_used_edges}")

    # ---- Verbose details (only if VERBOSE=True) ----
    if VERBOSE:
        print("\n-- Selected edges (x) --")
        for t in T:
            for k in K:
                for (i, j) in A:
                    valx = x[i, j, k, t].X
                    if abs(valx) > 1e-6:
                        print(f"t={t} k={k}: edge {i}-{j}  x={valx:.0f}")

        print("\n-- Visits y[i,k,t] --")
        for t in T:
            for i in V0:
                for k in K:
                    if y[i, k, t].X > 0.5:
                        print(f"t={t}  k={k}  visit ATM {i}")

        print("\n-- Deliveries q[i,k,t] --")
        for t in T:
            for i in V0:
                for k in K:
                    valq = q[i, k, t].X
                    if abs(valq) > 1e-6:
                        print(f"t={t}  k={k}  ATM {i}: q={valq:.2f}")

        print("\n-- Withdrawals p[i,k,t] --")
        for t in T:
            for i in V0:
                for k in K:
                    valp = p[i, k, t].X
                    if abs(valp) > 1e-6:
                        print(f"t={t}  k={k}  ATM {i}: p={valp:.2f}")

    # ---- Inventory / Shortage summary (always compact) ----
    total_short = 0.0
    max_short = 0.0
    max_short_at = None
    cnt_short_events = 0

    total_inv = 0.0
    max_inv = 0.0
    max_inv_at = None

    for i in V0:
        for tp in Tp1:
            Iv = float(I[i, tp].X)
            Sv = float(S[i, tp].X)

            total_inv += Iv
            if Iv > max_inv:
                max_inv = Iv
                max_inv_at = (i, tp)

            total_short += Sv
            if Sv > 1e-6:
                cnt_short_events += 1
            if Sv > max_short:
                max_short = Sv
                max_short_at = (i, tp)

    log("\n-- Inventory/Shortage Summary --")
    log(f"Total inventory sum(I): {total_inv:.2f}")
    if max_inv_at:
        log(f"Max inventory: {max_inv:.2f} at ATM {max_inv_at[0]} t={max_inv_at[1]}")
    log(f"Total shortage sum(S): {total_short:.2f}")
    log(f"Shortage events count: {cnt_short_events}")
    if max_short_at:
        log(f"Max shortage: {max_short:.2f} at ATM {max_short_at[0]} t={max_short_at[1]}")

    # =========================================================
    # SANITY CHECKS (پیش‌فرض: فقط FAILها)
    # =========================================================
    eps = 1e-6
    passed = True

    def X(var):
        try:
            return float(var.X)
        except Exception:
            return None

    def report_check(ok: bool, ok_msg: str, fail_msg: str):
        nonlocal_passed = None  # فقط برای خوانایی (پایتون اجازه nonlocal برای passed در این سطح نمی‌دهد)
        global passed
        if ok:
            if PRINT_SANITY_PASSES:
                print(ok_msg)
            return True
        else:
            print(fail_msg)
            passed = False
            return False

    print("\n================= SANITY CHECKS =================")

    # 1) Inventory balance consistency
    print("\n-- Checking Inventory Balance --")
    checked_inv = 0
    failed_inv = 0
    for i in V0:
        for t in T:
            checked_inv += 1
            Ii_t   = X(I[i, t])
            Ii_tp1 = X(I[i, t+1])
            Si_tp1 = X(S[i, t+1])
            if Ii_t is None or Ii_tp1 is None or Si_tp1 is None:
                # در حالت عادی نباید اینجا بیاید چون SolCount>0 است
                failed_inv += 1
                report_check(False, "", f"✗ ATM {i}, t={t} skipped (no solution values)")
                continue

            qsum = sum(X(q[i, k, t]) or 0.0 for k in K)
            psum = sum(X(p[i, k, t]) or 0.0 for k in K)

            bal_val = Ii_t - r[i, t] + qsum - psum
            lhs = Ii_tp1 - Si_tp1

            ok = abs(lhs - bal_val) <= 1e-5
            if not ok:
                failed_inv += 1

            report_check(
                ok,
                ok_msg=f"✓ ATM {i}, t={t} OK",
                fail_msg=f"✗ ATM {i}, t={t} balance mismatch: (I-S)={lhs:.6f} vs bal={bal_val:.6f}"
            )
    print(f"Inventory balance checks: {checked_inv} checked, {failed_inv} failed")

    # 2) Cassette consistency only when visited / q>0
    print("\n-- Checking Cassette Consistency (only when visited / q>0) --")
    checked_cas = 0
    failed_cas = 0
    for i in V0:
        for k in K:
            for t in T:
                yikt = X(y[i, k, t]) or 0.0
                qikt = X(q[i, k, t]) or 0.0
                if yikt > 0.5 or qikt > eps:
                    checked_cas += 1
                    rhs = sum(gamma[w] * (X(v[i, w, k, t]) or 0.0) for w in W)
                    ok = abs(qikt - rhs) <= 1e-3
                    if not ok:
                        failed_cas += 1
                    report_check(
                        ok,
                        ok_msg=f"✓ ATM {i}, k={k}, t={t} OK",
                        fail_msg=f"✗ ATM {i}, k={k}, t={t} cassette mismatch: q={qikt:.6f} vs rhs={rhs:.6f}"
                    )
    print(f"Cassette checks: {checked_cas} checked, {failed_cas} failed")

    # 3) Pickup logic only when visited / p>0
    print("\n-- Checking Pickup Logic (only when visited / p>0) --")
    checked_pick = 0
    failed_pick = 0
    for i in V0:
        for k in K:
            for t in T:
                yikt = X(y[i, k, t]) or 0.0
                pikt = X(p[i, k, t]) or 0.0
                Ii_t = X(I[i, t])
                if (yikt > 0.5 or pikt > eps):
                    checked_pick += 1
                    if Ii_t is None:
                        failed_pick += 1
                        report_check(False, "", f"✗ ATM {i}, k={k}, t={t} pickup check skipped (no I value)")
                        continue
                    ok = pikt <= Ii_t + 1e-5
                    if not ok:
                        failed_pick += 1
                    report_check(
                        ok,
                        ok_msg=f"✓ Pickup logic ATM {i}, k={k}, t={t} OK",
                        fail_msg=f"✗ ATM {i}, k={k}, t={t} pickup exceeds inventory: p={pikt:.6f} > I={Ii_t:.6f}"
                    )
    print(f"Pickup checks: {checked_pick} checked, {failed_pick} failed")

    # 4) Degree constraints
    print("\n-- Checking Degree Constraints --")
    checked_deg = 0
    failed_deg = 0
    for node in V:
        for k in K:
            for t in T:
                checked_deg += 1
                yv = X(y[node, k, t])
                if yv is None:
                    failed_deg += 1
                    report_check(False, "", f"✗ Node {node}, k={k}, t={t} skipped (no y value)")
                    continue
                incident = 0.0
                for (i, j) in A:
                    if i == node or j == node:
                        incident += (X(x[i, j, k, t]) or 0.0)
                ok = abs(incident - 2.0 * yv) <= 1e-5
                if not ok:
                    failed_deg += 1
                report_check(
                    ok,
                    ok_msg=f"✓ Node {node}, k={k}, t={t} OK",
                    fail_msg=f"✗ Node {node}, k={k}, t={t} degree mismatch: incident={incident:.6f} vs 2y={2*yv:.6f}"
                )
    print(f"Degree checks: {checked_deg} checked, {failed_deg} failed")

    print("\n========================================")
    if passed:
        print("All sanity checks PASSED.")
    else:
        print("Some sanity checks FAILED.")
    print("========================================\n")

# =========================================================
# FINAL SUMMARY (همیشه واضح، آخرِ خروجی)
# =========================================================
log("\n" + "="*80)
log("FINAL SUMMARY")
log(f"Status: {model.Status} | SolCount: {model.SolCount}")
if model.SolCount > 0:
    log(f"Objective: {model.ObjVal}")
else:
    log("Objective: N/A (no feasible solution)")
try:
    log(f"Sanity checks passed? {passed}")
except NameError:
    log("Sanity checks passed? N/A")
log("="*80 + "\n")

print_service_summary(
    model=model,
    V0=V0, K=K, T=T, W=W,
    y=y, q=q, p=p, v=v,
    gamma=gamma,
    x=x, A=A,                      # اگر مسیر را هم می‌خواهی
    node_to_atm=node_to_atm        # اگر می‌خواهی ATM_ID واقعی چاپ شود
)
print_daily_routes(
    model=model,
    T=T, K=K,
    A=A, x=x,
    dist=dist,
    node_to_atm=node_to_atm,
    depot=depot,
    cost_per_km=cost_per_km
)

print("\n-- Optimality diagnostics --")
print(f"Status: {model.Status}")          # 2 یعنی OPTIMAL
print(f"SolCount: {model.SolCount}")

if model.SolCount > 0:
    print(f"ObjVal   : {model.ObjVal:.6f}")
    print(f"ObjBound : {model.ObjBound:.6f}")   # بهترین کران (LB برای MIN)
    print(f"MIPGap   : {model.MIPGap:.10f}")    # شکاف نسبی
    print(f"AbsGap   : {abs(model.ObjVal - model.ObjBound):.6f}")
# =========================================================
# FINAL SUMMARY (همیشه واضح، آخرِ خروجی)
# =========================================================
log("\n" + "="*80)
log("FINAL SUMMARY")
log(f"Status: {model.Status} | SolCount: {model.SolCount}")

if model.SolCount > 0:
    log(f"Objective: {model.ObjVal}")

    # -----------------------------
    # OBJECTIVE COMPONENTS PRINT
    # -----------------------------
    inv_cost_val      = inv_cost.getValue()
    stockout_cost_val = stockout_cost.getValue()
    route_cost_val    = route_cost.getValue()
    total_val         = inv_cost_val + stockout_cost_val + route_cost_val

    log("\n================= OBJECTIVE BREAKDOWN =================")
    log(f"inv_cost      = {inv_cost_val:.6f}")
    log(f"stockout_cost = {stockout_cost_val:.6f}")
    log(f"route_cost    = {route_cost_val:.6f}")
    log(f"sum(components)= {total_val:.6f}")
    log(f"model.ObjVal   = {model.ObjVal:.6f}")
    log(f"diff           = {abs(total_val - model.ObjVal):.6e}")
    log("=======================================================\n")

else:
    log("Objective: N/A (no feasible solution)")

try:
    log(f"Sanity checks passed? {passed}")
except NameError:
    log("Sanity checks passed? N/A")

log("="*80 + "\n")
