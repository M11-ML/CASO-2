import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import time

# ── Datos del problema ──────────────────────────────────────────────────────
NODES = [0, 3, 10, 21, 22, 35, 40, 41, 47, 65, 71, 76, 77]
N = len(NODES)  # 13
IDX = {v: i for i, v in enumerate(NODES)}  # nodo real → índice

# Matriz de distancias (km, enteros)
RAW = [
    #  0    3   10   21   22   35   40   41   47   65   71   76   77
    [  0,  29,  18,  16,  24,  32,  17,  28,  29,  27,  27,  33,  34],  # 0
    [ 29,   0,  38,  40,  29,  48,  15,  57,  27,  53,  55,  24,  44],  # 3
    [ 18,  38,   0,  31,  42,  49,  22,  27,  21,  19,  36,  50,  16],  # 10
    [ 16,  40,  31,   0,  21,  18,  32,  25,  45,  30,  16,  34,  47],  # 21
    [ 24,  29,  42,  21,   0,  20,  30,  46,  46,  48,  36,  14,  56],  # 22
    [ 32,  48,  49,  18,  20,   0,  45,  40,  60,  47,  25,  33,  65],  # 35
    [ 17,  15,  22,  32,  30,  45,   0,  44,  16,  39,  45,  32,  30],  # 40
    [ 28,  57,  27,  25,  46,  40,  44,   0,  48,  11,  17,  59,  41],  # 41
    [ 29,  27,  21,  45,  46,  60,  16,  48,   0,  40,  54,  48,  18],  # 47
    [ 27,  53,  19,  30,  48,  47,  39,  11,  40,   0,  26,  60,  30],  # 65
    [ 27,  55,  36,  16,  36,  25,  45,  17,  54,  26,   0,  50,  52],  # 71
    [ 33,  24,  50,  34,  14,  33,  32,  59,  48,  60,  50,   0,  62],  # 76
    [ 34,  44,  16,  47,  56,  65,  30,  41,  18,  30,  52,  62,   0],  # 77
]
DIST = np.array(RAW, dtype=int)

# Posiciones aproximadas para el grafo (layout manual)
POS = {
    0:  (3.0, 5.0),
    3:  (1.5, 7.0),
    10: (5.5, 6.5),
    21: (2.5, 4.0),
    22: (1.5, 3.0),
    35: (2.0, 1.5),
    40: (0.5, 5.5),
    41: (6.0, 4.5),
    47: (0.0, 4.0),
    65: (6.5, 3.5),
    71: (5.0, 2.5),
    76: (1.0, 2.0),
    77: (5.5, 5.5),
}


# ── Algoritmos ───────────────────────────────────────────────────────────────

def dist_route(route):
    """Distancia total de una ruta (lista de índices, sin repetir depósito)."""
    total = 0
    for i in range(len(route) - 1):
        total += DIST[route[i]][route[i + 1]]
    return int(total)


def nearest_neighbor(start=0):
    """Heurística del vecino más cercano."""
    unvisited = list(range(N))
    route = [start]
    unvisited.remove(start)
    while unvisited:
        last = route[-1]
        nearest = min(unvisited, key=lambda j: DIST[last][j])
        route.append(nearest)
        unvisited.remove(nearest)
    route.append(start)
    return route, dist_route(route)


def two_opt(route, max_iter=5000):
    """Mejora 2-opt sobre una ruta."""
    best = route[:]
    best_dist = dist_route(best)
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                new = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                d = dist_route(new)
                if d < best_dist:
                    best, best_dist = new, d
                    improved = True
        iterations += 1
    return best, best_dist


def solve_tsp():
    """Resuelve TSP: NN + 2-opt desde todos los nodos de inicio."""
    best_route, best_dist = None, float("inf")
    for start in range(N):
        r, _ = nearest_neighbor(start)
        r2, d2 = two_opt(r)
        if d2 < best_dist:
            best_dist = d2
            best_route = r2
    return best_route, best_dist


# ── Gráfico ──────────────────────────────────────────────────────────────────

def draw_graph(route, dist_total):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    G = nx.Graph()
    for n in NODES:
        G.add_node(n)

    # Todas las aristas (gris tenue)
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(NODES[i], NODES[j], weight=DIST[i][j])

    pos = POS

    # Dibujar aristas de fondo
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#2a3f5f", width=0.6, alpha=0.4
    )

    # Aristas de la ruta óptima
    route_nodes = [NODES[i] for i in route]
    route_edges = [(route_nodes[k], route_nodes[k + 1]) for k in range(len(route_nodes) - 1)]
    route_weights = [DIST[IDX[u]][IDX[v]] for u, v in route_edges]

    nx.draw_networkx_edges(
        G, pos, edgelist=route_edges, ax=ax,
        edge_color="#f0a500", width=3.0, alpha=0.95
    )

    # Etiquetas de distancia en aristas de la ruta
    edge_labels = {(u, v): f"{DIST[IDX[u]][IDX[v]]} km" for u, v in route_edges}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_color="#f0a500", font_size=7.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="#0d1b2a", ec="none", alpha=0.7)
    )

    # Nodos
    node_colors = ["#e74c3c" if n == 0 else "#00bcd4" for n in NODES]
    node_sizes  = [500 if n == 0 else 380 for n in NODES]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edgecolors="#ffffff", linewidths=1.5
    )

    # Etiquetas de nodos
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: str(n) for n in NODES},
        font_color="white", font_size=8, font_weight="bold"
    )

    # Orden de visita sobre la ruta
    for k, idx in enumerate(route[:-1]):
        node = NODES[idx]
        x, y = pos[node]
        ax.annotate(
            f"#{k + 1}", xy=(x, y), xytext=(x + 0.15, y + 0.18),
            color="#a0f0a0", fontsize=7, fontweight="bold",
            ha="left", va="bottom"
        )

    # Leyenda y título
    patch_depot  = mpatches.Patch(color="#e74c3c", label="Depósito (nodo 0)")
    patch_client = mpatches.Patch(color="#00bcd4", label="Cliente")
    patch_route  = mpatches.Patch(color="#f0a500", label="Ruta óptima")
    ax.legend(
        handles=[patch_depot, patch_client, patch_route],
        loc="lower right", facecolor="#1a2a3a", edgecolor="#4a6080",
        labelcolor="white", fontsize=9
    )
    ax.set_title(
        f"Ruta óptima  ·  Distancia total: {dist_total} km",
        color="white", fontsize=14, fontweight="bold", pad=14
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TSP · Distribución 13 nodos",
    page_icon="🚚",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0d1b2a; color: #e0eaf5; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #f0a500 !important; font-weight: 700; }
    h2, h3 { color: #00bcd4 !important; }
    .metric-card {
        background: #132233;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #1e3a55;
        text-align: center;
    }
    .metric-big { font-size: 2.4rem; font-weight: 700; color: #f0a500; }
    .metric-label { font-size: 0.85rem; color: #7fa8c9; margin-top: 0.2rem; }
    .route-pill {
        display: inline-block;
        background: #1e3a55;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.9rem;
        color: #e0eaf5;
        border: 1px solid #2a5080;
    }
    .route-pill.depot { background: #6b1010; border-color: #e74c3c; color: #ffaaaa; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🚚 TSP · Distribución — 13 nodos")
st.markdown("**Caso 2 · Enfoque por etapas** — depósito en nodo **0**, 12 clientes, ruta de menor distancia total.")

st.divider()

# Parámetros / botón
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("▶ Resolver TSP", type="primary", use_container_width=True)

with col_info:
    st.markdown("""
    **Método:** Vecino Más Cercano (heurística inicial) + mejora **2-opt**  
    Se prueba cada uno de los 13 nodos como punto de inicio y se conserva la mejor solución.
    """)

st.divider()

if run:
    with st.spinner("Optimizando ruta…"):
        t0 = time.time()
        best_route, best_dist = solve_tsp()
        elapsed = time.time() - t0

    # Convertir índices → nodos reales
    route_real = [NODES[i] for i in best_route]

    # Métricas
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-big">{best_dist} km</div>
            <div class="metric-label">Distancia total de la ruta</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-big">{N - 1}</div>
            <div class="metric-label">Clientes visitados</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-big">{elapsed:.2f} s</div>
            <div class="metric-label">Tiempo de cómputo</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Secuencia de visita")
    pills_html = ""
    for n in route_real:
        cls = "depot" if n == 0 else ""
        label = f"{'🏭 ' if n == 0 else ''}{n}"
        pills_html += f'<span class="route-pill {cls}">{label}</span> → '
    pills_html = pills_html.rstrip(" → ")
    st.markdown(pills_html + ' <span class="route-pill depot">🏭 0</span>', unsafe_allow_html=True)

    st.markdown("### Detalle de tramos")
    rows = []
    for k in range(len(best_route) - 1):
        orig = NODES[best_route[k]]
        dest = NODES[best_route[k + 1]]
        d    = DIST[best_route[k]][best_route[k + 1]]
        rows.append({"Tramo": k + 1, "Desde": orig, "Hasta": dest, "Distancia (km)": d})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Tramo": "—", "Desde": "—", "Hasta": "TOTAL", "Distancia (km)": best_dist}
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Restricciones del modelo ────────────────────────────────────────────
    with st.expander("📋 Restricciones del modelo TSP", expanded=True):
        st.markdown("""
**Variables de decisión**

$$x_{ij} \\in \\{0, 1\\} \\quad \\forall\\, i \\neq j$$

$x_{ij} = 1$ si se viaja del nodo $i$ al nodo $j$; 0 en caso contrario.

---

**1. Salida única por nodo** — de cada nodo se sale exactamente una vez:

$$\\sum_{j \\neq i} x_{ij} = 1 \\qquad \\forall\\, i \\in \\{0, 3, 10, 21, 22, 35, 40, 41, 47, 65, 71, 76, 77\\}$$

**2. Entrada única por nodo** — a cada nodo se llega exactamente una vez:

$$\\sum_{i \\neq j} x_{ij} = 1 \\qquad \\forall\\, j \\in \\{0, 3, 10, 21, 22, 35, 40, 41, 47, 65, 71, 76, 77\\}$$

**3. Eliminación de sub-tours** (restricciones MTZ) — evita que se formen ciclos desconectados del depósito:

$$u_i - u_j + N \\cdot x_{ij} \\leq N - 1 \\qquad \\forall\\, i \\neq j,\\; i,j \\neq 0$$

$$1 \\leq u_i \\leq N - 1 \\qquad \\forall\\, i \\neq 0$$

donde $u_i$ es el orden de visita del nodo $i$ (variable auxiliar entera).

**4. Dominio binario:**

$$x_{ij} \\in \\{0, 1\\} \\qquad \\forall\\, i, j$$

---
""")

        # Tabla de verificación de restricciones sobre la solución encontrada
        st.markdown("**Verificación sobre la solución óptima encontrada:**")
        check_rows = []
        for k, idx in enumerate(best_route[:-1]):
            node = NODES[idx]
            salidas = sum(1 for kk in range(len(best_route)-1) if best_route[kk] == idx)
            entradas = sum(1 for kk in range(1, len(best_route)) if best_route[kk] == idx)
            check_rows.append({
                "Nodo": node,
                "Orden visita (u)": k + 1,
                "Salidas (= 1 ✓)": salidas,
                "Entradas (= 1 ✓)": entradas,
            })
        st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)

    st.markdown("### Grafo de la ruta óptima")
    fig = draw_graph(best_route, best_dist)
    st.pyplot(fig)

    # Matriz de distancias (expandible)
    with st.expander("📊 Matriz de distancias completa (13×13)"):
        df_mat = pd.DataFrame(DIST, index=NODES, columns=NODES)
        st.dataframe(df_mat, use_container_width=True)

else:
    st.info("Presiona **▶ Resolver TSP** para calcular la ruta óptima.")

    with st.expander("📊 Ver matriz de distancias"):
        df_mat = pd.DataFrame(DIST, index=NODES, columns=NODES)
        st.dataframe(df_mat, use_container_width=True)
