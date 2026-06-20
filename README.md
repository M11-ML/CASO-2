# TSP · Distribución — 13 nodos

Aplicación Streamlit que resuelve el **Travelling Salesman Problem (TSP)** para una empresa de distribución con depósito en el nodo 0 y 12 clientes.

## Problema
- 13 puntos relevantes: `0, 3, 10, 21, 22, 35, 40, 41, 47, 65, 71, 76, 77`
- Matriz de distancias 13×13 (km, enteros)
- Objetivo: minimizar la distancia total recorriendo todos los clientes y regresando al depósito

## Método
1. **Heurística Vecino Más Cercano** desde cada uno de los 13 nodos de inicio
2. **Mejora 2-opt** sobre cada solución inicial
3. Se reporta la mejor solución encontrada

## Uso local

```bash
pip install -r requirements.txt
streamlit run tsp_app.py
```

## Deploy en Streamlit Cloud
1. Fork/push este repo a GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar el repo y apuntar a `tsp_app.py`
