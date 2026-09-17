# 🕸️ Red de Clientes — Big Data

Aplicación de generación y visualización de redes de clientes a gran escala, construida con Python y Streamlit. Pensada para demostrar manejo de volúmenes grandes de datos y visualización de grafos.

## 🔗 Demo en vivo

**[https://red-clientes-bigdata-6qumnvappgrrb7jsid9tck.streamlit.app/](https://red-clientes-bigdata-6qumnvappgrrb7jsid9tck.streamlit.app/)**

No requiere instalación ni registro — abre el link directamente en el navegador. La base de datos se genera desde cero la primera vez que se usa la app (no viene precargada).

## Qué hace

- **Generación masiva de datos**: inserción vectorizada por lotes (numpy + `executemany`), no fila por fila — pensado para simular cientos de miles de registros de forma realista.
- **Base de datos optimizada**: SQLite con PRAGMAs de rendimiento (modo WAL, caché en memoria) para escritura rápida a escala.
- **Visualización de grafos interactiva**: la red de clientes se muestra sobre una muestra representativa (ningún navegador soporta renderizar millones de nodos a la vez).
- **Clustering**: segmentación de nodos con `MiniBatchKMeans` para agrupar clientes por patrones de relación.
- **Vista paginada**: la base de datos nunca se carga completa a memoria, incluso con volúmenes grandes.

## Correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack técnico

Python · Streamlit · SQLite · NetworkX · Pyvis · scikit-learn · Plotly
