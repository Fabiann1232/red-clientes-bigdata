"""
Red de Clientes — versión "big data friendly"
------------------------------------------------------------------------------------
Cambios clave frente a la versión anterior:
- Sin formulario manual: todo se genera aleatoriamente, en volumen.
- Inserción masiva vectorizada (numpy + executemany por lotes), NO fila por fila.
- Sin verificación "existe/no existe" por cliente (el cuello de botella real a escala).
- PRAGMAs de SQLite para escritura rápida (WAL, cache en memoria).
- Vista de base de datos paginada (nunca carga todo a memoria).
- El grafo se visualiza sobre una MUESTRA (ningún navegador soporta millones de nodos).

Cómo correrlo:
    pip install -r requirements.txt
    python -m streamlit run app.py
"""

import sqlite3
import time
import io

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
from sklearn.cluster import MiniBatchKMeans
import streamlit.components.v1 as components

DB_PATH = "red_clientes.db"
LOTE = 50_000  # tamaño de cada bloque de inserción

# ----------------------------------------------------------------------
# 1. CONEXIÓN Y AJUSTES DE RENDIMIENTO (lo más "big data" que da SQLite)
# ----------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")       # escritura sin bloquear lecturas
    conn.execute("PRAGMA synchronous=NORMAL")     # menos fsync, mucho más rápido
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-64000")      # ~64MB de caché en RAM
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nodos (
            id INTEGER PRIMARY KEY,
            tipo TEXT NOT NULL,
            nombre TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS relaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origen_id INTEGER NOT NULL,
            destino_id INTEGER NOT NULL,
            etiqueta TEXT NOT NULL
        )
    """)
    # Índices: esto es lo que hace rápidas las consultas cuando ya hay millones de filas
    cur.execute("CREATE INDEX IF NOT EXISTS idx_nodos_tipo ON nodos(tipo)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_origen ON relaciones(origen_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_destino ON relaciones(destino_id)")
    conn.commit()
    conn.close()
    asegurar_columna_fecha()


def asegurar_columna_fecha():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(nodos)")
    columnas = [c[1] for c in cur.fetchall()]
    if "fecha_registro" not in columnas:
        cur.execute("ALTER TABLE nodos ADD COLUMN fecha_registro TEXT")
        conn.commit()
    conn.close()


def borrar_todo():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM relaciones")
    cur.execute("DELETE FROM nodos")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()


def contar_registros():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM nodos")
    n_nodos = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM relaciones")
    n_rel = cur.fetchone()[0]
    conn.close()
    return n_nodos, n_rel


# ----------------------------------------------------------------------
# 2. CATÁLOGO FIJO (categorías/productos/preferencias no crecen con el volumen)
# ----------------------------------------------------------------------

PRODUCTOS_POR_CATEGORIA = {
    "Ropa": ["Camisa", "Pantalón", "Chaqueta", "Zapatos", "Vestido"],
    "Tecnología": ["Laptop", "Celular", "Auriculares", "Televisor", "Tablet"],
    "Hogar": ["Sofá", "Mesa", "Silla", "Cafetera", "Refrigerador"],
    "Deporte": ["Bicicleta", "Balón", "Tenis deportivos", "Pesas", "Raqueta"],
    "Accesorios": ["Reloj", "Perfume", "Bolso", "Gafas de sol", "Cinturón"],
}
PREFERENCIAS = [
    "precio bajo", "envío rápido", "calidad premium", "garantía extendida",
    "pago en cuotas", "atención personalizada", "producto ecológico",
    "entrega el mismo día", "marca reconocida", "descuentos frecuentes",
]

NOMBRES = [
    "María", "José", "Juan", "Ana", "Luis", "Carmen", "Carlos", "Laura", "Andrés", "Sofía",
    "Diego", "Valentina", "Miguel", "Camila", "David", "Isabella", "Santiago", "Daniela",
    "Sebastián", "Gabriela", "Fernando", "Paula", "Ricardo", "Natalia", "Jorge", "Andrea",
    "Alejandro", "Mariana", "Pedro", "Lucía",
]
APELLIDOS = [
    "García", "Rodríguez", "Martínez", "López", "González", "Pérez", "Sánchez", "Ramírez",
    "Torres", "Flores", "Rivera", "Gómez", "Díaz", "Morales", "Ortiz", "Castro", "Vargas",
    "Rojas", "Jiménez", "Moreno", "Muñoz", "Álvarez", "Romero", "Suárez", "Herrera",
]


def asegurar_catalogo_fijo():
    """Crea (si no existen) los nodos de categoría/producto/preferencia, con IDs fijos y bajos."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM nodos WHERE tipo != 'cliente'")
    if cur.fetchone()[0] > 0:
        conn.close()
        return  # ya existe

    siguiente_id = 1
    ids = {"categoria": {}, "producto": {}, "preferencia": {}}
    filas_nodos = []
    filas_relaciones = []

    for categoria, productos in PRODUCTOS_POR_CATEGORIA.items():
        cat_id = siguiente_id; siguiente_id += 1
        ids["categoria"][categoria] = cat_id
        filas_nodos.append((cat_id, "categoria", categoria))
        for producto in productos:
            prod_id = siguiente_id; siguiente_id += 1
            ids["producto"][producto] = prod_id
            filas_nodos.append((prod_id, "producto", producto))
            filas_relaciones.append((prod_id, cat_id, "pertenece_a"))

    for pref in PREFERENCIAS:
        pref_id = siguiente_id; siguiente_id += 1
        ids["preferencia"][pref] = pref_id
        filas_nodos.append((pref_id, "preferencia", pref))

    cur.executemany("INSERT INTO nodos (id, tipo, nombre) VALUES (?, ?, ?)", filas_nodos)
    cur.executemany(
        "INSERT INTO relaciones (origen_id, destino_id, etiqueta) VALUES (?, ?, ?)",
        filas_relaciones,
    )
    conn.commit()
    conn.close()


def obtener_catalogo_ids():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, nombre FROM nodos WHERE tipo='producto'")
    productos = cur.fetchall()
    cur.execute("SELECT id, nombre FROM nodos WHERE tipo='preferencia'")
    preferencias = cur.fetchall()
    conn.close()
    return productos, preferencias


# ----------------------------------------------------------------------
# 3. GENERACIÓN MASIVA VECTORIZADA (el corazón del rendimiento)
# ----------------------------------------------------------------------

def generar_datos_masivos(n_clientes: int, progreso_cb=None):
    asegurar_catalogo_fijo()
    productos, preferencias = obtener_catalogo_ids()
    productos_ids = np.array([p[0] for p in productos])
    preferencias_ids = np.array([p[0] for p in preferencias])

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(id), 0) FROM nodos")
    siguiente_id = cur.fetchone()[0] + 1

    total_insertado = 0
    t0 = time.time()

    while total_insertado < n_clientes:
        lote = min(LOTE, n_clientes - total_insertado)

        # --- Generación vectorizada con numpy (sin loops de Python fila a fila) ---
        nombres = np.random.choice(NOMBRES, lote)
        apellidos = np.random.choice(APELLIDOS, lote)
        ids_cliente = np.arange(siguiente_id, siguiente_id + lote)
        # el id va en el nombre para garantizar unicidad sin tener que consultar la BD
        nombres_completos = [f"{n} {a} #{i}" for n, a, i in zip(nombres, apellidos, ids_cliente)]

        productos_elegidos = np.random.choice(productos_ids, lote)
        preferencias_elegidas = np.random.choice(preferencias_ids, lote)

        # Fecha de registro simulada en los últimos 90 días, para poder mostrar tendencias
        dias_atras = np.random.randint(0, 90, size=lote)
        fechas = (pd.Timestamp.now().normalize() - pd.to_timedelta(dias_atras, unit="D")).strftime("%Y-%m-%d").tolist()

        filas_nodos = list(zip(ids_cliente.tolist(), ["cliente"] * lote, nombres_completos, fechas))

        filas_relaciones = list(zip(ids_cliente.tolist(), productos_elegidos.tolist(), ["compró/interesado_en"] * lote))
        filas_relaciones += list(zip(ids_cliente.tolist(), preferencias_elegidas.tolist(), ["prefiere"] * lote))

        cur.executemany(
            "INSERT INTO nodos (id, tipo, nombre, fecha_registro) VALUES (?, ?, ?, ?)", filas_nodos
        )
        cur.executemany(
            "INSERT INTO relaciones (origen_id, destino_id, etiqueta) VALUES (?, ?, ?)",
            filas_relaciones,
        )
        conn.commit()

        siguiente_id += lote
        total_insertado += lote
        if progreso_cb:
            velocidad = total_insertado / max(time.time() - t0, 0.001)
            progreso_cb(total_insertado, n_clientes, velocidad)

    conn.close()
    return time.time() - t0


def seccion_generar_datos():
    st.subheader("🎲 Generación masiva de datos")
    st.caption("Generación vectorizada por lotes — pensada para volúmenes grandes (cientos de miles a millones).")

    n_nodos, n_rel = contar_registros()
    st.write(f"Actualmente en la base de datos: **{n_nodos:,} nodos** / **{n_rel:,} relaciones**.")

    n = st.number_input(
        "¿Cuántos clientes generar?",
        min_value=1_000, max_value=5_000_000, value=100_000, step=10_000,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Generar", type="primary"):
            barra = st.progress(0.0)
            texto = st.empty()

            def cb(hecho, total, velocidad):
                barra.progress(hecho / total)
                texto.text(f"{hecho:,} / {total:,} clientes — {velocidad:,.0f} filas/seg")

            duracion = generar_datos_masivos(int(n), progreso_cb=cb)
            st.success(f"Listo: {n:,} clientes generados en {duracion:.1f} segundos.")
            st.rerun()
    with col2:
        if st.button("🗑️ Borrar toda la base de datos"):
            borrar_todo()
            st.warning("Base de datos vaciada.")
            st.rerun()


# ----------------------------------------------------------------------
# 4. GRAFO SOBRE MUESTRA (nunca sobre el total si hay millones de filas)
# ----------------------------------------------------------------------

COLORES_POR_TIPO = {
    "cliente": "#4CAF50",
    "producto": "#2196F3",
    "categoria": "#FF9800",
    "preferencia": "#9C27B0",
}


def construir_y_mostrar_grafo(tamano_muestra: int):
    conn = get_conn()
    cur = conn.cursor()

    # El catálogo (categoría/producto/preferencia) siempre se incluye completo: es pequeño y fijo
    cur.execute("SELECT id, tipo, nombre FROM nodos WHERE tipo != 'cliente'")
    nodos_catalogo = cur.fetchall()

    # Muestra aleatoria de clientes usando ORDER BY RANDOM() con límite (rápido incluso a escala
    # gracias al índice; para volúmenes extremos se podría muestrear por rango de IDs)
    cur.execute(
        "SELECT id, tipo, nombre FROM nodos WHERE tipo='cliente' ORDER BY RANDOM() LIMIT ?",
        (tamano_muestra,),
    )
    nodos_clientes = cur.fetchall()

    ids_muestra = [n[0] for n in nodos_clientes]
    if not ids_muestra:
        conn.close()
        st.info("Todavía no hay clientes. Genera datos primero.")
        return

    placeholders = ",".join("?" * len(ids_muestra))
    cur.execute(
        f"SELECT origen_id, destino_id, etiqueta FROM relaciones WHERE origen_id IN ({placeholders})",
        ids_muestra,
    )
    relaciones_clientes = cur.fetchall()

    cur.execute("SELECT origen_id, destino_id, etiqueta FROM relaciones WHERE etiqueta='pertenece_a'")
    relaciones_catalogo = cur.fetchall()
    conn.close()

    G = nx.Graph()
    for node_id, tipo, nombre in nodos_catalogo + nodos_clientes:
        G.add_node(node_id, label=nombre, title=f"Tipo: {tipo}", color=COLORES_POR_TIPO.get(tipo, "#888888"))
    for origen_id, destino_id, etiqueta in relaciones_clientes + relaciones_catalogo:
        if origen_id in G.nodes and destino_id in G.nodes:
            G.add_edge(origen_id, destino_id, title=etiqueta, label=etiqueta)

    net = Network(height="650px", width="100%", bgcolor="#111111", font_color="white")
    net.from_nx(G)
    net.force_atlas_2based(gravity=-40, central_gravity=0.01, spring_length=120)
    net.show_buttons(filter_=["physics"])
    net.save_graph("grafo.html")
    with open("grafo.html", "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=680, scrolling=True)


def seccion_grafo():
    st.subheader("🌐 Red de conexiones (muestra)")
    n_nodos, _ = contar_registros()
    st.caption(
        f"Hay {n_nodos:,} nodos en total. Ningún navegador puede dibujar eso de forma legible, "
        "así que se visualiza una muestra aleatoria."
    )
    tamano = st.slider("Tamaño de la muestra de clientes a visualizar", 50, 2000, 300, step=50)
    if st.button("🔄 Generar / refrescar muestra"):
        st.session_state["mostrar_grafo"] = True
    if st.session_state.get("mostrar_grafo"):
        construir_y_mostrar_grafo(tamano)


# ----------------------------------------------------------------------
# 5. VISTA PAGINADA DE LA BASE DE DATOS (nunca carga todo a memoria)
# ----------------------------------------------------------------------

def seccion_ver_base_datos():
    st.subheader("🗄️ Explorar base de datos (paginado)")
    n_nodos, n_rel = contar_registros()
    st.write(f"Total: **{n_nodos:,} nodos**, **{n_rel:,} relaciones**.")

    if n_nodos == 0:
        st.info("Todavía no hay datos.")
        return

    conn = get_conn()
    tipos = pd.read_sql_query("SELECT DISTINCT tipo FROM nodos", conn)["tipo"].tolist()
    conn.close()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        filtro_tipo = st.selectbox("Filtrar por tipo", ["Todos"] + sorted(tipos))
    with col2:
        tam_pagina = st.selectbox("Filas por página", [50, 100, 500, 1000], index=1)
    with col3:
        pagina = st.number_input("Página", min_value=1, value=1, step=1)

    offset = (pagina - 1) * tam_pagina
    conn = get_conn()
    if filtro_tipo == "Todos":
        query = "SELECT * FROM nodos LIMIT ? OFFSET ?"
        params = (tam_pagina, offset)
    else:
        query = "SELECT * FROM nodos WHERE tipo=? LIMIT ? OFFSET ?"
        params = (filtro_tipo, tam_pagina, offset)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    st.dataframe(df, use_container_width=True, height=400)
    st.caption(f"Mostrando filas {offset + 1:,} a {offset + len(df):,}")


# ----------------------------------------------------------------------
# 6. EXPORTAR (pensado para volumen: CSV en streaming + descarga del .db completo)
# ----------------------------------------------------------------------

def seccion_exportar():
    st.subheader("📤 Exportar datos")
    n_nodos, n_rel = contar_registros()
    if n_nodos == 0:
        st.info("Todavía no hay datos para exportar.")
        return

    st.markdown(
        "Para volúmenes grandes, lo más eficiente es descargar el **archivo de base de datos "
        "completo** (SQLite), en vez de convertir todo a CSV en memoria."
    )
    with open(DB_PATH, "rb") as f:
        st.download_button(
            "⬇️ Descargar base de datos completa (.db)",
            data=f.read(),
            file_name="red_clientes.db",
            mime="application/octet-stream",
        )

    st.markdown("---")
    st.markdown("**Exportar como CSV** (se genera en streaming por bloques; con millones de filas puede tardar):")

    tabla = st.selectbox("Tabla a exportar", ["nodos", "relaciones"])
    limite = st.number_input(
        "Límite de filas a exportar (deja en 0 para exportar todo — puede ser pesado)",
        min_value=0, value=100_000, step=10_000,
    )

    if st.button("Generar CSV"):
        conn = get_conn()
        query = f"SELECT * FROM {tabla}"
        if limite > 0:
            query += f" LIMIT {int(limite)}"

        buffer = io.StringIO()
        primero = True
        filas_totales = 0
        for chunk in pd.read_sql_query(query, conn, chunksize=100_000):
            chunk.to_csv(buffer, index=False, header=primero, mode="a")
            primero = False
            filas_totales += len(chunk)
        conn.close()

        st.download_button(
            f"⬇️ Descargar {tabla}.csv ({filas_totales:,} filas)",
            data=buffer.getvalue().encode("utf-8-sig"),
            file_name=f"{tabla}.csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------------
# 7. IA — SEGMENTACIÓN DE CLIENTES (MiniBatchKMeans, escalable)
# ----------------------------------------------------------------------

MAX_FILAS_CLUSTERING = 300_000  # límite razonable para entrenar en segundos, no minutos


def asegurar_columna_segmento():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(nodos)")
    columnas = [c[1] for c in cur.fetchall()]
    if "segmento" not in columnas:
        cur.execute("ALTER TABLE nodos ADD COLUMN segmento INTEGER")
        conn.commit()
    conn.close()


def obtener_dataset_clientes(limite=None):
    """Une cada cliente con su producto y preferencia elegidos, vía las relaciones."""
    conn = get_conn()
    query = """
        SELECT r1.origen_id AS cliente_id, r1.destino_id AS producto_id, r2.destino_id AS preferencia_id
        FROM relaciones r1
        JOIN relaciones r2 ON r1.origen_id = r2.origen_id AND r2.etiqueta = 'prefiere'
        WHERE r1.etiqueta = 'compró/interesado_en'
    """
    if limite:
        query += f" LIMIT {int(limite)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def ejecutar_clustering(n_clusters: int):
    asegurar_columna_segmento()
    t0 = time.time()

    df = obtener_dataset_clientes(limite=MAX_FILAS_CLUSTERING)
    if df.empty:
        return None, 0

    # One-hot de producto y preferencia: son pocas categorías (≈35), así que esto es barato
    X = pd.get_dummies(df[["producto_id", "preferencia_id"]].astype(str))

    modelo = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init="auto", batch_size=2048)
    df["cluster"] = modelo.fit_predict(X)

    # Guardar los segmentos en la BD (solo de la muestra usada para entrenar/etiquetar)
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE nodos SET segmento = ? WHERE id = ?",
        list(zip(df["cluster"].tolist(), df["cliente_id"].tolist())),
    )
    conn.commit()
    conn.close()

    duracion = time.time() - t0
    return df, duracion


def describir_segmentos(df: pd.DataFrame):
    """Para cada cluster, encuentra el producto y preferencia más frecuentes (el 'perfil' del segmento)."""
    conn = get_conn()
    nombres_producto = pd.read_sql_query("SELECT id, nombre FROM nodos WHERE tipo='producto'", conn)
    nombres_pref = pd.read_sql_query("SELECT id, nombre FROM nodos WHERE tipo='preferencia'", conn)
    conn.close()

    mapa_prod = dict(zip(nombres_producto["id"], nombres_producto["nombre"]))
    mapa_pref = dict(zip(nombres_pref["id"], nombres_pref["nombre"]))

    filas = []
    for cluster_id, grupo in df.groupby("cluster"):
        producto_top = grupo["producto_id"].mode().iloc[0]
        pref_top = grupo["preferencia_id"].mode().iloc[0]
        filas.append({
            "Segmento": f"Segmento {cluster_id}",
            "Tamaño": len(grupo),
            "Producto más común": mapa_prod.get(producto_top, "?"),
            "Preferencia más común": mapa_pref.get(pref_top, "?"),
        })
    return pd.DataFrame(filas).sort_values("Tamaño", ascending=False)


def seccion_segmentacion_ia():
    st.subheader("🤖 Segmentación de clientes con IA (clustering)")
    st.caption(
        "Agrupa automáticamente a los clientes por patrones de comportamiento (producto + preferencia) "
        "usando MiniBatchKMeans — la variante de K-Means diseñada para volúmenes grandes."
    )

    n_nodos, _ = contar_registros()
    if n_nodos == 0:
        st.info("Genera datos primero en la pestaña 'Generar datos masivos'.")
        return

    n_clusters = st.slider("Número de segmentos a crear", 2, 10, 5)

    if st.button("🧠 Ejecutar segmentación", type="primary"):
        with st.spinner("Entrenando el modelo de clustering..."):
            df, duracion = ejecutar_clustering(n_clusters)
        if df is None:
            st.warning("No hay suficientes datos para segmentar.")
            return
        st.session_state["df_clusters"] = df
        st.success(
            f"Segmentación completada sobre {len(df):,} clientes en {duracion:.2f} segundos."
        )

    df = st.session_state.get("df_clusters")
    if df is None:
        return

    perfiles = describir_segmentos(df)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.pie(
            perfiles, names="Segmento", values="Tamaño",
            title="Distribución de clientes por segmento", hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(
            perfiles.sort_values("Tamaño"), x="Tamaño", y="Segmento", orientation="h",
            title="Tamaño de cada segmento", text="Tamaño",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Perfil de cada segmento** (lo que más define a cada grupo):")
    st.dataframe(perfiles, use_container_width=True, hide_index=True)

    st.caption(
        f"Nota de escalabilidad: el modelo se entrena y etiqueta sobre una muestra de hasta "
        f"{MAX_FILAS_CLUSTERING:,} clientes para mantener la respuesta en segundos. "
        "Para producción, esto se haría con predicción por lotes (batches) sobre el total."
    )


# ----------------------------------------------------------------------
# 8. IA — RECOMENDACIÓN POR CO-OCURRENCIA
# ----------------------------------------------------------------------

def tabla_coocurrencia():
    df = obtener_dataset_clientes(limite=MAX_FILAS_CLUSTERING)
    conn = get_conn()
    nombres_producto = pd.read_sql_query("SELECT id, nombre FROM nodos WHERE tipo='producto'", conn)
    nombres_pref = pd.read_sql_query("SELECT id, nombre FROM nodos WHERE tipo='preferencia'", conn)
    conn.close()

    mapa_prod = dict(zip(nombres_producto["id"], nombres_producto["nombre"]))
    mapa_pref = dict(zip(nombres_pref["id"], nombres_pref["nombre"]))

    df["producto"] = df["producto_id"].map(mapa_prod)
    df["preferencia"] = df["preferencia_id"].map(mapa_pref)
    return df


def seccion_recomendaciones():
    st.subheader("🔗 Motor de recomendación (co-ocurrencia)")
    st.caption(
        "Responde preguntas tipo: 'los clientes que prefieren X, ¿qué suelen comprar?' "
        "— la base de cualquier sistema de recomendación tipo 'clientes que compraron esto también...'."
    )

    n_nodos, _ = contar_registros()
    if n_nodos == 0:
        st.info("Genera datos primero en la pestaña 'Generar datos masivos'.")
        return

    df = tabla_coocurrencia()

    modo = st.radio("¿Qué quieres explorar?", ["Por preferencia → productos", "Por producto → preferencias"], horizontal=True)

    if modo == "Por preferencia → productos":
        opciones = sorted(df["preferencia"].unique())
        seleccion = st.selectbox("Elige una preferencia", opciones)
        resultado = (
            df[df["preferencia"] == seleccion]["producto"]
            .value_counts()
            .reset_index()
        )
        resultado.columns = ["Producto", "Clientes"]
        titulo = f"Productos más asociados a la preferencia '{seleccion}'"
    else:
        opciones = sorted(df["producto"].unique())
        seleccion = st.selectbox("Elige un producto", opciones)
        resultado = (
            df[df["producto"] == seleccion]["preferencia"]
            .value_counts()
            .reset_index()
        )
        resultado.columns = ["Preferencia", "Clientes"]
        titulo = f"Preferencias más asociadas al producto '{seleccion}'"

    fig = px.bar(resultado, x="Clientes", y=resultado.columns[0], orientation="h", title=titulo)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Basado en una muestra de hasta {MAX_FILAS_CLUSTERING:,} clientes.")


# ----------------------------------------------------------------------
# 9. DASHBOARD DE DATOS (KPIs + tendencias — la parte de analítica/"Big Data")
# ----------------------------------------------------------------------

MAX_FILAS_DASHBOARD = 300_000  # cap para que el join de 3 tablas responda en segundos


def obtener_dataset_dashboard(limite=MAX_FILAS_DASHBOARD):
    """Una sola consulta que une cliente -> producto -> categoría y cliente -> preferencia."""
    conn = get_conn()
    query = f"""
        SELECT
            c.id AS cliente_id,
            c.fecha_registro AS fecha_registro,
            p.nombre AS producto,
            cat.nombre AS categoria,
            pref.nombre AS preferencia
        FROM nodos c
        JOIN relaciones r_prod ON r_prod.origen_id = c.id AND r_prod.etiqueta = 'compró/interesado_en'
        JOIN nodos p ON p.id = r_prod.destino_id
        JOIN relaciones r_cat ON r_cat.origen_id = p.id AND r_cat.etiqueta = 'pertenece_a'
        JOIN nodos cat ON cat.id = r_cat.destino_id
        JOIN relaciones r_pref ON r_pref.origen_id = c.id AND r_pref.etiqueta = 'prefiere'
        JOIN nodos pref ON pref.id = r_pref.destino_id
        WHERE c.tipo = 'cliente'
        LIMIT {int(limite)}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def seccion_dashboard():
    st.subheader("📊 Dashboard de datos")
    st.caption("Métricas y tendencias agregadas sobre toda la base — la capa de analítica del proyecto.")

    conn = get_conn()
    n_clientes = pd.read_sql_query("SELECT COUNT(*) as n FROM nodos WHERE tipo='cliente'", conn)["n"].iloc[0]
    conn.close()

    if n_clientes == 0:
        st.info("Genera datos primero en la pestaña 'Generar datos masivos'.")
        return

    t0 = time.time()
    df = obtener_dataset_dashboard()
    duracion = time.time() - t0

    # ---- KPIs ----
    categoria_top = df["categoria"].mode().iloc[0]
    preferencia_top = df["preferencia"].mode().iloc[0]
    producto_top = df["producto"].mode().iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes totales", f"{n_clientes:,}")
    k2.metric("Categoría más popular", categoria_top)
    k3.metric("Producto más popular", producto_top)
    k4.metric("Preferencia más común", preferencia_top)

    # ---- Tendencia temporal ----
    tendencia = df.groupby("fecha_registro").size().reset_index(name="clientes")
    tendencia = tendencia.sort_values("fecha_registro")
    fig_tendencia = px.line(
        tendencia, x="fecha_registro", y="clientes", markers=True,
        title="Clientes registrados por día (últimos 90 días, simulado)",
    )
    st.plotly_chart(fig_tendencia, use_container_width=True)

    # ---- Distribuciones ----
    col1, col2 = st.columns(2)
    with col1:
        por_categoria = df["categoria"].value_counts().reset_index()
        por_categoria.columns = ["Categoría", "Clientes"]
        fig_cat = px.bar(por_categoria, x="Categoría", y="Clientes", title="Clientes por categoría", text="Clientes")
        st.plotly_chart(fig_cat, use_container_width=True)
    with col2:
        por_pref = df["preferencia"].value_counts().reset_index()
        por_pref.columns = ["Preferencia", "Clientes"]
        fig_pref = px.pie(por_pref, names="Preferencia", values="Clientes", title="Distribución por preferencia")
        st.plotly_chart(fig_pref, use_container_width=True)

    top_productos = df["producto"].value_counts().head(10).reset_index()
    top_productos.columns = ["Producto", "Clientes"]
    fig_top = px.bar(
        top_productos.sort_values("Clientes"), x="Clientes", y="Producto", orientation="h",
        title="Top 10 productos",
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.caption(
        f"Calculado sobre una muestra de {len(df):,} clientes (join de 3 tablas) en {duracion:.2f} segundos."
    )


# ----------------------------------------------------------------------
# 10. ESTILO VISUAL (tema oscuro tipo "centro de comando")
# ----------------------------------------------------------------------

def aplicar_estilo():
    st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at 20% 0%, #1a1f3a 0%, #0b0e1a 45%, #05060c 100%);
        }
        h1, h2, h3 {
            background: linear-gradient(90deg, #4CAF50, #2196F3, #9C27B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 0 20px rgba(76, 175, 80, 0.08);
        }
        div[data-testid="stMetricValue"] {
            color: #4CAF50 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.04);
            border-radius: 10px 10px 0 0;
            padding: 8px 14px;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, rgba(76,175,80,0.25), rgba(33,150,243,0.25));
        }
        div.stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(76, 175, 80, 0.4);
            background: rgba(76, 175, 80, 0.08);
        }
        div.stButton > button:hover {
            border: 1px solid #4CAF50;
            box-shadow: 0 0 12px rgba(76, 175, 80, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 11. GRAFO 3D INTERACTIVO (rotable, zoom libre — el visual "avanzado")
# ----------------------------------------------------------------------

def construir_grafo_3d(tamano_muestra: int, colorear_por_segmento: bool = False):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, tipo, nombre, segmento FROM nodos WHERE tipo != 'cliente'")
    nodos_catalogo = cur.fetchall()

    cur.execute(
        "SELECT id, tipo, nombre, segmento FROM nodos WHERE tipo='cliente' ORDER BY RANDOM() LIMIT ?",
        (tamano_muestra,),
    )
    nodos_clientes = cur.fetchall()

    ids_muestra = [n[0] for n in nodos_clientes]
    if not ids_muestra:
        conn.close()
        st.info("Todavía no hay clientes. Genera datos primero.")
        return

    placeholders = ",".join("?" * len(ids_muestra))
    cur.execute(
        f"SELECT origen_id, destino_id, etiqueta FROM relaciones WHERE origen_id IN ({placeholders})",
        ids_muestra,
    )
    relaciones_clientes = cur.fetchall()
    cur.execute("SELECT origen_id, destino_id, etiqueta FROM relaciones WHERE etiqueta='pertenece_a'")
    relaciones_catalogo = cur.fetchall()
    conn.close()

    G = nx.Graph()
    todos_nodos = nodos_catalogo + nodos_clientes
    for node_id, tipo, nombre, segmento in todos_nodos:
        G.add_node(node_id, tipo=tipo, nombre=nombre, segmento=segmento)
    for origen_id, destino_id, etiqueta in relaciones_clientes + relaciones_catalogo:
        if origen_id in G.nodes and destino_id in G.nodes:
            G.add_edge(origen_id, destino_id)

    # Layout en 3D (fuerza dirigida, como una red neuronal flotando en el espacio)
    pos = nx.spring_layout(G, dim=3, seed=42, k=0.6)

    # ---- Trazo de aristas ----
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    trazo_aristas = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(150,150,180,0.25)", width=1.5),
        hoverinfo="none",
    )

    # ---- Trazo de nodos, agrupados por color (tipo o segmento) ----
    trazos_nodos = []
    if colorear_por_segmento:
        grupos = {}
        for node_id, data in G.nodes(data=True):
            clave = f"Segmento {data['segmento']}" if data["tipo"] == "cliente" and data["segmento"] is not None else data["tipo"]
            grupos.setdefault(clave, []).append(node_id)
        paleta = px.colors.qualitative.Vivid
    else:
        grupos = {}
        for node_id, data in G.nodes(data=True):
            grupos.setdefault(data["tipo"], []).append(node_id)
        paleta_fija = {"cliente": "#4CAF50", "producto": "#2196F3", "categoria": "#FF9800", "preferencia": "#9C27B0"}

    for i, (etiqueta, ids) in enumerate(grupos.items()):
        xs = [pos[n][0] for n in ids]
        ys = [pos[n][1] for n in ids]
        zs = [pos[n][2] for n in ids]
        textos = [G.nodes[n]["nombre"] for n in ids]
        tamanos = [14 if G.nodes[n]["tipo"] != "cliente" else 5 for n in ids]
        color = paleta[i % len(paleta)] if colorear_por_segmento else paleta_fija.get(etiqueta, "#888888")

        trazos_nodos.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            name=str(etiqueta),
            text=textos,
            hoverinfo="text",
            marker=dict(size=tamanos, color=color, opacity=0.85, line=dict(width=0)),
        ))

    fig = go.Figure(data=[trazo_aristas] + trazos_nodos)
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def seccion_grafo_3d():
    st.subheader("🚀 Red 3D interactiva")
    st.caption("Arrastra para rotar, scroll para hacer zoom. Esta es la vista más 'demo-able' del proyecto.")

    n_nodos, _ = contar_registros()
    if n_nodos == 0:
        st.info("Genera datos primero en la pestaña 'Generar datos masivos'.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        tamano = st.slider("Tamaño de la muestra", 50, 1500, 400, step=50, key="slider_3d")
    with col2:
        hay_segmentos = st.session_state.get("df_clusters") is not None
        colorear_segmento = st.checkbox(
            "Colorear por segmento IA", value=False, disabled=not hay_segmentos,
            help="Corre primero la Segmentación IA para habilitar esta opción." if not hay_segmentos else None,
        )

    if st.button("🔄 Generar vista 3D", type="primary"):
        st.session_state["mostrar_3d"] = True

    if st.session_state.get("mostrar_3d"):
        construir_grafo_3d(tamano, colorear_por_segmento=colorear_segmento)


# ----------------------------------------------------------------------
# 12. APP PRINCIPAL
# ----------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Red de Clientes — Big Data", layout="wide")
    aplicar_estilo()
    st.title("🕸️ Red de Clientes y Conexiones")
    st.caption("Generación masiva → analítica → IA → visualización 3D → exportación")

    init_db()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎲 Generar datos masivos",
        "🚀 Red 3D",
        "📊 Dashboard de datos",
        "🤖 Segmentación IA",
        "🔗 Recomendaciones",
        "🌐 Red 2D (clásica)",
        "🗄️ Ver base de datos",
        "📤 Exportar datos",
    ])

    with tab1:
        seccion_generar_datos()
    with tab2:
        seccion_grafo_3d()
    with tab3:
        seccion_dashboard()
    with tab4:
        seccion_segmentacion_ia()
    with tab5:
        seccion_recomendaciones()
    with tab6:
        seccion_grafo()
    with tab7:
        seccion_ver_base_datos()
    with tab8:
        seccion_exportar()


if __name__ == "__main__":
    main()