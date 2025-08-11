
import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
import duckdb
conn = duckdb.connect()
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np



def crear_dataset(sell, productos):     
    duckdb.register('sell', sell)
    
    # 1. Cargás el dataset completo     
    df = conn.execute("""         
        SELECT              
            periodo,              
            product_id,              
            customer_id,             
            SUM(cust_request_qty) as cust_request_qty,             
            SUM(cust_request_tn) as cust_request_tn,             
            SUM(tn) AS tn_cliente         
        FROM sell         
        GROUP BY periodo, product_id, customer_id     
    """).fetchdf()      

    # 2. Calculás el total por periodo-producto     
    df_total = df.groupby(['periodo', 'product_id'])['tn_cliente'].sum().reset_index()     
    df_total = df_total.rename(columns={'tn_cliente': 'tn'})      

    # 3. Pivotás los clientes     
    df_pivot = df.pivot_table(         
        index=['periodo', 'product_id'],         
        columns='customer_id',         
        values='tn_cliente',         
        fill_value=0     
    ).reset_index()      

    # 4. Renombrás columnas (opcional)     
    df_pivot.columns = ['periodo', 'product_id'] + [f'cliente_{c}' for c in df_pivot.columns[2:]]      

    # 5. Unís el total     
    df_final = df_total.merge(df_pivot, on=['periodo', 'product_id'])     
    
    # Agrupamos por producto               
    duckdb.register('sell', df_final)     
    sell_group_product = conn.execute("""             
        SELECT              
            periodo,             
            product_id,             
            SUM(tn) as tn             
        FROM sell             
        GROUP BY periodo, product_id     
    """).fetchdf()          
    
    duckdb.register('sell_group_product', sell_group_product)          

    # ===== CORRECCIÓN: Generar solo períodos bimestrales válidos =====
    sell_group_product = conn.execute("""         
        WITH valid_periods AS (
            -- Generar solo períodos bimestrales válidos
            SELECT period_num AS periodo FROM (
                VALUES 
                    (201701),(201702),(201703),(201704),(201705),(201706),(201707),(201708),(201709),(201710),(201711),(201712),
                    (201801),(201802),(201803),(201804),(201805),(201806),(201807),(201808),(201809),(201810),(201811),(201812),
                    (201901),(201902),(201903),(201904),(201905),(201906),(201907),(201908),(201909),(201910),(201911),(201912)
            ) AS t(period_num)
        ),
        product_period_grid AS (             
            SELECT DISTINCT v.product_id, vp.periodo             
            FROM sell_group_product v             
            CROSS JOIN valid_periods vp         
        ),         
        ultimo_periodo_por_producto AS (             
            SELECT product_id, MAX(periodo) AS max_periodo             
            FROM sell_group_product             
            GROUP BY product_id         
        ),         
        combinaciones_a_rellenar AS (             
            SELECT ppg.product_id, ppg.periodo             
            FROM product_period_grid ppg             
            JOIN ultimo_periodo_por_producto upp             
            ON ppg.product_id = upp.product_id             
            WHERE ppg.periodo > upp.max_periodo         
        ),         
        datos_completos AS (             
            SELECT periodo, product_id, tn FROM sell_group_product             
            UNION ALL             
            SELECT periodo, product_id, 0.0 AS tn             
            FROM combinaciones_a_rellenar         
        )         
        SELECT *         
        FROM datos_completos         
        ORDER BY product_id, periodo;         
    """).fetchdf()          

    # clase tn + 2      
    sell_clase = conn.execute("""          
        SELECT *,         
            LEAD(tn, 2) OVER (PARTITION BY product_id ORDER BY periodo) as clase         
        FROM sell_group_product         
        ORDER BY product_id, periodo;          
    """).fetchdf()     
    
    df_final = df_final.drop(columns=['tn'])     
    sell_clase = sell_clase.merge(df_final, on=['periodo', 'product_id'], how='left')     
    sell_clase.fillna(0, inplace=True)     
    
    sell_clase = conn.execute("""          
        SELECT * from sell_clase         
        ORDER BY periodo;          
    """).fetchdf()          
    
    sell_clase = pd.merge(sell_clase, productos, on='product_id', how='left')          
    
    return sell_clase



def agregar_dolar(train):   
    # Descargar USD/ARS
    #ticker = yf.Ticker("USDARS=X")
    #hist = ticker.history(start="2017-01-01", end="2020-01-01")
    
    dolar_df = pd.read_csv('data/dolar.csv', sep=',')
    
    # Fix: Ensure the index is a proper DatetimeIndex
    #hist.index = pd.to_datetime(hist.index)
    """
    # Último día de cada mes
    dolar_mensual = hist['Close'].resample('M').last()
    
    # 1. Preparar datos del dólar
    # Convertir DatetimeIndex a período YYYYMM
    dolar_df = pd.DataFrame({
        'periodo': dolar_mensual.index.strftime('%Y%m').astype(int),
        'dolar': dolar_mensual.values
    })
    #dolar_df.to_csv("dolar.csv", index= False)
    """
    # 2. Hacer merge con train
    train = train.merge(dolar_df, on='periodo', how='left')
    
    # Calcular incremento porcentual del dólar vs mes anterior
    periodos_unicos = train[['periodo', 'dolar']].drop_duplicates().sort_values('periodo')
    periodos_unicos['dolar_incremento'] = periodos_unicos['dolar'].pct_change()

    # Merge de vuelta al dataset completo
    train = train.merge(
        periodos_unicos[['periodo', 'dolar_incremento']], 
        on='periodo', 
        how='left'
    )

    print(f"\nTrain inc dolar: {train.shape}")
    
    return train

# Variables políticas - PASO Agosto 2019 
def crear_variables_politicas(df):
    df = df.copy()
    
    # 1. Mes exacto de las PASO (agosto 2019)
    df['paso_agosto19'] = (df['periodo'] == 201908).astype(int)
    
    # 2. Período pre-electoral (4 meses antes: abril-julio 2019)  
    df['pre_electoral'] = df['periodo'].isin([201904, 201905, 201906, 201907]).astype(int)
    
    # 3. Período post-electoral (4 meses después: sep-dic 2019)
    df['post_electoral'] = df['periodo'].isin([201909, 201910, 201911, 201912]).astype(int)
    
    # 4. Distancia a la elección (en meses)
    df['meses_desde_paso'] = df['periodo'] - 201908
    
    # 5. Solo períodos post-PASO (más simple para usar)
    df['post_paso'] = (df['periodo'] > 201908).astype(int)
    
    return df

# Aplicar las variables
#train = crear_variables_politicas(train)

def crear_clustes_jerarquicos(clusters,train):      
    #clusters = pd.read_csv("hierarchical_results_50.csv", sep=',')
    # Opción 2: Si product_id ya es numérico pero con formato tensor
    #clusters3['product_id'] = clusters3['product_id'].astype(str).str.extract('(\d+)').astype(int)
    train= train.merge(clusters[['product_id', 'cluster']], on='product_id', how='left')
    
    return train    

def crear_clusters(clusters,train):   
    
    # Opción 2: Si product_id ya es numérico pero con formato tensor
    clusters['product_id'] = clusters['product_id'].astype(str).str.extract('(\d+)').astype(int)
    train = train.merge(clusters[['product_id', 'cluster']], on='product_id', how='left')
 
    return train   

def crear_clusters_dtw(clusters,train):
    train = train.merge(clusters[['product_id', 'cluster']], on='product_id', how='left')
    return train

# Particion para entrenamiento
def crear_particicion(train):
    train_mask = train['periodo'] <= 201908  # Hasta agosto
    val_mask = train['periodo'] == 201910   # Solo octubre
    X_train = train[train_mask].drop('clase', axis=1)
    y_train = train[train_mask]['clase']
    X_val = train[val_mask].drop('clase', axis=1)
    y_val = train[val_mask]['clase']

    return X_train,y_train,X_val,y_val

def analizar_ciclo_vida_productos(df):
    # Limpiar NaN en categorías primero
    df_clean = df.copy()
    df_clean['cat1'] = df_clean['cat1'].fillna('sin_categoria')
    df_clean['cat2'] = df_clean['cat2'].fillna('sin_categoria') 
    df_clean['cat3'] = df_clean['cat3'].fillna('sin_categoria')
    
    # Análisis básico por producto
    producto_stats = df_clean.groupby('product_id').agg({
        'periodo': ['min', 'max', 'count'],
        'tn': ['sum', 'mean', 'std', 'min', 'max'],
    }).round(4)
    
    # Aplanar columnas
    producto_stats.columns = ['_'.join(col) for col in producto_stats.columns]
    producto_stats = producto_stats.reset_index()
    
    # Renombrar
    producto_stats = producto_stats.rename(columns={
        'periodo_min': 'periodo_inicio',
        'periodo_max': 'periodo_muerte', 
        'periodo_count': 'meses_activos',
        'tn_sum': 'volumen_total',
        'tn_mean': 'volumen_promedio',
        'tn_std': 'volumen_volatilidad',
        'tn_min': 'volumen_min',
        'tn_max': 'volumen_max'
    })
    
    # CALCULAR PARTICIPACIONES por categorías
    df_extended = df_clean.copy()
    
    for periodo in df_clean['periodo'].unique():
        mask = df_extended['periodo'] == periodo
        periodo_data = df_extended[mask]
        
        # Total mercado
        total_mercado = periodo_data['tn'].sum()
        
        # Total por categorías
        total_cat1 = periodo_data.groupby('cat1')['tn'].sum()
        total_cat2 = periodo_data.groupby('cat2')['tn'].sum() 
        total_cat3 = periodo_data.groupby('cat3')['tn'].sum()
        
        # Calcular participaciones
        df_extended.loc[mask, 'market_share_total'] = df_extended.loc[mask, 'tn'] / total_mercado
        
        # Participaciones por categoría
        for idx in periodo_data.index:
            cat1_val = df_extended.loc[idx, 'cat1']
            cat2_val = df_extended.loc[idx, 'cat2'] 
            cat3_val = df_extended.loc[idx, 'cat3']
            
            df_extended.loc[idx, 'market_share_cat1'] = df_extended.loc[idx, 'tn'] / total_cat1[cat1_val]
            df_extended.loc[idx, 'market_share_cat2'] = df_extended.loc[idx, 'tn'] / total_cat2[cat2_val]
            df_extended.loc[idx, 'market_share_cat3'] = df_extended.loc[idx, 'tn'] / total_cat3[cat3_val]
    
    # Agregar participaciones promedio al stats
    share_stats = df_extended.groupby('product_id')[
        ['market_share_total', 'market_share_cat1', 'market_share_cat2', 'market_share_cat3']
    ].mean().round(6)
    
    # Merge
    resultado_final = producto_stats.merge(share_stats, on='product_id')
    
    return resultado_final

def agregar_features_avanzadas(stats_final, df_clean):
    stats_extended = stats_final.copy()
    
    # Calcular último período con ventas por producto (donde tn > 0)
    ultimo_periodo_por_producto = (
        df_clean[df_clean['tn'] > 0]
        .groupby('product_id')['periodo']
        .max()
        .to_dict()
    )
    
    # Calcular último período del dataset completo
    ultimo_periodo_dataset = df_clean['periodo'].max()
    
    # Agregar último período por producto al dataframe
    stats_extended['ultimo_periodo_con_ventas'] = stats_extended['product_id'].map(ultimo_periodo_por_producto)
    stats_extended['ultimo_periodo_dataset'] = ultimo_periodo_dataset
    
    # LAGS DE EVENTOS
    # Meses desde inicio (cuánto tiempo lleva en el mercado)
    def calcular_meses_diferencia(periodo_inicio, periodo_ref):
        inicio_year = periodo_inicio // 100
        inicio_month = periodo_inicio % 100
        ref_year = periodo_ref // 100
        ref_month = periodo_ref % 100
        return (ref_year - inicio_year) * 12 + (ref_month - inicio_month)
    
    # Usar el último período con ventas específico de cada producto
    stats_extended['meses_desde_inicio'] = stats_extended.apply(
        lambda row: calcular_meses_diferencia(row['periodo_inicio'], row['ultimo_periodo_con_ventas']), 
        axis=1
    )
    
    # Meses desde muerte (solo para productos muertos)
    # Un producto está muerto/discontinuado si:
    # 1. Tiene período de muerte definido Y
    # 2. Su último período con ventas NO es el último período del dataset
    stats_extended['esta_muerto'] = (
        (stats_extended['periodo_muerte'].notna()) & 
        (stats_extended['ultimo_periodo_con_ventas'] < ultimo_periodo_dataset)
    )
    
    stats_extended['meses_desde_muerte'] = stats_extended.apply(
        lambda row: calcular_meses_diferencia(row['periodo_muerte'], row['ultimo_periodo_con_ventas']) 
        if row['esta_muerto'] else 0, axis=1
    )
    
    # CLASIFICACIÓN POR MADUREZ
    def clasificar_madurez(row):
        if row['esta_muerto']:
            return 'discontinuado'
        elif row['meses_desde_inicio'] < 6:
            return 'nuevo'
        elif row['meses_desde_inicio'] < 12:
            return 'medio'
        else:
            return 'maduro'
    
    stats_extended['categoria_madurez'] = stats_extended.apply(clasificar_madurez, axis=1)
    
    # CARACTERÍSTICAS DE PERFORMANCE
    # Tendencia de crecimiento (comparar primeros vs últimos meses)
    tendencias = []
    aceleracion = []
    estacionalidad = []
    medias_anual = []
    
    for pid in stats_extended['product_id']:
        # Filtrar solo períodos con ventas (tn > 0)
        prod_data = df_clean[
            (df_clean['product_id'] == pid) & (df_clean['tn'] > 0)
        ].sort_values('periodo')
        
        if len(prod_data) >= 6:
            # Tendencia: primeros 3 vs últimos 3 meses CON VENTAS
            primeros_3 = prod_data['tn'].head(3).mean()
            ultimos_3 = prod_data['tn'].tail(3).mean()
            tendencia = (ultimos_3 - primeros_3) / primeros_3 if primeros_3 > 0 else 0
            
            # Aceleración: diferencia en crecimiento entre períodos
            if len(prod_data) >= 9:
                medio_3 = prod_data['tn'].iloc[3:6].mean()
                acel = ((ultimos_3 - medio_3) - (medio_3 - primeros_3)) / primeros_3 if primeros_3 > 0 else 0
            else:
                acel = 0
                
            
            # Estacionalidad: desviación de la tendencia
            trend_line = np.linspace(primeros_3, ultimos_3, len(prod_data))
            residuos = prod_data['tn'].values - trend_line
            estacional = np.std(residuos) / np.mean(prod_data['tn']) if np.mean(prod_data['tn']) > 0 else 0
            
        else:
            tendencia = 0
            acel = 0
            estacional = 0
        
        media_anual = prod_data['tn'].head(12).mean()
        
        medias_anual.append(media_anual)
        tendencias.append(tendencia)
        aceleracion.append(acel)
        estacionalidad.append(estacional)
    
    stats_extended['media_anual'] = medias_anual
    stats_extended['tendencia_crecimiento'] = tendencias
    stats_extended['aceleracion'] = aceleracion
    stats_extended['indice_estacionalidad'] = estacionalidad
    
    # CARACTERÍSTICAS DE VOLUMEN
    # Coeficiente de variación
    stats_extended['coef_variacion'] = (
        stats_extended['volumen_volatilidad'] / stats_extended['volumen_promedio']
    ).fillna(0)
    
    # Intensidad de ventas (volumen promedio vs duración)
    stats_extended['intensidad_ventas'] = stats_extended['volumen_total'] / stats_extended['meses_activos']
    
    # Momentum (ventas de últimos 3 meses CON VENTAS vs promedio histórico)
    momentum = []
    for pid in stats_extended['product_id']:
        prod_data = df_clean[
            (df_clean['product_id'] == pid) & (df_clean['tn'] > 0)
        ].sort_values('periodo')
        
        if len(prod_data) >= 6:
            ultimos_3 = prod_data['tn'].tail(3).mean()
            promedio_historico = prod_data['tn'].head(-3).mean()
            mom = ultimos_3 / promedio_historico if promedio_historico > 0 else 1
        else:
            mom = 1
        momentum.append(mom)
    
    stats_extended['momentum'] = momentum
    
    # LAGS DE EVENTOS ESPECÍFICOS (últimos 6 meses)
    # Productos que nacieron en últimos 6 meses
    stats_extended['es_recien_nacido'] = stats_extended['meses_desde_inicio'] <= 6
    
    # Productos que murieron en últimos 6 meses
    stats_extended['murio_recientemente'] = (
        stats_extended['esta_muerto'] & (stats_extended['meses_desde_muerte'] <= 6)
    )
    
    # Edad en trimestres (útil para features categóricas)
    stats_extended['edad_trimestres'] = (stats_extended['meses_desde_inicio'] / 3).astype(int)
    
    return stats_extended

def crear_ciclo_de_vida_producto(X_train):
    stats_train = analizar_ciclo_vida_productos(X_train)
    stats_train_completo = agregar_features_avanzadas(stats_train, X_train)

    # Merge con X_train
    X_train = X_train.merge(
        stats_train_completo, 
        on='product_id', 
        how='left'
    )
    print(f"X_train con todas las stats: {X_train.shape}")
    return X_train,stats_train_completo


def canaritos(X_train: pd.DataFrame, semilla: int, cantidad: int = 80) -> pd.DataFrame:
    """
    Agrega columnas 'canarito1' a 'canaritoN' con valores aleatorios uniformes entre 0 y 1.

    Args:
        X_train (pd.DataFrame): Dataset de entrenamiento al que se le agregarán los canaritos.
        semilla (int): Semilla para reproducibilidad.
        cantidad (int): Cantidad de columnas canarito a agregar. Default: 154.

    Returns:
        pd.DataFrame: Dataset con columnas canarito agregadas.
    """
    np.random.seed(semilla)
    for i in range(1, cantidad + 1):
        X_train[f'canarito{i}'] = np.random.uniform(0, 1, size=len(X_train))

    return X_train

def analizar_errores_por_producto(y_val, y_pred, X_val):
    # Asegurar que todos los arrays tengan el mismo tamaño
    min_len = min(len(y_val), len(y_pred), len(X_val))
    
    # Truncar todos a la misma longitud
    y_val_clean = y_val[:min_len]
    y_pred_clean = y_pred[:min_len]
    X_val_clean = X_val.iloc[:min_len]
    
    print(f"Usando {min_len} registros para análisis de errores")
    
    # Crear DataFrame con errores por producto
    df_errores = pd.DataFrame({
        'product_id': X_val_clean['product_id'].values,
        'y_real': y_val_clean,
        'y_pred': y_pred_clean,
        'error_abs': np.abs(y_val_clean - y_pred_clean),
        'error_rel': np.abs(y_val_clean - y_pred_clean) / np.abs(y_val_clean) * 100,  # Evitar división por 0
        'contribucion_error': np.abs(y_val_clean - y_pred_clean) / np.sum(np.abs(y_val_clean - y_pred_clean)) * 100
    })
    
    # Ordenar por contribución al error total
    df_errores = df_errores.sort_values('contribucion_error', ascending=False)
    
    return df_errores

# Usar la función corregida
#df_errores = analizar_errores_por_producto(y_val, y_pred, X_val)
#df_errores = df_errores.round(2)
#df_errores.to_csv("Errores.csv", index=False)

# Función para graficar los peores productos
def plot_peores_productos(df_errores, n=5):
   fig, axes = plt.subplots(n, 1, figsize=(12, 3*n))
   if n == 1:
       axes = [axes]
   
   top_productos = df_errores.head(n)
   
   for i, (idx, row) in enumerate(top_productos.iterrows()):
       pid = row['product_id']
       
       # Datos para el gráfico
       real = row['y_real']
       pred = row['y_pred']
       error_contrib = row['contribucion_error']
       
       # Gráfico de barras comparativo
       axes[i].bar(['Real', 'Predicho'], [real, pred], 
                  color=['blue', 'red'], alpha=0.7)
       axes[i].set_title(f'Producto {pid} - Contribuye {error_contrib:.1f}% al error total')
       axes[i].set_ylabel('Valor')
       
       # Agregar valores en las barras
       axes[i].text(0, real/2, f'{real:.1f}', ha='center', va='center')
       axes[i].text(1, pred/2, f'{pred:.1f}', ha='center', va='center')
   
   plt.tight_layout()
   plt.show()
   
   
def agregar_prophet_train(prophet,X_train):
   prophet = prophet.sort_values('periodo')
   # Merge train con prophet usando left join
   X_train = X_train.merge(
      prophet, 
      on=['periodo', 'product_id'], 
      how='left'
   )
   return X_train

def comparar_columnas(df1, df2, nombre1="df1", nombre2="df2"):
   # Obtener columnas de cada dataframe
   cols1 = set(df1.columns)
   cols2 = set(df2.columns)
   
   # Columnas que están en df1 pero no en df2
   solo_en_df1 = cols1 - cols2
   
   # Columnas que están en df2 pero no en df1
   solo_en_df2 = cols2 - cols1
   
   # Columnas en común
   comunes = cols1 & cols2
   
   print(f"Columnas solo en {nombre1}: {len(solo_en_df1)}")
   if solo_en_df1:
       print(list(solo_en_df1)[:10])  # Mostrar primeras 10
   
   print(f"\nColumnas solo en {nombre2}: {len(solo_en_df2)}")
   if solo_en_df2:
       print(list(solo_en_df2)[:10])  # Mostrar primeras 10
   
   #print(f"\nColumnas en común: {len(comunes)}")
   #solo_en_df1, solo_en_df2, comunes
   
   return solo_en_df1, solo_en_df2, comunes


def categorizar(X_train):
    categorical_cols = ['product_id','customer_id', 'cluster', 'cat1', 'cat2', 'cat3', 'brand', 'sku_size','categoria_madurez']

    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
        
            
    return X_train    


##Crear Clusters

# cluster prototipos con euclideana
#dtw= pd.read_csv("qv_euc_resultados_agosto_57.csv", sep=',')
#train=crear_clusters(dtw,train)

#Cluster dtw - jerarquicos
#clusters = pd.read_csv("clusters/hierarchical_results_agosto_50.csv", sep=',') 
def crear_set_entrenamiento(train):
    
    # Dólar
    
    train_1=agregar_dolar(train)
    
    # Cluster
    
    #clusters = pd.read_csv("clusters/cluster_dtw_qv_octubre/cluster_assignments_20250731_002450.csv", sep=',') 
    #train_1=crear_clusters_dtw(clusters,train_1)      
    
    #Cluster dtw - jerarquicos
    clusters = pd.read_csv("clusters/hierarchical_results_agosto_50.csv", sep=',') 
    train_1=crear_clustes_jerarquicos(clusters,train_1)
    
    ## Políticas 
       
       
    train_1 = crear_variables_politicas(train_1)
    
    ## Particionar 
    X_train,y_train,X_val,y_val=crear_particicion(train_1)
    X_train,stats_train_completo=crear_ciclo_de_vida_producto(X_train)
    
    ## Unos voladores y no son dolares
    #X_train=canaritos(X_train,1018)

    ## Tendencia y estacionalidad

    prophet_train = pd.read_csv("prophet/ph_agosto_train.csv", sep=',')
    X_train=agregar_prophet_train(prophet_train,X_train)

    ## Ver estas variables luego
    X_train = X_train.drop('descripcion', axis=1, errors='ignore')
    X_train = X_train.drop('ultimo_periodo_dataset', axis=1, errors='ignore')


    #=================== x_val =============================================
    ## Particionar 
    #X_train,y_train,X_val,y_val=crear_particicion(train_1)
    datos_octubre = train_1[train_1['periodo'] <= 201910]

    X_octubre,stats_octubre=crear_ciclo_de_vida_producto(datos_octubre)
    #X_val=crear_ciclo_de_vida_producto_val(X_val,stats_train_completo)

    ## Unos voladores y no son dolares
    #X_octubre=canaritos(X_octubre,1018)

    ## Tendencia y estacionalidad

    prophet_train = pd.read_csv("prophet/ph_octubre_train.csv", sep=',')
    X_octubre=agregar_prophet_train(prophet_train,X_octubre)



    ## Ver estas variables luego
    X_octubre = X_octubre.drop('descripcion', axis=1, errors='ignore')
    X_octubre = X_octubre.drop('ultimo_periodo_dataset', axis=1, errors='ignore')
    X_octubre = X_octubre.drop('clase', axis=1, errors='ignore')

    X_val= X_octubre[X_octubre['periodo'] == 201910]
    
    
    return X_train,y_train,X_val,y_val,X_octubre

def entrenar_baseline(model,X_train, y_train,X_val,y_val):
      model.fit(X_train, y_train)
      y_pred = model.predict(X_val)
      print(f"MSE: {mean_squared_error(y_val, y_pred)}")
      print(f"R2: {r2_score(y_val, y_pred)}")

      return y_val,y_pred
  
def forecast_error(y_val,y_pred):
    total_forecast_error = np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)

    print(f"Total Forecast Error: {total_forecast_error:.4f}")
    print(f"Total Forecast Error (%): {total_forecast_error * 100:.2f}%")  
