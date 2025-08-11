from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import funciones as fun

def entrenar_modelo_lgbm(model,X_train,y_train,X_val,y_val):

   model.fit(X_train, y_train)

  #bagging_fraction=0.9309021206979129, bagging_freq=10,
               #feature_fraction=0.5,
   #peso= ,sample_weight=sample_weights
  
   # Validar
   y_pred = model.predict(X_val)
   
   # Métricas
   mse = mean_squared_error(y_val, y_pred)
   rmse = np.sqrt(mse)
   mae = mean_absolute_error(y_val, y_pred)
   r2 = r2_score(y_val, y_pred)
   total_forecast_error = np.sum(np.abs(y_val - y_pred)) / np.sum(y_val) * 100
   
   print(f"R2: {r2:.4f}")
   print(f"RMSE: {rmse:.4f}")
   print(f"MAE: {mae:.4f}")
   print(f"Total Forecast Error: {total_forecast_error:.2f}%")
   
   return model, r2

"""
def agregar_features_train(df):
    
    # Para cada fila, agregar info agregada del período
    for periodo in df['periodo'].unique():
        mask_periodo = df['periodo'] == periodo
        
        # Total de ventas del período (todos los productos)
        total_periodo = df[mask_periodo]['tn'].sum()
        df.loc[mask_periodo, 'total_mercado_periodo'] = total_periodo
        
        # Participación del producto en el período
        df.loc[mask_periodo, 'market_share'] = df.loc[mask_periodo, 'tn'] / total_periodo
        
        # Total por cluster en el período
        # TOTAL Y PARTICIPACIÓN POR CATEGORÍAS
        for cat in ['cat3']:
           # Total por categoría en el período
           cat_totals = df[mask_periodo].groupby(cat)['tn'].sum()
           
           for cat_value in cat_totals.index:
               mask_cat = mask_periodo & (df[cat] == cat_value)
               
               # Total de la categoría
               #df.loc[mask_cat, f'total_{cat}_periodo'] = cat_totals[cat_value]
               
               # Participación dentro de la categoría
               df.loc[mask_cat, f'market_share_{cat}'] = df.loc[mask_cat, 'tn'] / cat_totals[cat_value]
               
               # Participación de la categoría vs mercado total
               #df.loc[mask_cat, f'cat_share_{cat}'] = cat_totals[cat_value] / total_periodo
       

        
        

        
    # Features de cambios en participación
    df['market_share_lag1'] = df.groupby('product_id')['market_share'].shift(1)
    df['market_share_change'] = df['market_share'] - df['market_share_lag1']
    
    # Cambios en totales de mercado
    df['total_mercado_lag1'] = df.groupby('product_id')['total_mercado_periodo'].shift(1)
    df['crecimiento_mercado'] = (df['total_mercado_periodo'] - df['total_mercado_lag1']) / df['total_mercado_lag1']
    
    return df

"""
def calcular_maximos_acumulados(df, ventanas=[12, 6, 3, 2]):
    """
    Calcula los máximos acumulados de TN para cada product_id en cada período
    considerando diferentes ventanas de tiempo hacia atrás
    PRESERVANDO EL ORDEN ORIGINAL DEL DATAFRAME
    
    Parameters:
    df: DataFrame con columnas 'product_id', 'periodo', 'tn'
    ventanas: lista de meses hacia atrás para calcular máximos (default: [12, 6, 3, 2])
    
    Returns:
    DataFrame original con columnas agregadas: max_12m, max_6m, max_3m, max_2m
    """
    
    # Preservar el orden original guardando el índice
    df_con_orden = df.reset_index(drop=True)
    df_con_orden['_orden_original'] = range(len(df_con_orden))
    
    # Diccionario para almacenar los resultados por ventana
    resultados_por_ventana = {}
    
    # Inicializar listas para cada ventana
    for ventana in ventanas:
        resultados_por_ventana[f'max_{ventana}m'] = []
    
    # Iterar por cada fila del DataFrame en su orden original
    for idx, row in df_con_orden.iterrows():
        product_id_actual = row['product_id']
        periodo_actual = row['periodo']
        
        # Para cada ventana de tiempo
        for ventana in ventanas:
            # Calcular periodo límite (ventana meses atrás desde el período actual)
            año = periodo_actual // 100
            mes = periodo_actual % 100
            
            # Restar la cantidad de meses de la ventana
            años_a_restar = (ventana - 1) // 12  # -1 porque incluimos el mes actual
            meses_a_restar = (ventana - 1) % 12
            
            if mes > meses_a_restar:
                año_limite = año - años_a_restar
                mes_limite = mes - meses_a_restar
            else:
                año_limite = año - años_a_restar - 1
                mes_limite = mes + 12 - meses_a_restar
            
            periodo_limite = año_limite * 100 + mes_limite
            
            # Filtrar datos para este product_id SIN reordenar el DataFrame completo
            # desde periodo_limite hasta periodo_actual (inclusive)
            datos_ventana = df_con_orden[
                (df_con_orden['product_id'] == product_id_actual) & 
                (df_con_orden['periodo'] >= periodo_limite) & 
                (df_con_orden['periodo'] <= periodo_actual)  # Incluir el período actual
            ]
            
            if len(datos_ventana) > 0:
                maximo_ventana = datos_ventana['tn'].max()
            else:
                maximo_ventana = np.nan
            
            # Agregar a la lista correspondiente
            resultados_por_ventana[f'max_{ventana}m'].append(
                round(maximo_ventana, 2) if pd.notna(maximo_ventana) else np.nan
            )
    
    # Crear resultado manteniendo el orden y tipos originales
    df_resultado = df.copy()  # Esto preserva tipos, orden e índice original
    
    for ventana in ventanas:
        df_resultado[f'max_{ventana}m'] = resultados_por_ventana[f'max_{ventana}m']
    
    return df_resultado

def calcular_brecha_por_periodo(df):
    """
    Calcula la brecha respetando el orden original del DataFrame (serie temporal)
    """
    
    # Preservar el orden original guardando el índice
    df_con_orden = df.reset_index(drop=True)
    df_con_orden['_orden_original'] = range(len(df_con_orden))
    
    # Trabajar con los datos (sin reordenar)
    brechas = []
    
    # Iterar respetando el orden original
    for idx, row in df_con_orden.iterrows():
        product_id_actual = row['product_id']
        periodo_actual = row['periodo']
        tn_actual = row['tn']
        
        # Calcular periodo límite (12 meses atrás)
        año = periodo_actual // 100
        mes = periodo_actual % 100
        
        if mes > 12:
            año_limite = año
            mes_limite = mes - 12
        else:
            año_limite = año - 1
            mes_limite = mes + 12 - 12
        
        periodo_limite = año_limite * 100 + mes_limite
        
        # Filtrar datos históricos SIN reordenar el DataFrame completo
        datos_historicos = df_con_orden[
            (df_con_orden['product_id'] == product_id_actual) & 
            (df_con_orden['periodo'] >= periodo_limite) & 
            (df_con_orden['periodo'] < periodo_actual)
        ]
        
        if len(datos_historicos) > 0:
            promedio_historico = datos_historicos['tn'].mean()
            brecha = tn_actual - promedio_historico
        else:
            brecha = np.nan
        
        brechas.append(round(brecha, 2) if pd.notna(brecha) else np.nan)
    
    # Crear resultado manteniendo el orden y tipos originales
    df_resultado = df.copy()  # Esto preserva tipos, orden e índice original
    df_resultado['brecha'] = brechas
    
    return df_resultado

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class GPUElasticityCalculator:
    """
    Calculadora de elasticidad optimizada para GPU/paralelización
    """
    
    def __init__(self, window_size=6, min_periods=3, use_gpu=True, use_multiprocessing=True):
        self.window_size = window_size
        self.min_periods = min_periods
        self.use_gpu = use_gpu
        self.use_multiprocessing = use_multiprocessing
        self.device = None
        
        # Detectar qué acelerar está disponible
        self._setup_acceleration()
    
    def _setup_acceleration(self):
        
        #Detecta y configura la mejor opción de aceleración disponible   
        
        # 2. Intentar Numba para paralelización CPU
        if self.use_multiprocessing:
            try:
                from numba import jit, prange
                self.device = 'numba'
                self.jit = jit
                self.prange = prange
                print("✅ Numba (CPU paralelo) configurado")
                return
            except ImportError:
                print("⚠️  Numba no disponible")
        
        # 3. Multiprocessing estándar
        try:
            import multiprocessing as mp
            self.device = 'multiprocessing'
            self.mp = mp
            print(f"✅ Multiprocessing configurado: {mp.cpu_count()} cores")
            return
        except ImportError:
            print("⚠️  Multiprocessing no disponible")
        
        # 4. Fallback a NumPy optimizado
        self.device = 'numpy'
        print("📝 Usando NumPy optimizado (fallback)")
    
    def fit_transform(self, df):
        """
        Calcula elasticidades usando la mejor aceleración disponible
        """
        print("🚀 INICIANDO CÁLCULO ACELERADO DE ELASTICIDADES")
        print("=" * 60)
        
        # Validar datos
        required_cols = ['periodo', 'product_id', 'tn']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada")
        
        # Análisis del dataset
        product_list = sorted(df['product_id'].unique())
        n_products = len(product_list)
        n_periods = df['periodo'].nunique()
        valid_periods = n_periods - self.window_size
        
        print(f"📊 DATASET INFO:")
        print(f"   - Productos: {n_products:,}")
        print(f"   - Períodos válidos: {valid_periods}")
        print(f"   - Aceleración: {self.device.upper()}")
        
        # Crear pivot
        print(f"\n⏳ Creando pivot table...")
        pivot_df = df.pivot_table(
            index='periodo', 
            columns='product_id', 
            values='tn', 
            fill_value=0
        )
        print(f"✅ Pivot completado - Shape: {pivot_df.shape}")
        
        # Calcular elasticidades según el método disponible
        print(f"\n🔄 CALCULANDO CON {self.device.upper()}...")
        
        if self.device == 'multiprocessing':
            elasticity_features = self._calculate_multiprocessing(pivot_df, product_list)
        else:
            elasticity_features = self._calculate_numpy_optimized(pivot_df, product_list)
        
        # Merge con dataset original
        print(f"\n🔗 Mergeando resultados...")
        result_df = df.merge(elasticity_features, on=['periodo', 'product_id'], how='left')
        
        elasticity_cols = [col for col in result_df.columns if col.startswith('elasticity_')]
        result_df[elasticity_cols] = result_df[elasticity_cols].fillna(0)
        
        print(f"✅ COMPLETADO - {len(elasticity_cols)} features generadas")
        return result_df
    
    
    
    def _calculate_multiprocessing(self, pivot_df, product_list):
        """
        Cálculo usando multiprocessing (versión corregida)
        """
        print("   ⚠️  Multiprocessing tiene limitaciones, usando NumPy optimizado...")
        return self._calculate_numpy_optimized(pivot_df, product_list)
    
    def _calculate_numpy_optimized(self, pivot_df, product_list):
        """
        Cálculo optimizado con NumPy vectorizado
        """
        data_matrix = pivot_df.values
        all_results = []
        
        for period_idx in range(self.window_size, len(pivot_df)):
            period = pivot_df.index[period_idx]
            window_data = data_matrix[period_idx-self.window_size:period_idx]
            
            # Cambios porcentuales vectorizados
            pct_changes = np.diff(window_data, axis=0) / (window_data[:-1] + 1e-8)
            
            # Matriz de correlaciones completa
            correlations = np.corrcoef(pct_changes.T)
            
            # Volatilidades
            volatilities = np.std(pct_changes, axis=0)
            volatility_ratios = volatilities[:, None] / (volatilities[None, :] + 1e-8)
            
            # Elasticidades
            elasticities_matrix = correlations * volatility_ratios
            elasticities_matrix = np.clip(elasticities_matrix, -10, 10)
            
            # Procesar resultados
            for i, product in enumerate(product_list):
                elasticities = {'periodo': period, 'product_id': product}
                
                for j, other_product in enumerate(product_list):
                    if i != j:
                        elasticities[f'elasticity_{other_product}'] = elasticities_matrix[i, j]
                    else:
                        elasticities[f'elasticity_{other_product}'] = 0.0
                
                product_elasticities = [elasticities_matrix[i, j] for j in range(len(product_list)) if i != j]
                elasticities.update(self._calculate_aggregate_features(product_elasticities))
                all_results.append(elasticities)
        
        return pd.DataFrame(all_results)
    
    def _calculate_single_elasticity(self, target_series, other_series):
        """
        Cálculo de elasticidad individual optimizado
        """
        try:
            target_pct = np.diff(target_series) / (target_series[:-1] + 1e-8)
            other_pct = np.diff(other_series) / (other_series[:-1] + 1e-8)
            
            if len(target_pct) < self.min_periods:
                return 0.0
            
            target_std = np.std(target_pct)
            other_std = np.std(other_pct)
            
            if target_std == 0 or other_std == 0:
                return 0.0
            
            correlation = np.corrcoef(target_pct, other_pct)[0, 1]
            if np.isnan(correlation):
                return 0.0
            
            volatility_ratio = target_std / other_std
            elasticity = correlation * volatility_ratio
            
            return np.clip(elasticity, -10, 10)
        except:
            return 0.0
    
    def _calculate_aggregate_features(self, elasticity_values):
        """
        Features agregados optimizados
        """
        if not elasticity_values:
            return {}
        
        ela_array = np.array(elasticity_values)
        
        return {
            'elasticity_mean': np.mean(ela_array),
            'elasticity_std': np.std(ela_array),
            'elasticity_max': np.max(ela_array),
            'elasticity_min': np.min(ela_array),
            'elasticity_negative_count': np.sum(ela_array < -0.5),
            'elasticity_positive_count': np.sum(ela_array > 0.5),
            'elasticity_neutral_count': np.sum(np.abs(ela_array) <= 0.5),
            'elasticity_cannibalization_risk': np.sum(ela_array < -1.0),
        }

# Función principal
def create_elasticity_features_accelerated(df, window_size=6, min_periods=3, use_gpu=True, use_multiprocessing=True):
    """
    Cálculo acelerado de elasticidades cruzadas
    
    Parameters:
    - df: DataFrame con datos
    - window_size: Ventana temporal
    - min_periods: Períodos mínimos
    - use_gpu: Intentar usar GPU si está disponible
    - use_multiprocessing: Usar paralelización CPU
    
    Returns:
    - DataFrame con features de elasticidad
    """
    calculator = GPUElasticityCalculator(window_size, min_periods, use_gpu, use_multiprocessing)
    return calculator.fit_transform(df)

# Para instalación rápida de dependencias
def install_acceleration_packages():
    """
    Guía de instalación de paquetes de aceleración
    """
    print("📦 INSTALACIÓN DE PAQUETES DE ACELERACIÓN:")
    print("")
    print("1. Para GPU (CuPy):")
    print("   conda install -c conda-forge cupy")
    print("   pip install cudf-cu11  # o cu12 según tu CUDA")
    print("")
    print("2. Para PyTorch GPU:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("")
    print("3. Para Numba (CPU paralelo):")
    print("   conda install numba")
    print("   # o")
    print("   pip install numba")
    print("")
    print("4. Multiprocessing viene incluido con Python")

def crear_features_train_lags(df):
    """
    Función completa para crear features de lags, incrementos y ratios entre productos
    Reemplazado df_feature_creation por df y agregados ratios entre productos
    """
    print("Iniciando creación completa de features...")
    print(f"DataFrame inicial: {df.shape}")
    
    # Feature básica con dólar
    df['tnxdolar'] = df['dolar'] / df['tn']
    df['dolar_lag2']=df['dolar'].shift(2)
    df['tnxdolar'] = df['dolar_lag2'] / df['tn']
    
    
    # 2. Lags hasta donde haya datos disponibles
    #print("Creando lags...")
    #for lag in range(1, 36):  # Hasta 24 meses de lag
    #    df[f'tn_lag{lag}'] = df.groupby('product_id')['tn'].shift(lag)
    
    
    lags = [1,2,3,4,5,6,7,8,9,10,11,12,20,24]  # puedes elegir según estacionalidad o importancia previa

    # Crear solo esos lags
    for lag in lags:
        df[f'tn_lag{lag}'] = df.groupby('product_id')['tn'].shift(lag)
    
    
    max_lag=24
    
    
    
    clientes=[10001,10002,10003,10004,10005,10006,10007,10008,10009,10010,10011,
    10012,10013,10014,10015,10016,10017,10018,10019,10020,10021,10022,10023,10024,10025]
    
    
    clientes_columnas = [f"cliente_{cid}" for cid in clientes]
    
  
    
    
    for cliente in clientes_columnas:
        if cliente in df.columns:
            print(f"Procesando {cliente}...")
            # Crear lags para este cliente agrupando por product_id
            #for lag in range(1, max_lag + 1):
            for lag in lags:    
                nombre_lag = f'{cliente}_lag{lag}'
                df[nombre_lag] = df.groupby('product_id')[cliente].shift(lag)
     
    # 3. Medias móviles hasta 12 meses (con manejo de errores)
    print("Creando medias móviles...")
    #for window in range(2, 37):
    for window in lags:
        try:
            df[f'tn_rolling_mean_{window}'] = (
                df.groupby('product_id')['tn']
                .rolling(window, min_periods=1)  # min_periods=1 evita el error
                .mean()
                .reset_index(0, drop=True)
            )
        except:
            df[f'tn_rolling_mean_{window}'] = 0

    # 4. Desvíos móviles hasta 12 meses (con manejo de errores)
    print("Creando desvíos móviles...")
    for window in lags:
        try:
            df[f'tn_rolling_std_{window}'] = (
                df.groupby('product_id')['tn']
                .rolling(window, min_periods=1)  # min_periods=1 evita el error
                .std()
                .reset_index(0, drop=True)
            )
        except:
            df[f'tn_rolling_std_{window}'] = 0
            
    # Agregar mediana a las features de entrenamiento también
    print("Creando medianas móviles...")
    for window in lags:
        try:
            df[f'tn_rolling_median_{window}'] = (
                df.groupby('product_id')['tn']
                .rolling(window, min_periods=1)  # min_periods=1 evita el error
                .median()
                .reset_index(0, drop=True)
            )
        except:
            df[f'tn_rolling_median_{window}'] = 0        

    # 5. Features temporales
    print("Creando features temporales...")
    df['month'] = df['periodo'] % 100
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['quarter'] = ((df['periodo'] % 100 - 1) // 3) + 1
    
    # Incrementos por períodos
    print("Creando incrementos por períodos...")
    # Incremento mes a mes (1 período)
    df['incremento_mensual'] = df.groupby('product_id')['tn'].pct_change(1)

    # Incremento trimestral (3 períodos)
    df['incremento_trimestral'] = df.groupby('product_id')['tn'].pct_change(3)

    # Incremento semestral (6 períodos) 
    df['incremento_semestral'] = df.groupby('product_id')['tn'].pct_change(6)

    # Incremento anual (12 períodos)
    df['incremento_anual'] = df.groupby('product_id')['tn'].pct_change(12)

    # Crear lags manuales para incrementos específicos
    print("Creando lags manuales...")
    grupo = df.groupby('product_id')['tn']
    lag_01 = grupo.shift(1)
    lag_02 = grupo.shift(2)
    lag_03 = grupo.shift(3)
    lag_04 = grupo.shift(4) 
    lag_05 = grupo.shift(5)
    lag_06 = grupo.shift(6)
    lag_07 = grupo.shift(7)
    lag_08 = grupo.shift(8)
    lag_09 = grupo.shift(9)    
    lag_10 = grupo.shift(10)
    lag_11 = grupo.shift(11)
    lag_12 = grupo.shift(12)
    lag_13 = grupo.shift(13)
    
    # Features de incrementos específicos
    print("Creando features de incrementos específicos...")
    
    df['inc_13y12'] = (lag_12 - lag_13) / lag_13
    df['inc_12y11'] = (lag_11 - lag_12) / lag_12
    df['inc_11y10'] = (lag_10 - lag_11) / lag_11
    df['inc_10y09'] = (lag_09 - lag_10) / lag_10
    df['inc_09y08'] = (lag_08 - lag_09) / lag_09
    df['inc_08y07'] = (lag_07 - lag_08) / lag_08
    df['inc_07y06'] = (lag_06 - lag_07) / lag_07
    df['inc_06y05'] = (lag_05 - lag_06) / lag_06
    df['inc_05y04'] = (lag_04 - lag_05) / lag_05
    df['inc_04y03'] = (lag_03 - lag_04) / lag_04
    df['inc_03y02'] = (lag_02 - lag_03) / lag_03
    df['inc_02y01'] = (lag_01 - lag_02) / lag_02
    
    
    df['inc_12y10'] = (lag_10 - lag_12) / lag_12
    df['tn_lag10_2']= df['tn_lag10']**2
    df['tn_lag12_2']= df['tn_lag12']**2
    return df


def agregar_features_train(df):
    
    # Para cada fila, agregar info agregada del período
    for periodo in df['periodo'].unique():
        mask_periodo = df['periodo'] == periodo
        
        # Total de ventas del período (todos los productos)
        total_periodo = df[mask_periodo]['tn'].sum()
        df.loc[mask_periodo, 'total_mercado_periodo'] = total_periodo
        
        # Participación del producto en el período
        df.loc[mask_periodo, 'market_share'] = df.loc[mask_periodo, 'tn'] / total_periodo
        
        # Total por cluster en el período
        # TOTAL Y PARTICIPACIÓN POR CATEGORÍAS
        for cat in ['cat3', 'cat2', 'cat1' ]:
           # Total por categoría en el período
           cat_totals = df[mask_periodo].groupby(cat)['tn'].sum() #V23
           
           for cat_value in cat_totals.index:
               mask_cat = mask_periodo & (df[cat] == cat_value)
               
               # Total de la categoría
               df.loc[mask_cat, f'total_{cat}_periodo'] = cat_totals[cat_value]    #V1 IMPORTANTE
               
               # Participación dentro de la categoría
               df.loc[mask_cat, f'market_share_{cat}'] = df.loc[mask_cat, 'tn'] / cat_totals[cat_value]   #V2
               
               # Participación de la categoría vs mercado total
               #df.loc[mask_cat, f'cat_share_{cat}'] = cat_totals[cat_value] / total_periodo  
               df.loc[mask_cat, f'{cat}_share_vs_market'] = cat_totals[cat_value] / total_periodo    ## V3 IMPORTANTE


    #df['rel_share_intensity_3'] = df['market_share'] / df['cat3_share_vs_market'] # V4
    #df['rel_share_intensity_2'] = df['market_share'] / df['cat2_share_vs_market'] # V5
    #df['rel_share_intensity_1'] = df['market_share'] / df['cat1_share_vs_market'] # V6

    # Features de cambios en participación
    df['market_share_lag1'] = df.groupby('product_id')['market_share'].shift(1) #V7 IMPORTANTE
    df['market_share_change'] = df['market_share'] - df['market_share_lag1']  #V8

    df['total_cat3_lag1'] = df.groupby('cat3')['total_cat3_periodo'].shift(1) ## V9 IMPORTANTE
    #df['total_cat2_lag1'] = df.groupby('cat2')['total_cat2_periodo'].shift(1) ## V10
    #df['total_cat1_lag1'] = df.groupby('cat1')['total_cat1_periodo'].shift(1) ## V11

    # Cambios en totales de mercado
    df['total_mercado_lag1'] = df.groupby('product_id')['total_mercado_periodo'].shift(1) # V12 
    df['crecimiento_mercado'] = (df['total_mercado_periodo'] - df['total_mercado_lag1']) / df['total_mercado_lag1'] # V13


    ### NUEVAS

    # Cambio en total de categoría
    df['total_cat3_lag1'] = df.groupby('product_id')['total_mercado_periodo'].shift(1)  # si cada product_id solo pertenece a una cat3, o agrupar por cat3 → .groupby(['cat3'])  #V14
    df['crecimiento_cat3'] = (df['market_share_cat3'] - df.groupby('product_id')['market_share_cat3'].shift(1)) #V15
    df['crecimiento_cat2'] = (df['market_share_cat2'] - df.groupby('product_id')['market_share_cat2'].shift(1)) #V16
    #f['crecimiento_cat1'] = (df['market_share_cat1'] - df.groupby('product_id')['market_share_cat1'].shift(1)) #V17

    # Ratio market_share / cat3_share_vs_market (intensidad relativa. Cuan grande es la participación de un producto en relación a su categoría)
    df['rel_share_intensity_3'] = df['market_share'] / df['cat3_share_vs_market'] # V18
    df['rel_share_intensity_2'] = df['market_share'] / df['cat2_share_vs_market'] # V19
    df['rel_share_intensity_1'] = df['market_share'] / df['cat1_share_vs_market'] # V20

    # Variables globales CAT1
    #print("Agregando variables globales CAT1/2...")
    for dept in ['FOODS', 'HC', 'PC', 'REF']:
        df[f'is_dept_{dept}'] = (df['cat1'] == dept).astype(int) #V21

    for st in ['ADEREZOS', 'CABELLO', 'DENTAL','DEOS','HOGAR','OTROS','PIEL1','PIEL2','PROFESIONAL','ROPA ACONDICIONADOR','ROPA LAVADO','ROPA MANCHAS','SOPAS Y CALDOS','TE','VAJILLA']:
        df[f'es_{st}'] = (df['cat2'] == st).astype(int) #V22

    return df


    
def crear_diciembre(train):    
    train_3=fun.agregar_dolar(train)

    clusters = pd.read_csv("clusters/cluster_dtw_qv_octubre/cluster_assignments_20250731_002450.csv", sep=',') ####cluster hasta dici
    train_3=fun.crear_clusters_dtw(clusters,train_3)      
    
    
    ## Agregar variables políticas    
    #train = crear_variables_politicas(train)


    ## Particionar 
    #X_train,y_train,X_val,y_val=crear_particicion(train_1)
    datos_diciembre = train_3[train_3['periodo'] <= 201912]

    X_diciembre,stats_octubre=fun.crear_ciclo_de_vida_producto(datos_diciembre)


    ## Tendencia y estacionalidad

    prophet_train = pd.read_csv("prophet/ph_completo.csv", sep=',')
    X_diciembre=fun.agregar_prophet_train(prophet_train,X_diciembre)


    ## Ver estas variables luego
    X_diciembre = X_diciembre.drop('descripcion', axis=1, errors='ignore')
    X_diciembre = X_diciembre.drop('ultimo_periodo_dataset', axis=1, errors='ignore')
    
    return X_diciembre

def crear_particicion_final(train,periodo):
    train_mask = train['periodo'] <= periodo # Hasta agosto
    #val_mask = train['periodo'] == 201910   # Solo octubre
    X_train = train[train_mask].drop('clase', axis=1)
    y_train = train[train_mask]['clase']
    #X_val = train[val_mask].drop('clase', axis=1)
    #y_val = train[val_mask]['clase']

    return X_train,y_train

def entrenar_modelo_podado_lgbm(model_podado,X_train_podado,y_train,X_val_podado,y_val,X_train,features_disponibles):


    #sample_weights = np.log1p(X_train_podado['tn'])



    print("Entrenando modelo podado...")
    
    model_podado.fit(X_train_podado, y_train)
    y_pred_podado = model_podado.predict(X_val_podado)

    # Comparar métricas
    r2_podado = r2_score(y_val, y_pred_podado)
    rmse_podado = np.sqrt(mean_squared_error(y_val, y_pred_podado))
    mae_podado = mean_absolute_error(y_val, y_pred_podado)

    print(f"\n=== MODELO PODADO (features) ===")
    print(f"R2: {r2_podado:.4f}")
    print(f"RMSE: {rmse_podado:.4f}")
    print(f"MAE: {mae_podado:.4f}")

    # Feature importance del modelo podado
    feature_importance_podado = model_podado.feature_importances_
    print(f"\n=== FEATURE IMPORTANCE MODELO PODADO ===")
    for i, feature in enumerate(features_disponibles):
        print(f"{feature}: {feature_importance_podado[i]:.0f}")


    # Si quieres comparar con el modelo original
    print(f"\n=== COMPARACIÓN ===")
    print(f"Features: Original {X_train.shape[1]} → Podado {X_train_podado.shape[1]}")
    return model_podado,X_train_podado,y_train,X_val_podado,y_val, y_pred_podado,X_train,feature_importance_podado,features_disponibles

def podar(threshold,feature_importances_,X_train,X_val):
    
    min_len = min(len(X_train.columns), len(feature_importances_))

    feature_df = pd.DataFrame({
        'feature': X_train.columns[:min_len],
        'importance': feature_importances_[:min_len]
    }).sort_values('importance', ascending=False)



    #threshold = 48
    features_to_keep = feature_df[feature_df['importance'] >= threshold]['feature'].tolist()
    
    features_disponibles = [f for f in features_to_keep if f in X_train.columns and f in X_val.columns]
    
    # borra los canarios
    
    features_disponibles = [f for f in features_disponibles if not f.startswith('canarito')]
    
    return features_disponibles    