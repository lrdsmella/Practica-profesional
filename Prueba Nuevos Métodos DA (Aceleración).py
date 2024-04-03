# Databricks notebook source
# MAGIC %md
# MAGIC # Librerías y Funciones

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Librerías

# COMMAND ----------

!pip install --upgrade scipy
!pip install ruptures
!pip install --upgrade --no-deps statsmodels
!pip install pystan~=2.14
!pip install fbprophet

import datetime as dt
import itertools
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import pyspark.pandas as ps
import re
import requests
import seaborn as sns
import time
import ruptures as rpt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, DoubleType, IntegerType, StringType, StructType, StructField, MapType, TimestampType
from scipy import stats
from scipy.optimize import Bounds, minimize
from sklearn import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import logit
from sklearn.ensemble import IsolationForest
from statsmodels.compat import lzip
from scipy.stats import shapiro, kstest
from scipy import stats
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from dateutil.parser import parse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.WARNING)

%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 3)
plt.style.use('seaborn-bright')

plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = True

# Habilita cache en cluster Databricks ##
spark.conf.set("spark.databricks.io.cache.enabled", "true")

sqlContext.clearCache()

# Enable Arrow-based columnar data transfers
# spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Funciones: Bases de Datos

# COMMAND ----------

def get_alerts(env):
  if env == 'prod':
    conexion = {'HOST':'psql-drt-manto-prod-eastus2-001.postgres.database.azure.com',
                'PORT':'5432',
                'USER':'ds_user@psql-drt-manto-prod-eastus2-001',
                'PASS':'9MBshSvtGqWDJcth5e8kp89RWdnzmu',
                'DB':'analyticsdb'
               }
  else:
    conexion = {'HOST':'psql-drt-manto-dev-eastus2-001.postgres.database.azure.com',
                'PORT':'5432',
                'USER':'app_user@psql-drt-manto-dev-eastus2-001',
                'PASS':'mgNjSYpFhQ6tTEAMA3EjssDFurBaAT',
                'DB':'analyticsdb' #,
              # 'SSLMODE':'require'
               }
  connstr = "host=%s port=%s user=%s password=%s dbname=%s " % (conexion['HOST'], conexion['PORT'], conexion['USER'], conexion['PASS'], conexion['DB'])
  #sslmode=%s , conexion['SSLMODE']
  conn = psycopg2.connect(connstr)
  recom = pd.read_sql("""
  select * from public.recommendations base
  JOIN new_jerarquia nj ON nj.i_indicador = base.hierarchy_id::double precision
  """, con = conn)
  return recom

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Funciones: Gráficos

# COMMAND ----------

def plot_serietemp(df):
        
    df.set_index('fh', inplace = True)
    df['fh'] = df.index
    
    plt.style.use('seaborn-darkgrid')
    
    plt.figure(figsize = (12, 6))
    plt.xlabel('Fecha', fontsize = 14)
    plt.ylabel('RMS', fontsize = 12)
    
    plt.plot(df['fh'], df['RMS'])
    
    plt.title('Serie Temporal RMS', fontsize = 16)
    plt.show()
    
def plot_outliersIF(df, outliers):

    plt.figure(figsize = (16, 8))

    plt.plot(df['RMS'], marker = '.')
    plt.plot(outliers['RMS'], 'o', color = 'red', label = 'outlier')
    plt.title('Detección por Isolation Forest')
      
    plt.xlabel('Fecha')
    plt.ylabel('RMS')
    plt.legend()
    
def regression_plot(df, modelolr, modelowr):

    #Gráfica de dispersión y rectas de regresión
    plt.figure(figsize=(10, 5.5))
    plt.scatter(df.loc[:, ['Time']], df.loc[:, 'RMS'], label='Datos')
    plt.plot(df.loc[:, ['Time']], modelolr.predict(), "r--.", label='LR')
    plt.plot(df.loc[:, ['Time']], modelowr.fit().fittedvalues, "g--.", label='WR')
    plt.xlabel('Tiempo')
    plt.ylabel('RMS')
    plt.title('Dispersión y rectas de regresión')
    plt.legend(loc='best')
    
def resid_plots(df, modelo):

    #Gráficas RMS observado y predicho vs residuos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,4))
    fig.suptitle('RMS vs Residuos')
    
    sns.residplot(x=df.loc[:, ['RMS']],y=modelo.resid,lowess=True, scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, order=1, ax=ax1)
    ax1.set(xlabel='Valores observados RMS', ylabel='Residuos modelo')

    sns.residplot(x=modelo.predict(),y=modelo.resid,lowess=True, scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, order=1, ax=ax2)
    ax2.set(xlabel='Valores predichos RMS', ylabel='Residuos modelo')
    
def dist_resid(modelo):
  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,4))
    fig.suptitle('Distribución de los residuos')
    
    #histograma
    ax1.hist(x=modelo.resid, color="orange")
    ax1.set(xlabel='Residuos modelo', ylabel='Frecuencia')
    #qq plot
    sm.qqplot(model.resid, line ='q', ax=ax2)

def time_resid(df, modelo):
    
    plt.scatter(df.loc[:,'Time'], modelo.resid)
    plt.axhline(0, color='red')
    plt.xlabel('Tiempo')
    plt.ylabel('Residuos')
    plt.title('Tiempo vs Residuos')
    
def boxplot_pendiente(alertasaceptadas, alertasrechazadas):

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.boxplot(alertasaceptadas['pendiente'], positions=[0], widths=0.5)


    ax.boxplot(alertasrechazadas['pendiente'], positions=[1], widths=0.5)


    ax.set_xticks([0,1])
    ax.set_xticklabels(['Alertas aceptadas', 'Alertas rechazadas'])
    ax.set_ylabel('Pendiente')
    ax.set_title('Boxplots pendiente')

    plt.show()
    
def histograma_kde_pendientes(alertasrechazadas, alertasaceptadas):
   
    fig = plt.figure(figsize=(10, 5.5))
    #Histograma
    sns.histplot(data=alertasrechazadas, x='pendiente', alpha=0.5, color='black', label='Pendientes_rechazadas')
    sns.histplot(data=alertasaceptadas, x='pendiente', alpha=0.5, color='orange', label='Pendientes_aceptadas')

    #Agregar KDE para estimar la distribución
    sns.kdeplot(data=alertasrechazadas, x='pendiente', color='black', label='KDE Pendientes_rechazadas')
    sns.kdeplot(data=alertasaceptadas, x='pendiente', color='orange', label='KDE Pendientes_aceptadas')
   
    plt.xlim(0, alertasaceptadas['pendiente'].max())
    plt.legend()
    plt.show()
    
def boxplot_angulo(alertasaceptadas, alertasrechazadas):

    alertasaceptadas['ángulo']=np.arctan(alertasaceptadas['pendiente'])
    alertasrechazadas['ángulo']=np.arctan(alertasrechazadas['pendiente'])

    fig, ax = plt.subplots()

    ax.boxplot(alertasaceptadas['ángulo'], positions=[0], widths=0.5)


    ax.boxplot(alertasrechazadas['ángulo'], positions=[1], widths=0.5)


    ax.set_xticks([0,1])
    ax.set_xticklabels(['Alertas aceptadas', 'Alertas rechazadas'])
    ax.set_ylabel('Ángulo')
    ax.set_title('Boxplots ángulo')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Funciones: Creación de Variables

# COMMAND ----------

def get_comment_type(df_recom, temperature = False):
  # Función que estandariza los comentarios escritos por los expertos en confiabilidad. Previo al proceso de estandarización,
  # se realizan los siguientes pasos:
  # 1) Convertimos todos los caracteres a minúscula.
  # 2) Convertimos caracteres especiales ('ñ' -> 'n') y acentos gráficos ('ó' -> 'o', por ejemplo).
  # 3) Símbolos de puntuación y caracteres numéricos quedan intactos.
  comments = (df_recom['comment'].str.lower()
              .str.normalize('NFKD')
              .str.encode('ascii', errors = 'ignore')
              .str.decode('utf-8'))

  # Creamos una variable ('comment_type') en 'df_recom' donde almacenaremos los distintos tipos de comentarios. Cabe destacar
  # lo siguiente:
  #
  #   1) Un registro puede tener más de un tipo de comentario asociado.
  #   2) Cada tipo de comentario se identifica a partir de un conjunto de palabras clave.
  #
  # Para facilitar la manipulación de los tipos de comentarios, cada entrada en 'comment_type' es 'Python list', cuyos elementos
  # son de clase 'string'.
  df_recom['comment_type'] = pd.array([''] * len(df_recom), dtype = 'string')
  df_recom['comment_type'] = df_recom['comment_type'].apply(lambda x: [x])
  
  # Alertas aceptadas.
  df_recom['alerta_aceptada'] = 0
  condicion = (df_recom['status'] == 'Approved')
  df_recom.loc[condicion, 'alerta_aceptada'] = 1
  df_recom['comment_type'] = (df_recom[['comment_type', 'alerta_aceptada']]
                              .apply(lambda x: x[0] + ['Alerta aceptada'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por medición errónea.
  df_recom['medicion_erronea'] = 0
  claves = ['medicion erronea']
  condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
  df_recom.loc[condicion, 'medicion_erronea'] = 1
  df_recom['comment_type'] = (df_recom[['comment_type', 'medicion_erronea']]
                              .apply(lambda x: x[0] + ['Medicion erronea'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por tendencia estable.
  if temperature == True:
    df_recom['tendencia_estable'] = 0
    claves = ['tendencia estable', 'rangos normales', 'estable', 'leve alza', 'no se observa', 'no se observan']
    condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
    df_recom.loc[condicion, 'tendencia_estable'] = 1
    df_recom['comment_type'] = (df_recom[['comment_type', 'tendencia_estable']]
                                .apply(lambda x: x[0] + ['Tendencia estable'] if x[1] == 1 else x[0] + [''], axis = 1))
  else:
    df_recom['tendencia_estable'] = 0
    claves = ['tendencia estable', 'rangos normales', 'estable', 'leve alza']
    condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
    df_recom.loc[condicion, 'tendencia_estable'] = 1
    df_recom['comment_type'] = (df_recom[['comment_type', 'tendencia_estable']]
                                .apply(lambda x: x[0] + ['Tendencia estable'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por alza puntual (con posterior normalización).
  df_recom['alza_puntual'] = 0
  claves = ['alza puntual', 'normalizacion', 'alzas puntuales']
  condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
  df_recom.loc[condicion, 'alza_puntual'] = 1
  df_recom['comment_type'] = (df_recom[['comment_type', 'alza_puntual']]
                              .apply(lambda x: x[0] + ['Alza puntual'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por fecha desfasada.
  df_recom['alerta_desfasada'] = 0
  claves = ['desfasada', r'\benero\b', r'\bfebrero\b', r'\bmarzo\b', r'\babril\b', r'\bmayo\b', r'\bjunio\b',
            r'\bjulio\b', r'\bagosto\b', r'\bseptiembre\b', r'\boctubre\b', r'\bnoviembre\b', r'\bdiciembre\b']
  condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
  df_recom.loc[condicion, 'alerta_desfasada'] = 1
  df_recom['comment_type'] = (df_recom[['comment_type', 'alerta_desfasada']]
                              .apply(lambda x: x[0] + ['Alerta desfasada'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por bajo voltaje en los equipos.
  df_recom['bajo_voltaje'] = 0
  claves = ['bajo voltaje']
  condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
  df_recom.loc[condicion, 'bajo_voltaje'] = 1
  df_recom['comment_type'] = (df_recom[['comment_type', 'bajo_voltaje']]
                              .apply(lambda x: x[0] + ['Bajo voltaje'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por señalar rangos de frecuencia incorrectos/inexistentes.
  if temperature == False:
    df_recom['rango_incorrecto'] = 0
    claves = ['no corresponde', 'no en el rango', 'no se observa', 'ni se observa', 'no existe', 'el alza corresponde a', 
              'no se observan', 'no indica', 'no coinciden', 'incorrecto', 'no se detecta', 'no hay componentes']
    condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
    df_recom.loc[condicion, 'rango_incorrecto'] = 1
    df_recom['comment_type'] = (df_recom[['comment_type', 'rango_incorrecto']]
                                .apply(lambda x: x[0] + ['Rango incorrecto'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por equipo detenido previo a la emisión de la alerta.
  df_recom['equipo_detenido'] = 0
  claves = ['detencion', 'detenido']
  condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
  df_recom.loc[condicion, 'equipo_detenido'] = 1
  df_recom['comment_type'] = (df_recom[['comment_type', 'equipo_detenido']]
                              .apply(lambda x: x[0] + ['Equipo detenido'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas por tendencia a la baja.
  df_recom['tendencia_baja'] = 0
  claves = ['a la baja']
  condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
  df_recom.loc[condicion, 'tendencia_baja'] = 1
  df_recom['comment_type'] = (df_recom[['comment_type', 'tendencia_baja']]
                              .apply(lambda x: x[0] + ['Tendencia a la baja'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Alertas rechazadas que no cargan en el sistema.
  if temperature == True:
    df_recom['no_carga'] = 0
    claves = ['no carga', 'no se visualiza', 'no se visualizan', 'no se puede', 'no se pueden', 'no es posible']
    condicion = (df_recom['status'] == 'Declined') & (comments.str.contains('|'.join(claves)) == True)
    df_recom.loc[condicion, 'no_carga'] = 1
    df_recom['comment_type'] = (df_recom[['comment_type', 'no_carga']]
                                .apply(lambda x: x[0] + ['No carga'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Almacenamos todos los tipos de comentarios.
  if temperature == False:
    list_comments = ['alerta_aceptada', 'medicion_erronea', 'tendencia_estable', 'alza_puntual', 'alerta_desfasada', 
                   'bajo_voltaje', 'rango_incorrecto', 'equipo_detenido', 'tendencia_baja']
  else:
    list_comments = ['alerta_aceptada', 'medicion_erronea', 'tendencia_estable', 'alza_puntual', 'alerta_desfasada', 
                   'bajo_voltaje', 'equipo_detenido', 'tendencia_baja', 'no_carga']
  
  # Etiquetamos como 'sin_clasificar' a los comentarios que no están asociados a algunos de los tipos de comentarios
  # previamente definidos.
  df_recom['sin_clasificar'] = df_recom[list_comments].sum(axis = 1)
  df_recom['sin_clasificar'] = df_recom['sin_clasificar'].apply(lambda x: 1 if x == 0 else 0)
  df_recom['comment_type'] = (df_recom[['comment_type', 'sin_clasificar']]
                              .apply(lambda x: x[0] + ['Sin clasificar'] if x[1] == 1 else x[0] + [''], axis = 1))
  
  # Por cada alerta rechazada, unimos todos los tipos de comentarios detectados, separados por el símbolo '|'.
  list_comments = list_comments + ['sin_clasificar']
  df_recom['comment_type'] = df_recom['comment_type'].apply(lambda x: '|'.join([text for text in x if text]))
  
  # Retornamos 'df_recom', junto con los tipos de comentarios ('list_comments').
  return df_recom, list_comments

def get_component_type(df_recom):
  df_recom['component_type'] = 'Otros'
  df_recom.loc[df_recom['component'].str.contains('Motor') == True, 'component_type'] = 'Motor'
  df_recom.loc[df_recom['component'].str.contains('Reductor') == True, 'component_type'] = 'Reductor'
  df_recom.loc[df_recom['component'].str.contains('Polea') == True, 'component_type'] = 'Polea'
  return df_recom

def remove_alerts(df_recom, remove_comments):
  df_recom = df_recom[~df_recom['comment_type'].str.contains(remove_comments)]
  df_recom = df_recom.sort_values('comment_type')
  return df_recom

def get_year_month(df_recom):
  df_recom['year_month'] = df_recom['fh'].dt.to_period('M').astype(str)
  return df_recom

def get_week(df_recom):
  df_recom['week'] = df_recom['fh'].dt.isocalendar().week.astype(str)
  return df_recom

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Funciones: Extracción de RMS

# COMMAND ----------

def get_rms_acel(i_indicador, fh_alert):
  date_range = 120
  date_compare = (fh_alert - dt.timedelta(days = date_range)).strftime('%Y-%m-%d')
  date_compare = dt.datetime.strptime(str(date_compare) + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

  today = dt.datetime.strptime(str(fh_alert), '%Y-%m-%d %H:%M:%S')
  upper_date = today.strftime('%Y-%m-%d')
  lower_date = (today - dt.timedelta(days = date_range)).strftime('%Y-%m-%d')

  rms_df = spark.sql(f"""
  select fh,
         rms_2_6_khz as RMS
  from delta.`/mnt/refined/drt/mantenimiento/siamflex/high_frequency_metric_calculation/OUT`
  where fecha >= '{lower_date}' and fecha <= '{upper_date}'
  and i_indicador = {i_indicador}
  and fh >= '{date_compare}' and fh <= '{today}'
  order by fh
  """).toPandas()
  
  return rms_df

def get_features(df_alerts):
  df_rms = pd.DataFrame()

  for _, row in df_alerts.iterrows():
    i_indicador = row['i_indicador']
    fh_alert = row['fh']
    rms_acel = get_rms_acel(i_indicador = i_indicador, fh_alert = fh_alert)
    rms_acel['i_indicador'] = i_indicador
    rms_acel['id_alert'] = fh_alert
    df_rms = df_rms.append(rms_acel, ignore_index = True)
    
  return df_rms

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones: Extra

# COMMAND ----------

def signif_bkps(m):
  threshold = 0.01
  signif_changepoints = m.changepoints[np.abs(np.nanmean(m.params['delta'], axis = 0)) >= threshold]
  return signif_changepoints

# COMMAND ----------

def slope(ult_fecha, primera_fecha, ven):
  
 #Días entre la generación de alerta y el último punto de cambio.
  x1 = parse(ult_fecha)
  x0 = parse(primera_fecha)

  de = x1 - x0
  
  dias= de.total_seconds()/ (60 * 60 * 24)

  #Diferencia entre el último valor de tendencia y el primero.
  y1=ven['trend'].iloc[len(ven['trend'])-1]
  y0=ven['trend'].iloc[0]
  tend= y1-y0

  #Pendiente
  pendiente= tend/dias
  return(pendiente)

# COMMAND ----------

def intercept(ult_fecha, primera_fecha, ven):
  
  x1 = (parse(ult_fecha).total_seconds())/ (60 * 60 * 24)

  y1=float(ven['trend'].iloc[len(ven['trend'])-1])
  
  pendiente= slope(ult_fecha, primera_fecha, ven)
  
  intercepto=-x1*pendiente + y1
  return(intercepto)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Alertas Generadas (Recommendations)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Carga de alertas

# COMMAND ----------

recom = get_alerts(env = 'prod')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Aplicación de filtros

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Filtramos por:**
# MAGIC 
# MAGIC - Variable **crestFactorTag** \\( = \\) rms_2_6_khz.
# MAGIC - Variable **status** \\( = \\) Approved, Declined.

# COMMAND ----------

df = recom[(recom['crestFactorTag'] == 'rms_2_6_khz') & (recom['status'].isin(['Approved', 'Declined']))]
df = df.sort_values('fh').reset_index(drop = True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Nuevas variables

# COMMAND ----------

df, comments = get_comment_type(df, temperature = False)
df = get_component_type(df)
df = get_year_month(df)
df = get_week(df)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Nuevas Metodologías

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Aplicación de filtros

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Filtramos por:**
# MAGIC 
# MAGIC - Variable **fh** \\( \geq \\) 2022-09-01 00:00:00.
# MAGIC - Variable **comment_type** \\( = \\) Alerta aceptada, Tendencia estable.

# COMMAND ----------

today = dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
date_compare = dt.datetime.strptime('2022-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')

columns = ['fh', 'year_month', 'week', 'status', 'comment', 'comment_type', 'username', 'crestFactorTag', 'description',
           'level_1', 'level_2', 'level_3', 'component', 'component_type', 'sensor', 'i_indicador']
df_alerts = df[columns]

df_alerts = df_alerts[(df_alerts['fh'] >= date_compare) & (df_alerts['fh'] <= today)]
df_alerts = df_alerts[df_alerts['comment_type'].isin(['Alerta aceptada', 'Tendencia estable'])]
df_alerts = df_alerts.reset_index(drop = True)

# COMMAND ----------

display(df_alerts)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Información por tipo de componente

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Análisis de motores

# COMMAND ----------

df_alerts_mot = (df_alerts[df_alerts['component_type'] == 'Motor']).reset_index(drop = True)
display(df_alerts_mot)

# COMMAND ----------

df_features_mot = get_features(df_alerts_mot)
display(df_features_mot)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Creamos la tabla **df_info_mot** que contiene la fecha de cada alerta emitida (**id_alert**), junto con el respectivo **i_indicador** notificado.

# COMMAND ----------

df_info_mot = df_features_mot[['i_indicador', 'id_alert']].drop_duplicates()
df_info_mot = df_info_mot.reset_index(drop = True)

# COMMAND ----------

display(df_info_mot)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Ejemplo de serie temporal:
# MAGIC - **i_indicador** = 25142 e **id_alert** = 2022-09-01T00:00:00.000+0000

# COMMAND ----------

df_alerts_mot.loc[df_alerts_mot["i_indicador"]==25142]

# COMMAND ----------

# MAGIC %md 
# MAGIC Esta alerta fue **rechazada por tendencia estable**

# COMMAND ----------

row = df_info_mot.loc[0, : ]
i_indicador = row[0]
id_alert = row[1]

condition_1 = df_features_mot['i_indicador'] == i_indicador
condition_2 = df_features_mot['id_alert'] == id_alert
df_example_mot = df_features_mot[condition_1 & condition_2]
df_example_mot= df_example_mot.drop(columns = ['i_indicador', 'id_alert'])

# COMMAND ----------

display(df_example_mot)

# COMMAND ----------

#Gráfica de la serie temporal para este ejemplo 
plot_serietemp(df_example_mot)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Isolation Forest

# COMMAND ----------

#Nótese que los parámetros del modelo están en default

# modelo_IF1 = IsolationForest(random_state = 0)
# modelo_IF1.fit(df_example_mot[['RMS']])

# COMMAND ----------

#Se calcula el score y anomaly value. Si score es cercano a uno, corresponde a una anomalía. Si es menor que 0.5 corresponde a un punto normal
# Si anomaly value es -1, es anomalía. Si es 1, es un punto normal

# df_example_mot['score'] = modelo_IF1.decision_function(df_example_mot[['RMS']])
# df_example_mot['anomaly_value'] = modelo_IF1.predict(df_example_mot[['RMS']])

# COMMAND ----------

#Se definen los valores que fueron clasificados como outliers

# outliers_mot = df_example_mot.loc[df_example_mot['anomaly_value'] == -1]
# outlier_index = list(outliers_mot.index)

#conteo de valores anómalos y normales. 
# df_example_mot['anomaly_value'].value_counts()

# COMMAND ----------

# plot_outliersIF(df_example_mot, outliers_mot)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Offline Change Point Detection 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Pelt

# COMMAND ----------

# MAGIC %md
# MAGIC Se utiliza el método de búsqueda **Pelt** (Linearly penalized segmentation) pues busca una solución óptima para el problema de segmentación con número de cambios (breakpoints) desconocido. Consideraremos la **función de costo L1**.

# COMMAND ----------

#Preparar input para el modelo. Debe recibir un numpy array.
df_example_mot.set_index(df_example_mot['fh'], inplace = True)
y_mot = np.array(df_example_mot['RMS'].tolist())

# COMMAND ----------

# Creamos una lista de posibles valores para el parámetro de penalización (en un rango de [0,20] recorriendo de 0.5) y el número de breakpoints que genera cada uno. 

breaks_mot = list()
pen_mot = list()
pen = 0
while pen <= 20:
  
  m_pelt= rpt.Pelt(model="l1").fit(y_mot)
  bkpts_pelt = m_
  pelt.predict(pen=pen)
  
  breaks = []
  for i in bkpts_pelt:
    breaks.append(df_example_mot['RMS'].index[i-1])
  breaks= pd.to_datetime(breaks)
  
  breaks_mot.append(len(breaks))
  pen_mot.append(pen)
  pen += 0.5

# COMMAND ----------

#Gráfica del "método del codo" para valor de penalización vs número de breakpoints
plt.plot(pen_mot, breaks_mot)
plt.xlabel("Valor de penalización",size=15)
plt.ylabel("Número de bkpts", size = 15)

# COMMAND ----------

# Visualizando tuplas de penalizaciones y sus respectivos números de breakpoints

tuplas = [(pen_mot[i], breaks_mot[i]) for i in range(0, len(breaks_mot))]

tuplas


# COMMAND ----------

#Segmentaciones para ciertos valores de penalizaciones

for pen in [0.5, 1.0, 2.0, 4.0, 4.5]:

  m_pelt= rpt.Pelt(model="l1").fit(y_mot)
  bkpts_pelt = m_pelt.predict(pen=pen)
  
  breaks = []
  for i in bkpts_pelt:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)
  
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para pen={pen}')
  print_legend = True
  for i in breaks:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Por lo visto, seleccionaremos pen=1.0 (1.5 entrega el mismo resultado de segmentaciones).

# COMMAND ----------

#Para pen=1.0

m_pelt= rpt.Pelt(model="l1").fit(y_mot)
pen=1.0
bkpts_pelt = m_pelt.predict(pen=pen)
  
breaks = []
for i in bkpts_pelt:
  breaks.append(df_example_mot["RMS"].index[i-1])
breaks= pd.to_datetime(breaks)

print("Los breakpoints son:", breaks)

plt.figure(figsize=(12,5.5))
plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
plt.title(f'Segmentaciones para pen={pen}')
print_legend = True
for i in breaks:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
plt.grid()
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### LR

# COMMAND ----------

ult_bloque= df_example_mot.loc['2022-08-20 15:16:11':'2022-08-31 07:16:47']
ult_bloque['Time'] = np.arange(len(ult_bloque.index))

# COMMAND ----------

x = ult_bloque.loc[:, ['Time']] 
y = ult_bloque.loc[:, 'RMS']  
x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Con la información otorgada por el modelo de regresión lineal, vemos que el **parámetro Time no es significativo con p-value=0.776**, es decir, la alerta no debió ser emitida. Lo anterior concuerda con la realidad pues corresponde a una alerta rechazada por el analista.

# COMMAND ----------

# MAGIC %md
# MAGIC * Análisis de gráficos:

# COMMAND ----------

 resid_plots(ult_bloque, model)

# COMMAND ----------

# MAGIC %md
# MAGIC De la gráfica, vemos que los residuos parecen seguir un patrón más bien cuadrático.

# COMMAND ----------

dist_resid(model)

# COMMAND ----------

# MAGIC %md
# MAGIC Los residuos no parecen seguir una distribución normal.

# COMMAND ----------

time_resid(ult_bloque, model)

# COMMAND ----------

# MAGIC %md
# MAGIC Se aprecia una dispersión aleatoria sin una tendencia notable.

# COMMAND ----------

# MAGIC %md
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model.resid, model.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Como el p-value del test es menor que 0.05, se rechaza la hipótesis nula de homocedasticidad y **se asume heterocedasticidad** de los residuos.

# COMMAND ----------

# MAGIC %md
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(model.resid))
print(kstest(model.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC El p-value de ambos tests es mayor que 0.05, luego se acepta la hipótesis nula, es decir, **la muestra viene de una distribución normal.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###### WR

# COMMAND ----------

# MAGIC %md
# MAGIC Para el cálculo de los pesos, generalmente se utilizan los inversos de la varianza de los residuos. Para estimar la varianza de los residuos, se puede hacer una regresión del valor absoluto de los residuos de la RL realizada anteriormente con los valores ajustados. Los valores ajustados de esta nueva regresión son un estimado de la desviación estandar; luego se pueden calcular los pesos tomando el inverso del cuadrado de los valores ajustados.

# COMMAND ----------

y_resid = [abs(resid) for resid in model.resid]
X_resid = sm.add_constant(model.fittedvalues)

mod_resid = sm.OLS(y_resid, X_resid)
res_resid = mod_resid.fit()

mod_fv = res_resid.fittedvalues

weights = 1 / (mod_fv**2)
weights

# COMMAND ----------

model_wr = sm.WLS(y, x, weights = weights)
res_wls = model_wr.fit()

print(res_wls.summary())

# COMMAND ----------

regression_plot(ult_bloque, model, model_wr)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Transformaciones

# COMMAND ----------

# MAGIC %md
# MAGIC * **Tranformación con logaritmo:**

# COMMAND ----------

ult_bloque['log_RMS']=np.log(ult_bloque['RMS'])

# COMMAND ----------

x = ult_bloque.loc[:, ['Time']] 
y = ult_bloque.loc[:,['log_RMS']]
x = sm.add_constant(x)

modellog = sm.OLS(y, x).fit()
predictionslog = modellog.predict(x) 

print_modellog = modellog.summary()
print(print_modellog)

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value=0.519, **el parámetro Time sigue sin ser significativo.** Por otro lado, vemos que mejora bastante el ajuste de los datos al modelo, con un R-cuadrado=0.025 mientras que anteriormente el ajuste tenía un R-cuadrado=0.005.
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modellog.resid, modellog.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Por lo tanto con p-value menor a 0.05 se rechaza la hipótesis nula de homocedasticidad, asumiendo **heterocedasticidad de los residuos.**
# MAGIC 
# MAGIC * **Test de Shapiro-Wilk y de Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(modellog.resid))
print(kstest(modellog.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor a 0.05 en ambos test, se acepta la hipótesis nula de **normalidad para los residuos.**

# COMMAND ----------

# MAGIC %md
# MAGIC * **Transformación de Box-Cox:**

# COMMAND ----------

fitted_data, fitted_lambda = stats.boxcox(ult_bloque['RMS'])

# COMMAND ----------

x = ult_bloque.loc[:, ['Time']] 
y = fitted_data  
x = sm.add_constant(x)

modelbc = sm.OLS(y, x).fit()
predictionsbc = modelbc.predict(x) 

print_modelbc = modelbc.summary()
print(print_modelbc)

# COMMAND ----------

# MAGIC %md
# MAGIC Nuevamente notamos que el **parámetro Time no es significativo.** Además, vemos que este es el peor R-cuadrado obtenido.
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modelbc.resid, modelbc.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Otra vez, con p-value menor a 0.05 se rechaza la hipótesis nula de homocedasticidad, asumiendo **heterocedasticidad sobre los residuos.**
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(modelbc.resid))
print(kstest(modelbc.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor a 0.05 para ambos test, se asume **la normalidad de los residuos.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Diff
# MAGIC Pelt

# COMMAND ----------

y_mot_diff = np.array(df_example_mot['RMS'])
y_mot_diff= np.diff(y_mot_diff)

breaks_motd = list()
pen_motd = list()
pen = 0
while pen <= 1:
  
  m_pelt_d= rpt.Pelt(model="l1").fit(y_mot_diff)
  bkpts_pelt_d = m_pelt_d.predict(pen=pen)
  
  breaks_d = []
  for i in bkpts_pelt_d:
    breaks_d.append(df_example_mot['RMS'].index[i-1])
  breaks_d= pd.to_datetime(breaks_d)
  
  breaks_motd.append(len(breaks_d))
  pen_motd.append(pen)
  pen += 0.05
  
plt.plot(pen_motd, breaks_motd)
plt.xlabel("Valor de penalización",size=15)
plt.ylabel("Número de bkpts", size = 15)

# COMMAND ----------

tuplas = [(pen_motd[i], breaks_motd[i]) for i in range(0, len(breaks_motd))]

tuplas

# COMMAND ----------

for pen in [0.4, 0.5, 0.7]:

  m_pelt= rpt.Pelt(model="l1").fit(y_mot_diff)
  bkpts_pelt = m_pelt.predict(pen=pen)
  
  breaks = []
  for i in bkpts_pelt:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)
  
  print(f'Los breakpoints son= {breaks}')
  
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para pen={pen}')
  print_legend = True
  for i in breaks:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

#Para pen=0.5
ventana= df_example_mot.loc['2022-08-27 08:46:14':'2022-08-31 03:46:20']
ventana['Time'] = np.arange(len(ventana.index))

# COMMAND ----------

xdif =ventana.loc[:, ['Time']] 
ydif =ventana.loc[:, 'RMS']  
xdif = sm.add_constant(xdif)

modeldif = sm.OLS(ydif, xdif).fit()
predictionsdif = modeldif.predict(xdif) 

print_modeldif = modeldif.summary()
print(print_modeldif)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modeldif.resid, modeldif.model.exog)
lzip(name, test)

# COMMAND ----------


print(shapiro(modeldif.resid))
print(kstest(modeldif.resid, 'norm'))

# COMMAND ----------

y_residdif = [abs(resid) for resid in modeldif.resid]
X_residdif = sm.add_constant(modeldif.fittedvalues)

mod_residdif = sm.OLS(y_residdif, X_residdif)
res_residdif = mod_residdif.fit()

mod_fvdif = res_residdif.fittedvalues

weightsdif = 1 / (mod_fvdif**2)

# COMMAND ----------

modeldif_wr = sm.WLS(ydif, xdif, weights = weightsdif)
res_wlsdif = modeldif_wr.fit()

print(res_wlsdif.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Dynp

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora utilizaremos el método de búsqueda **Dynp** (Dynamic Programming), el cual encuentra puntos de cambios óptimos con programación dinámica. Dado un modelo de segmentación, calcula la mejor partición para la cual la suma de errores es mínima. Requiere el número de breakpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC ###### L1

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 20]:  
  m_dynp= rpt.Dynp(model="l1").fit(y_mot)
  bkpts_dynp=m_dynp.predict(n_bkps=n_bkps)
  
  breaks = []
  for i in bkpts_dynp:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)
  
  print("Los breakpoints son:", breaks)
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### LR
# MAGIC Considerando el último bloque obtenido con n_bkps=2.

# COMMAND ----------

ult_bloque_1=df_example_mot.loc['2022-06-26 17:16:09':'2022-08-31 07:16:47']
ult_bloque_1['Time'] = np.arange(len(ult_bloque_1.index))

# COMMAND ----------

x_dynp = ult_bloque_1.loc[:, ['Time']] 
y_dynp = ult_bloque_1.loc[:, 'RMS']  
x_dynp = sm.add_constant(x_dynp)

model_dynp = sm.OLS(y_dynp, x_dynp).fit()
predictions_dynp = model_dynp.predict(x_dynp) 

print_model_dynp = model_dynp.summary()
print(print_model_dynp)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que el **parámetro Time sí es significativo.** Además R-cuadrado=0.093.

# COMMAND ----------

# MAGIC %md
# MAGIC * Análisis gráfico:

# COMMAND ----------

 resid_plots(ult_bloque_1, model_dynp)

# COMMAND ----------

# MAGIC %md 
# MAGIC Se aprecia que los residuos siguen un patrón casi lineal.

# COMMAND ----------

dist_resid(model_dynp)

# COMMAND ----------

# MAGIC %md
# MAGIC Por las gráficas, se ve que los residuos tienen una distribución cercana a la normal.

# COMMAND ----------

time_resid(ult_bloque_1, model_dynp)

# COMMAND ----------

# MAGIC %md 
# MAGIC Se ve una dispersión aleatoria, de tendencia nula.

# COMMAND ----------

# MAGIC %md 
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp.resid, model_dynp.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor que 0.05, la hipótesis nula se acepta y **se asume homocedasticidad en los residuos.**

# COMMAND ----------

# MAGIC %md
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(model_dynp.resid))
print(kstest(model_dynp.resid, 'norm'))

# COMMAND ----------

# MAGIC %md 
# MAGIC Para Shapiro-Wilk, con p-value mayor que 0.05, se acepta la hipótesis nula y se asume que la muestra viene de una distribución normal. Mientras que para Kolmogorov-Smirnov, se rechaza la hipótesis nula y **no se puede asumir que la muestra viene de una distribución normal.** Notemos que en este caso, con una ventana temporal de 162 registros, se vuelve más adecuado Kolmogorov-Smirnov, pues Shapiro-Wilk no se recomienda para muestras con más de 50 valores.

# COMMAND ----------

# MAGIC %md
# MAGIC ###### WR

# COMMAND ----------

y_resid_dynp = [abs(resid) for resid in model_dynp.resid]
X_resid_dynp = sm.add_constant(model_dynp.fittedvalues)

mod_resid_dynp = sm.OLS(y_resid_dynp, X_resid_dynp)
res_resid_dynp = mod_resid_dynp.fit()

mod_fv_dynp = res_resid_dynp.fittedvalues

weights_dynp = 1 / (mod_fv_dynp**2)

# COMMAND ----------

model_dynp_wr = sm.WLS(y_dynp, x_dynp, weights = weights_dynp)
res_wls_dynp = model_dynp_wr.fit()

print(res_wls_dynp.summary())

# COMMAND ----------

regression_plot(ult_bloque_1, model_dynp, model_dynp_wr)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Transformaciones:

# COMMAND ----------

# MAGIC %md
# MAGIC * **Logaritmo:**

# COMMAND ----------

ult_bloque_1['log_RMS']=np.log(ult_bloque_1['RMS'])

# COMMAND ----------

x_dynp = ult_bloque_1.loc[:, ['Time']] 
y_dynp = ult_bloque_1.loc[:, 'log_RMS']  
x_dynp = sm.add_constant(x_dynp)

modellog_dynp = sm.OLS(y_dynp, x_dynp).fit()
predictionslog_dynp = modellog_dynp.predict(x_dynp) 

print_modellog_dynp = modellog_dynp.summary()
print(print_modellog_dynp)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que para este modelo, el **parámetro Time no es significativo** (p-value mayor que 0.05). Además, el R-cuadrado disminuye notoriamente.
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modellog_dynp.resid, modellog_dynp.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md 
# MAGIC Con p-value mayor que 0.05, se acepta la hipótesis nula de **homocedasticidad de los residuos.**
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(modellog_dynp.resid))
print(kstest(modellog_dynp.resid, 'norm'))

# COMMAND ----------

# MAGIC %md 
# MAGIC Con p-values menores que 0.05, **se rechaza la hipótesis nula de normalidad.**

# COMMAND ----------

# MAGIC %md
# MAGIC * **Transformación de Box-Cox:**

# COMMAND ----------

fitted_data_1, fitted_lambda_1 = stats.boxcox(ult_bloque_1['RMS'])


# COMMAND ----------

x_dynp = ult_bloque_1.loc[:, ['Time']] 
y_dynp = fitted_data_1 
x_dynp = sm.add_constant(x_dynp)

modelbc_dynp = sm.OLS(y_dynp, x_dynp).fit()
predictionsbc_dynp = modelbc_dynp.predict(x_dynp) 

print_modelbc_dynp = modelbc_dynp.summary()
print(print_modelbc_dynp)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que para este modelo, el **parámetro Time sí es significativo.** Además, posee un R-cuadrado mucho mejor que la transformación de logaritmo.
# MAGIC * **Test de Breusch-Pagan**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modelbc_dynp.resid, modelbc_dynp.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor que 0.05 se asume **homocedasticidad sobre los residuos.**
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(modelbc_dynp.resid))
print(kstest(modelbc_dynp.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-values menores a 0.05 **se rechaza la hipótesis nula de normalidad.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Lineal

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 12, 15,18, 20]:  
  m_dynp_l= rpt.Dynp(model="linear").fit(y_mot.reshape(-1, 1))
  bkpts_dynp_l=m_dynp_l.predict(n_bkps=n_bkps)

  breaks = []
  for i in bkpts_dynp_l:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)

  print("Los breakpoints son:", breaks)

  plt.figure(figsize=(24,4))
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
      if print_legend:
          plt.axvline(i, color='red',linestyle='dashed', label='breaks')
          print_legend = False
      else:
          plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### LR
# MAGIC Ajuste de un modelo de regresión lineal para la última segmentación obtenida con n_bkps=12.

# COMMAND ----------

ult_bloque_2= df_example_mot.loc['2022-06-12 20:16:30':'2022-08-31 07:16:47']
ult_bloque_2['Time'] = np.arange(len(ult_bloque_2.index))

# COMMAND ----------

x_dynp_l = ult_bloque_2.loc[:, ['Time']] 
y_dynp_l = ult_bloque_2.loc[:, 'RMS']  
x_dynp_l = sm.add_constant(x_dynp_l)

model_dynp_l = sm.OLS(y_dynp_l, x_dynp_l).fit()
predictions_dynp_l = model_dynp_l.predict(x_dynp_l) 

print_model_dynp_l = model_dynp_l.summary()
print(print_model_dynp_l)

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value=0.151, el **parámetro Time no es significativo.** Notar además que el R-cuadrado es de 0.010.

# COMMAND ----------

# MAGIC %md
# MAGIC * **Análisis gráficos:**

# COMMAND ----------

resid_plots(ult_bloque_2,model_dynp_l)

# COMMAND ----------

# MAGIC %md
# MAGIC Se aprecia que los residuos siguen un patrón más bien cuadrático.

# COMMAND ----------

dist_resid(model_dynp_l)

# COMMAND ----------

# MAGIC %md
# MAGIC La distribución de los residuos no parece ser tan similar a la normal.

# COMMAND ----------

time_resid(ult_bloque_2,model_dynp_l)

# COMMAND ----------

# MAGIC %md
# MAGIC Se ve una dispersión aleatoria.

# COMMAND ----------

# MAGIC %md
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp_l.resid, model_dynp_l.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor que 0.05, la hipótesis nula se acepta y se asume **homocedasticidad en los residuos.**

# COMMAND ----------

# MAGIC %md
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov :**

# COMMAND ----------

print(shapiro(model_dynp_l.resid))
print(kstest(model_dynp_l.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC El p-value de los tests es menor que 0.05, luego se rechaza la hipótesis nula, es decir, **la muestra no viene de una distribución normal.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###### WR

# COMMAND ----------

y_resid_dynp_l = [abs(resid) for resid in model_dynp_l.resid]
X_resid_dynp_l = sm.add_constant(model_dynp_l.fittedvalues)

mod_resid_dynp_l = sm.OLS(y_resid_dynp_l, X_resid_dynp_l)
res_resid_dynp_l = mod_resid_dynp_l.fit()

mod_fv_dynp_l = res_resid_dynp_l.fittedvalues

weights_dynp_l = 1 / (mod_fv_dynp_l**2)

# COMMAND ----------

model_dynp_l_wr = sm.WLS(y_dynp_l, x_dynp_l, weights = weights_dynp_l)
res_wls_dynp_l = model_dynp_l_wr.fit()

print(res_wls_dynp_l.summary())

# COMMAND ----------

regression_plot(ult_bloque_2, model_dynp_l, model_dynp_l_wr)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Transformaciones:

# COMMAND ----------

# MAGIC %md
# MAGIC * **Logaritmo:**

# COMMAND ----------

ult_bloque_2['log_RMS']=np.log(ult_bloque_2['RMS'])

# COMMAND ----------

x_dynp_l = ult_bloque_2.loc[:, ['Time']] 
y_dynp_l = ult_bloque_2.loc[:, 'log_RMS']  
x_dynp_l = sm.add_constant(x_dynp_l)

modellog_dynp_l = sm.OLS(y_dynp_l, x_dynp_l).fit()
predictionslog_dynp_l = modellog_dynp_l.predict(x_dynp_l) 

print_modellog_dynp_l = modellog_dynp_l.summary()
print(print_modellog_dynp_l)

# COMMAND ----------

# MAGIC %md
# MAGIC Notamos que para este modelo el **parámetro Time no es significativo**. Además, el ajuste presenta un R-cuadrado menor que en el modelo original.
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modellog_dynp_l.resid, modellog_dynp_l.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor a 0.05, se asume la **homocedasticidad de los residuos.**
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(modellog_dynp_l.resid))
print(kstest(modellog_dynp_l.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que **se rechaza la hipótesis nula de normalidad.**

# COMMAND ----------

# MAGIC %md
# MAGIC * **Box-Cox:**

# COMMAND ----------

fitted_data_2, fitted_lambda = stats.boxcox(ult_bloque_2['RMS'])

# COMMAND ----------

x_dynp_l = ult_bloque_2.loc[:, ['Time']] 
y_dynp_l = fitted_data_2
x_dynp_l = sm.add_constant(x_dynp_l)

modelbc_dynp_l = sm.OLS(y_dynp_l, x_dynp_l).fit()
predictionsbc_dynp_l = modelbc_dynp_l.predict(x_dynp_l) 

print_modelbc_dynp_l = modelbc_dynp_l.summary()
print(print_modelbc_dynp_l)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que en este modelo el **parámetro Time no es significativo.** El R-cuadrado es apenas menor que el del modelo original.
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modelbc_dynp_l.resid, modelbc_dynp_l.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Se acepta la hipótesis de **homocedasticidad** para los residuos.
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(modelbc_dynp_l.resid))
print(kstest(modelbc_dynp_l.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC **Se rechaza la hipótesis nula de normalidad.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###### CLineal

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 12, 15,18, 20]:  
  m_dynp_l= rpt.Dynp(model="clinear").fit(y_mot)
  bkpts_dynp_l=m_dynp_l.predict(n_bkps=n_bkps)

  breaks = []
  for i in bkpts_dynp_l:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)

  print("Los breakpoints son:", breaks)

  plt.figure(figsize=(24,4))
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
      if print_legend:
          plt.axvline(i, color='red',linestyle='dashed', label='breaks')
          print_legend = False
      else:
          plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### LR
# MAGIC Ajuste de un modelo de regresión lineal para el último bloque generado con n_bkps=2.

# COMMAND ----------

ult_bloque_3= df_example_mot.loc[ '2022-08-10 22:46:11':'2022-08-31 07:16:47']
ult_bloque_3['Time'] = np.arange(len(ult_bloque_3.index))

# COMMAND ----------

x_dynp_cl = ult_bloque_3.loc[:, ['Time']] 
y_dynp_cl = ult_bloque_3.loc[:, 'RMS']  
x_dynp_cl = sm.add_constant(x_dynp_cl)

model_dynp_cl = sm.OLS(y_dynp_cl, x_dynp_cl).fit()
predictions_dynp_cl = model_dynp_cl.predict(x_dynp_cl) 

print_model_dynp_cl = model_dynp_cl.summary()
print(print_model_dynp_cl)

# COMMAND ----------

# MAGIC %md 
# MAGIC Para este caso, el parámetro **Time sí es significativo.** Además, se aprecia un R-cuadrado mucho mayor a los vistos anteriormente.
# MAGIC * **Análisis Gráficos:**

# COMMAND ----------

resid_plots(ult_bloque_3, model_dynp_cl)

# COMMAND ----------

dist_resid(model_dynp_cl)

# COMMAND ----------

time_resid(ult_bloque_3,model_dynp_cl)

# COMMAND ----------

# MAGIC %md
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp_cl.resid, model_dynp_cl.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor a 0.05, se asume **homocedasticidad en los residuos.**
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(model_dynp_cl.resid))
print(kstest(model_dynp_cl.resid, 'norm'))

# COMMAND ----------

# MAGIC %md 
# MAGIC Notemos que para el test de Shapiro Wilk, se acepta la hipótesis nula de normalidad. En cambio, para Kolmogorv-Smirnov esta hipótesis se rechaza. En este caso, examinando 42 residuos, Shapiro-Wilk no pierde validez por tamaño de muestra. Y de hecho, es el test más preciso dentro de este rango de tamaño de muestras, por lo cual **se asume normalidad.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###### WR

# COMMAND ----------

y_resid_dynp_cl = [abs(resid) for resid in model_dynp_cl.resid]
X_resid_dynp_cl = sm.add_constant(model_dynp_cl.fittedvalues)

mod_resid_dynp_cl = sm.OLS(y_resid_dynp_cl, X_resid_dynp_cl)
res_resid_dynp_cl = mod_resid_dynp_cl.fit()

mod_fv_dynp_cl = res_resid_dynp_cl.fittedvalues

weights_dynp_cl = 1 / (mod_fv_dynp_cl**2)

# COMMAND ----------

model_dynp_cl_wr = sm.WLS(y_dynp_cl, x_dynp_cl, weights = weights_dynp_cl)
res_wls_dynp_cl = model_dynp_cl_wr.fit()

print(res_wls_dynp_cl.summary())

# COMMAND ----------

regression_plot(ult_bloque_3, model_dynp_cl, model_dynp_cl_wr)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Transformaciones

# COMMAND ----------

ult_bloque_3['log_RMS']=np.log(ult_bloque_3['RMS'])

# COMMAND ----------

x_dynp_cl = ult_bloque_3.loc[:, ['Time']] 
y_dynp_cl = ult_bloque_3.loc[:, 'log_RMS']  
x_dynp_cl = sm.add_constant(x_dynp_cl)

model_dynp_cl = sm.OLS(y_dynp_cl, x_dynp_cl).fit()
predictions_dynp_cl = model_dynp_cl.predict(x_dynp_cl) 

print_model_dynp_cl = model_dynp_cl.summary()
print(print_model_dynp_cl)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp_cl.resid, model_dynp_cl.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(model_dynp_cl.resid))
print(kstest(model_dynp_cl.resid, 'norm'))

# COMMAND ----------

fitted_data_3, fitted_lambda = stats.boxcox(ult_bloque_3['RMS'])

# COMMAND ----------

x_dynp_cl = ult_bloque_3.loc[:, ['Time']] 
y_dynp_cl = fitted_data_3  
x_dynp_cl = sm.add_constant(x_dynp_cl)

model_dynp_cl = sm.OLS(y_dynp_cl, x_dynp_cl).fit()
predictions_dynp_cl = model_dynp_cl.predict(x_dynp_cl) 

print_model_dynp_cl = model_dynp_cl.summary()
print(print_model_dynp_cl)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp_cl.resid, model_dynp_cl.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(model_dynp_cl.resid))
print(kstest(model_dynp_cl.resid, 'norm'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### AR

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 12, 15,18, 20]:  
  m_dynp_l= rpt.Dynp(model="ar").fit(y_mot)
  bkpts_dynp_l=m_dynp_l.predict(n_bkps=n_bkps)

  breaks = []
  for i in bkpts_dynp_l:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)

  print("Los breakpoints son:", breaks)

  plt.figure(figsize=(24,4))
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
      if print_legend:
          plt.axvline(i, color='red',linestyle='dashed', label='breaks')
          print_legend = False
      else:
          plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### LR
# MAGIC  Ajuste de un modelo de regresión lineal para el último bloque temporal generado con n_bkps=2

# COMMAND ----------

ult_bloque_4= df_example_mot.loc['2022-08-20 15:16:11':'2022-08-31 07:16:47']
ult_bloque_4['Time'] = np.arange(len(ult_bloque_4.index))

# COMMAND ----------

x_dynp_ar = ult_bloque_4.loc[:, ['Time']] 
y_dynp_ar = ult_bloque_4.loc[:, 'RMS']  
x_dynp_ar = sm.add_constant(x_dynp_ar)

model_dynp_ar = sm.OLS(y_dynp_ar, x_dynp_ar).fit()
predictions_dynp_ar = model_dynp_ar.predict(x_dynp_cl) 

print_model_dynp_ar = model_dynp_ar.summary()
print(print_model_dynp_ar)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que el parámetro **Time no es significativo.** Además, el R-cuadrado es bastante bajo.
# MAGIC * **Análisis Gráficos:**

# COMMAND ----------

resid_plots(ult_bloque_4, model_dynp_ar)

# COMMAND ----------

dist_resid(model_dynp_ar)

# COMMAND ----------

time_resid(ult_bloque_4, model_dynp_ar)

# COMMAND ----------

# MAGIC %md
# MAGIC * **Test de Breusch-Pagan:**

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp_ar.resid, model_dynp_ar.model.exog)
lzip(name, test)

# COMMAND ----------

# MAGIC %md
# MAGIC **Se rechaza la hipótesis nula de homocedasticidad de los residuos.**
# MAGIC * **Test de Shapiro-Wilk y Kolmogorov-Smirnov:**

# COMMAND ----------

print(shapiro(model_dynp_ar.resid))
print(kstest(model_dynp_ar.resid, 'norm'))

# COMMAND ----------

# MAGIC %md
# MAGIC Con p-value mayor a 0.05 en ambos test, **se acepta la hipótesis nula de normalidad.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###### WR

# COMMAND ----------

y_resid_dynp_ar = [abs(resid) for resid in model_dynp_ar.resid]
X_resid_dynp_ar = sm.add_constant(model_dynp_ar.fittedvalues)

mod_resid_dynp_ar = sm.OLS(y_resid_dynp_ar, X_resid_dynp_ar)
res_resid_dynp_ar = mod_resid_dynp_ar.fit()

mod_fv_dynp_ar = res_resid_dynp_ar.fittedvalues

weights_dynp_ar = 1 / (mod_fv_dynp_ar**2)

# COMMAND ----------

model_dynp_ar_wr = sm.WLS(y_dynp_ar, x_dynp_ar, weights = weights_dynp_ar)
res_wls_dynp_ar = model_dynp_ar_wr.fit()

print(res_wls_dynp_ar.summary())

# COMMAND ----------

regression_plot(ult_bloque_4, model_dynp_ar, model_dynp_ar_wr)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### transforamciones

# COMMAND ----------

ult_bloque_4['log_RMS']=np.log(ult_bloque_4['RMS'])

# COMMAND ----------

x_dynp_ar = ult_bloque_4.loc[:, ['Time']] 
y_dynp_ar = ult_bloque_4.loc[:, 'log_RMS']  
x_dynp_ar = sm.add_constant(x_dynp_ar)

model_dynp_ar = sm.OLS(y_dynp_ar, x_dynp_ar).fit()
predictions_dynp_ar = model_dynp_ar.predict(x_dynp_cl) 

print_model_dynp_ar = model_dynp_ar.summary()
print(print_model_dynp_ar)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp_ar.resid, model_dynp_ar.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(model_dynp_ar.resid))
print(kstest(model_dynp_ar.resid, 'norm'))

# COMMAND ----------

fitted_data_4, fitted_lambda = stats.boxcox(ult_bloque_4['RMS'])

# COMMAND ----------

x_dynp_ar = ult_bloque_4.loc[:, ['Time']] 
y_dynp_ar = fitted_data_4
x_dynp_ar = sm.add_constant(x_dynp_ar)

model_dynp_ar = sm.OLS(y_dynp_ar, x_dynp_ar).fit()
predictions_dynp_ar = model_dynp_ar.predict(x_dynp_cl) 

print_model_dynp_ar = model_dynp_ar.summary()
print(print_model_dynp_ar)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(model_dynp_ar.resid, model_dynp_ar.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(model_dynp_ar.resid))
print(kstest(model_dynp_ar.resid, 'norm'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Diff
# MAGIC * Dynp (L1):

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 20]:  
  m_dynp= rpt.Dynp(model="l1").fit(y_mot_diff)
  bkpts_dynp=m_dynp.predict(n_bkps=n_bkps)
  
  breaks = []
  for i in bkpts_dynp:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)
  
  print("Los breakpoints son:", breaks)
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

#Considerando n_bkps=2 (el anterior da una ventana que ya se tiene)
ventana_1= df_example_mot.loc['2022-07-19 21:46:13':'2022-08-31 03:46:20']
ventana_1['Time'] = np.arange(len(ventana_1.index))

# COMMAND ----------

xdif =ventana_1.loc[:, ['Time']] 
ydif =ventana_1.loc[:, 'RMS']  
xdif = sm.add_constant(xdif)

modeldif_dynp = sm.OLS(ydif, xdif).fit()
predictionsdif_dynp = modeldif_dynp.predict(xdif) 

print_modeldif_dynp = modeldif_dynp.summary()
print(print_modeldif_dynp)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modeldif_dynp.resid, modeldif_dynp.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(modeldif_dynp.resid))
print(kstest(modeldif_dynp.resid, 'norm'))

# COMMAND ----------

y_residdif = [abs(resid) for resid in modeldif_dynp.resid]
X_residdif = sm.add_constant(modeldif_dynp.fittedvalues)

mod_residdif = sm.OLS(y_residdif, X_residdif)
res_residdif = mod_residdif.fit()

mod_fvdif = res_residdif.fittedvalues

weightsdif = 1 / (mod_fvdif**2)


# COMMAND ----------

modeldif_dynp_wr = sm.WLS(ydif, xdif, weights = weightsdif)
res_wlsdif = modeldif_dynp_wr.fit()

print(res_wlsdif.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC * Lineal:

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 15, 20]:  
  m_dynp_l= rpt.Dynp(model="linear").fit(y_mot_diff.reshape(-1, 1))
  bkpts_dynp_l=m_dynp_l.predict(n_bkps=n_bkps)

  breaks = []
  for i in bkpts_dynp_l:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)

  print("Los breakpoints son:", breaks)

  plt.figure(figsize=(24,4))
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
      if print_legend:
          plt.axvline(i, color='red',linestyle='dashed', label='breaks')
          print_legend = False
      else:
          plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

#Considerando n_bkps=2
ventana_2= df_example_mot.loc['2022-08-10 22:46:11':'2022-08-31 03:46:20']
ventana_2['Time'] = np.arange(len(ventana_2.index))

xdif =ventana_2.loc[:, ['Time']] 
ydif =ventana_2.loc[:, 'RMS']  
xdif = sm.add_constant(xdif)

modeldif_dynp_l= sm.OLS(ydif, xdif).fit()
predictionsdif_dynp_l = modeldif_dynp_l.predict(xdif) 

print_modeldif_dynp_l = modeldif_dynp_l.summary()
print(print_modeldif_dynp_l)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modeldif_dynp_l.resid, modeldif_dynp_l.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(modeldif_dynp_l.resid))
print(kstest(modeldif_dynp_l.resid, 'norm'))

# COMMAND ----------

y_residdif = [abs(resid) for resid in modeldif_dynp_l.resid]
X_residdif = sm.add_constant(modeldif_dynp_l.fittedvalues)

mod_residdif = sm.OLS(y_residdif, X_residdif)
res_residdif = mod_residdif.fit()

mod_fvdif = res_residdif.fittedvalues

weightsdif = 1 / (mod_fvdif**2)

modeldif_dynp_l_wr = sm.WLS(ydif, xdif, weights = weightsdif)
res_wlsdif = modeldif_dynp_l_wr.fit()

print(res_wlsdif.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC * Clineal:

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 20]:  
  m_dynp= rpt.Dynp(model="clinear").fit(y_mot_diff)
  bkpts_dynp=m_dynp.predict(n_bkps=n_bkps)
  
  breaks = []
  for i in bkpts_dynp:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)
  
  print("Los breakpoints son:", breaks)
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

#Considerando n_bkps=1.
ventana_3= df_example_mot.loc['2022-08-19 04:17:15':'2022-08-31 03:46:20']
ventana_3['Time'] = np.arange(len(ventana_3.index))

xdif =ventana_3.loc[:, ['Time']] 
ydif =ventana_3.loc[:, 'RMS']  
xdif = sm.add_constant(xdif)

modeldif_dynp_cl = sm.OLS(ydif, xdif).fit()
predictionsdif_dynp_cl = modeldif_dynp_cl.predict(xdif) 

print_modeldif_dynp_cl = modeldif_dynp_cl.summary()
print(print_modeldif_dynp_cl)


# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modeldif_dynp_cl.resid, modeldif_dynp_cl.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(modeldif_dynp_cl.resid))
print(kstest(modeldif_dynp_cl.resid, 'norm'))

# COMMAND ----------

y_residdif = [abs(resid) for resid in modeldif_dynp_cl.resid]
X_residdif = sm.add_constant(modeldif_dynp_cl.fittedvalues)

mod_residdif = sm.OLS(y_residdif, X_residdif)
res_residdif = mod_residdif.fit()

mod_fvdif = res_residdif.fittedvalues

weightsdif = 1 / (mod_fvdif**2)

modeldif_dynp_cl_wr = sm.WLS(ydif, xdif, weights = weightsdif)
res_wlsdif = modeldif_dynp_cl_wr.fit()

print(res_wlsdif.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC * AR:

# COMMAND ----------

for n_bkps in [1, 2, 5, 10, 20]:  
  m_dynp= rpt.Dynp(model="ar").fit(y_mot_diff)
  bkpts_dynp=m_dynp.predict(n_bkps=n_bkps)
  
  breaks = []
  for i in bkpts_dynp:
    breaks.append(df_example_mot["RMS"].index[i-1])
  breaks= pd.to_datetime(breaks)
  
  print("Los breakpoints son:", breaks)
  plt.plot(df_example_mot["fh"], df_example_mot["RMS"])
  plt.title(f'Segmentaciones para n_bkps={n_bkps}')
  print_legend = True
  for i in breaks:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
  plt.grid()
  plt.legend()
  plt.show()

# COMMAND ----------

ventana_4= df_example_mot.loc['2022-06-24 18:16:19':'2022-08-31 03:46:20']
ventana_4['Time'] = np.arange(len(ventana_4.index))

xdif =ventana_4.loc[:, ['Time']] 
ydif =ventana_4.loc[:, 'RMS']  
xdif = sm.add_constant(xdif)

modeldif_dynp_ar = sm.OLS(ydif, xdif).fit()
predictionsdif_dynp_ar = modeldif_dynp_ar.predict(xdif) 

print_modeldif_dynp_ar = modeldif_dynp_ar.summary()
print(print_modeldif_dynp_ar)

# COMMAND ----------

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(modeldif_dynp_ar.resid, modeldif_dynp_ar.model.exog)
lzip(name, test)

# COMMAND ----------

print(shapiro(modeldif_dynp_ar.resid))
print(kstest(modeldif_dynp_ar.resid, 'norm'))

# COMMAND ----------

y_residdif = [abs(resid) for resid in modeldif_dynp_ar.resid]
X_residdif = sm.add_constant(modeldif_dynp_ar.fittedvalues)

mod_residdif = sm.OLS(y_residdif, X_residdif)
res_residdif = mod_residdif.fit()

mod_fvdif = res_residdif.fittedvalues

weightsdif = 1 / (mod_fvdif**2)

modeldif_dynp_cl_wr = sm.WLS(ydif, xdif, weights = weightsdif)
res_wlsdif = modeldif_dynp_cl_wr.fit()

print(res_wlsdif.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prophet

# COMMAND ----------

#Preparar los datos en el formato que lee Prophet
ts = df_features_mot[condition_1 & condition_2].drop(columns = ['i_indicador', 'id_alert'])
ts.rename(columns={'fh':'ds', 'RMS':'y'}, inplace=True)

# COMMAND ----------

plt.plot(ts['ds'], ts['y']);

# COMMAND ----------

# MAGIC %md
# MAGIC * **Primer modelo:** Modificamos changepoint_range para que se consideren changepoints a lo largo de toda la historia y no solo del primer 80%. También changepoint_prior_scale, en el valor arbitrario de 0.5 para que el modelo sea más flexible y detecte puntos de cambio.

# COMMAND ----------

m = Prophet(changepoint_range=1, changepoint_prior_scale=0.5) 
m.fit(ts) # df: Pandas dataframe con columnas "ds" (Datetime) e "y" (float).


in_sample_forecast = m.predict(ts)
fig = m.plot(in_sample_forecast)
a = add_changepoints_to_plot(fig.gca(), m, in_sample_forecast)

# COMMAND ----------

#Ver puntos de cambio.
signif_bkps(m)

# COMMAND ----------

#Definir el último bloque (entre último punto de cambio y fecha que se emitió la alerta)
ult_ven = in_sample_forecast.loc[(in_sample_forecast['ds'] >= '2022-08-13 14:16:14') & (in_sample_forecast['ds'] <= '2022-08-31 07:16:47')]
display(ult_ven)

# COMMAND ----------

#Recta del último bloque a estudiar
plt.plot(ult_ven['ds'], ult_ven['trend'])

# COMMAND ----------

slope('2022-08-31 07:16:47', '2022-08-13 14:16:14', ult_ven)

# COMMAND ----------

#Con polyfit:
temp_dates = ult_ven['ds'].apply(lambda x: x.timestamp())
ult_ven['dt_timestamp'] = temp_dates / 86400
m, i = np.polyfit(ult_ven['dt_timestamp'], ult_ven['trend'], 1)
print('Pendiente=', m, ",","Intercepto=", i)
plt.plot(ult_ven['dt_timestamp'], ult_ven['trend']);


# COMMAND ----------

# best_params = all_params[np.argmin(rmses)]
# print(best_params)

# COMMAND ----------

m1 = Prophet(changepoint_range=1, changepoint_prior_scale=0.8)
m1.fit(ts)

in_sample_forecast = m1.predict(ts)
fig = m1.plot(in_sample_forecast)
a = add_changepoints_to_plot(fig.gca(),m1,in_sample_forecast)

# COMMAND ----------

final_ds=ts['ds'].dt.date.max()
final_bkps= signif_bkps(m1).dt.date.max()
ult_ven1=  in_sample_forecast.loc[(in_sample_forecast['ds'].dt.date >= final_bkps) & (in_sample_forecast['ds'].dt.date <= final_ds)]

# COMMAND ----------

slope('2022-08-31 07:16:47','2022-08-19 14:16:07', ult_ven1)

# COMMAND ----------

temp_dates = ult_ven1['ds'].apply(lambda x: x.timestamp())
ult_ven1['dt_timestamp'] = temp_dates / 86400
m, i = np.polyfit(ult_ven1['dt_timestamp'], ult_ven1['trend'], 1)
print('Pendiente=', m, ",","Intercepto=", i)
plt.plot(ult_ven1['dt_timestamp'], ult_ven1['trend']);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Alertas Rechazadas

# COMMAND ----------

len(df_alerts_mot.loc[df_alerts_mot['status']=='Declined'])

# COMMAND ----------

# MAGIC %md
# MAGIC Como vemos, existen **16 alertas rechazadas.**

# COMMAND ----------

#Filtramos para obtener los i_indicadores asociados a alertas rechazadas.
#Luego se crea un id para cada una de estas alertas (esto debido a que un i_indicador puede estar relacionado a más de una alerta)
tempdf= df_alerts_mot.loc[df_alerts_mot['status']=='Declined']
tempdf['tempid'] = range(1, len(tempdf) + 1)
new_temp = tempdf.loc[:, ['i_indicador', 'tempid']]

#Guardamos en una lista los i_indicadores que se relacionan con alertas rechazadas, luego obtenemos las mediciones de los sensores, con sus fechas asociadas.
i_indicador_rechazadas= new_temp['i_indicador'].tolist()
declined= df_features_mot[df_features_mot['i_indicador'].isin(i_indicador_rechazadas)]

#Agregamos a las mediciones de RMS, fecha, i_indicador, id_alerta la columna tempid, que contiene un id para cada alerta rechazada.
rechazadas = pd.merge(declined, new_temp, on='i_indicador')
rechazadas['RMS_normalized'] = pp.minmax_scale(rechazadas.loc[ : , 'RMS'], feature_range = (0, 10), axis = 0, copy = True)

#Crear columna con la fecha
rechazadas['date'] = rechazadas['fh'].dt.date

#Columna con la mediana de los RMS normalizados por fecha
rechazadas['mediana']= rechazadas.groupby('date')['RMS_normalized'].transform('median')
 
#rechazadas.drop(columns=['fh', 'RMS'], inplace=True)

#Cambiamos los nombres de las columnas que utilizará el modelo en el formato adecuado.
rechazadas.rename(columns={'date':'ds', 'mediana':'y'}, inplace=True)

#Agrupamos el df anterior según el id de cada alerta rechazada, luego creamos una lista de df que contendrá las fechas, RMS, i_indicador, id_alert asociado a cada id de alerta rechazada.
grupo=rechazadas.groupby(['tempid'])

ts_r= []

for name, group in grupo:
      ts_r.append(group)

# COMMAND ----------

# Hacemos que df asociado a su alerta entre al modelo, se grafiquen puntos de cambio y posteriormente se calculen pendiente e intercepto de la tendencia en el último segmento (entre el último punto de cambio y la fecha de emisión de alerta)

for df in ts_r: 
  
  m = Prophet(changepoint_range=1, changepoint_prior_scale=0.8) 
  m.fit(df)

  column_value = str(df['tempid'].iloc[0])
  in_sample_forecast = m.predict(df)
  fig = m.plot(in_sample_forecast)
  a = add_changepoints_to_plot(fig.gca(), m, in_sample_forecast)
  plt.title(f'Segmentaciones para alerta_rechazada_id={column_value}')

  final_ds=df['ds'].max()

  if len(signif_bkps(m)) > 0:
      final_bkps= signif_bkps(m).dt.date.max()

  else:
      final_bkps=df['ds'].min()
      
  final_bloque= in_sample_forecast.loc[(in_sample_forecast['ds'].dt.date >= final_bkps) & (in_sample_forecast['ds'].dt.date <= final_ds)]
  temp_dates = final_bloque['ds'].apply(lambda x: x.timestamp())
  final_bloque['dt_timestamp'] = temp_dates / 86400
  pend, inte = np.polyfit(final_bloque['dt_timestamp'], final_bloque['trend'], 1)
  df['pendiente']=pend 
  df['intercepto']=inte

# COMMAND ----------

alertas_rechazadas = pd.concat(ts_r, axis=0, ignore_index=True).drop_duplicates(subset=['i_indicador','pendiente', 'intercepto', 'tempid'])
alertas_rechazadas.drop(['ds', 'y'], axis=1, inplace=True)
alertas_rechazadas['pendienteabs']=np.abs(alertas_rechazadas['pendiente'])
display(alertas_rechazadas)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Alertas Aceptadas

# COMMAND ----------

tempdf= df_alerts_mot.loc[df_alerts_mot['status']=='Approved']
tempdf['tempid'] = range(1, len(tempdf) + 1)
new_temp = tempdf.loc[:, ['i_indicador', 'tempid']]

i_indicador_aceptadas= new_temp['i_indicador'].tolist()
approved= df_features_mot[df_features_mot['i_indicador'].isin(i_indicador_aceptadas)]
aceptadas = pd.merge(approved, new_temp, on='i_indicador')
aceptadas['RMS_normalized'] = pp.minmax_scale(aceptadas.loc[ : , 'RMS'], feature_range = (0, 10), axis = 0, copy = True)

aceptadas['date'] = aceptadas['fh'].dt.date

#Columna con la mediana de los RMS normalizados por fecha
aceptadas['mediana']= aceptadas.groupby('date')['RMS_normalized'].transform('median')
 
aceptadas.drop(columns=['fh', 'RMS'], inplace=True)

#Cambiamos los nombres de las columnas que utilizará el modelo en el formato adecuado.
aceptadas.rename(columns={'date':'ds', 'mediana':'y'}, inplace=True)

grupo=aceptadas.groupby(['tempid'])

ts_a= []

for name, group in grupo:
    ts_a.append(group)

# COMMAND ----------

aceptadas

# COMMAND ----------

for df in ts_a: 
  
  m = Prophet(changepoint_range=1, changepoint_prior_scale=0.8) 
  m.fit(df) 

  column_value = str(df['tempid'].iloc[0])
  in_sample_forecast = m.predict(df)
  fig = m.plot(in_sample_forecast)
  a = add_changepoints_to_plot(fig.gca(), m, in_sample_forecast)
  plt.title(f'Segmentaciones para alerta_aceptada_id={column_value}')
  
  final_ds=df['ds'].max()
      
  if len(signif_bkps(m)) > 0:
      final_bkps= signif_bkps(m).dt.date.max()

  else:
      final_bkps=df['ds'].min()
      
  final_bloque= in_sample_forecast.loc[(in_sample_forecast['ds'].dt.date >= final_bkps) & (in_sample_forecast['ds'].dt.date <= final_ds)]
  temp_dates = final_bloque['ds'].apply(lambda x: x.timestamp())
  final_bloque['dt_timestamp'] = temp_dates / 86400
  pend, inte = np.polyfit(final_bloque['dt_timestamp'], final_bloque['trend'], 1)
  df['pendiente']=pend
  df['intercepto']=inte

# COMMAND ----------

alertas_aceptadas = pd.concat(ts_a, axis=0, ignore_index=True).drop_duplicates(subset=['i_indicador','pendiente', 'intercepto', 'tempid'])
alertas_aceptadas.drop(['ds', 'y'], axis=1, inplace=True)
display(alertas_aceptadas)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Gráficos

# COMMAND ----------

boxplot_pendiente(alertas_aceptadas.loc[(alertas_aceptadas['pendiente'] < 0.03) & (alertas_aceptadas['pendiente'] > -0.02)], alertas_rechazadas.loc[(alertas_rechazadas['pendiente'] < 0.005) & (alertas_rechazadas['pendiente'] > -0.008)])

# COMMAND ----------

# Selecciona las dos columnas que quieres comparar
columna1 = alertas_aceptadas['pendiente']
columna2 = alertas_rechazadas['pendiente']

# Calcula las estadísticas descriptivas de cada columna
media1 = columna1.mean()
media2 = columna2.mean()
std1 = columna1.std()
std2 = columna2.std()
n1 = len(columna1)
n2 = len(columna2)

# Calcula el estadístico t y los grados de libertad
t, p = stats.ttest_ind(columna1, columna2)
df = n1 + n2 - 2

# Selecciona el nivel de significancia y el valor crítico de t
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, df)

# Compara el valor de t con el valor crítico y concluye
if abs(t) > t_critical:
    print("Se rechaza la hipótesis nula: las medias son diferentes.")
else:
    print("No se puede rechazar la hipótesis nula: las medias son iguales.")

# COMMAND ----------

histograma_kde_pendientes(alertas_rechazadas.loc[(alertas_rechazadas['pendiente'] < 0.005) & (alertas_rechazadas['pendiente'] > -0.008)], alertas_aceptadas.loc[(alertas_aceptadas['pendiente'] < 0.03) & (alertas_aceptadas['pendiente'] > -0.02)])

# COMMAND ----------

boxplot_angulo(alertas_aceptadas, alertas_rechazadas)

# COMMAND ----------

# MAGIC %md
# MAGIC **Alertas rechazadas**

# COMMAND ----------

display(alertas_rechazadas['pendiente'].describe())

# COMMAND ----------

len(alertas_rechazadas.loc[alertas_rechazadas['pendiente'] < 0])

# COMMAND ----------

# MAGIC %md
# MAGIC **Alertas aceptadas***

# COMMAND ----------

display(alertas_aceptadas['pendiente'].describe())

# COMMAND ----------

len(alertas_aceptadas.loc[alertas_aceptadas['pendiente'] < 0])

# COMMAND ----------

tempid_neg_aceptadas= alertas_aceptadas.loc[alertas_aceptadas['pendiente'] < 0]['tempid'].tolist()
result_df= [df for df in ts_a if df['tempid'].isin(tempid_neg_aceptadas).any()]

for df in result_df: 
  
  m = Prophet(changepoint_range=1, changepoint_prior_scale=0.8) 
  m.fit(df)

  column_value = str(df['tempid'].iloc[0])
  in_sample_forecast = m.predict(df)
  fig = m.plot(in_sample_forecast)
  a = add_changepoints_to_plot(fig.gca(), m, in_sample_forecast)
  plt.title(f'Segmentaciones para alerta_aceptada_id={column_value}')

