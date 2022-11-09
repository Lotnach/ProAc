import imp
from django.shortcuts import render
from django.views.generic import TemplateView
from .models import Insec, InsecFactory
from django.contrib.auth.mixins import LoginRequiredMixin
from django.template import RequestContext

#-------------- imports para predecir valores accionarios ------
import math
import pandas_datareader as web
import numpy as np
import tensorflow as tf
np.random.seed(4)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yf
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)
#--------------------------------------
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.http import HttpResponse, request
import io
import urllib, base64
from io import BytesIO




# Creacion de vistas.---------------------------------------------------------------------------------------------------------

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)

class HomeInsecView(LoginRequiredMixin,TemplateView):
    def get(self, request, **kwargs):
        insecFactory=InsecFactory()
        return render(request, 'insecs.html', {'insecs': insecFactory.obtenerInsecs()})

class DetalleInsecView(LoginRequiredMixin,TemplateView):
    def get(self, request, **kwargs):
        insecFactory=InsecFactory()
        llave=kwargs["llave"]
        return render(request, 'insec.html', {'insec': insecFactory.getInsec(llave)})

class CopecPredictionView(LoginRequiredMixin,TemplateView):
    def get(self, request, **kwargs):
        #Generamos las cotizaciones
        df = web.DataReader('COPEC.SN', data_source='yahoo', start='2019-01-01', end='2022-09-21')
        df.shape

        #Seleccionamos los datos para la prediccion usando solo el cierre de la base historica de datos y convertimos en un numpy array
        datos = df.filter(['Close'])
        dataset = datos.values
        training_data_len = math.ceil(len(dataset) * .8) #numero de filas en el historico

        #normalizamos el set de entrenamiento
        sc = MinMaxScaler(feature_range=(0,1))
        set_entrenamiento_escalado = sc.fit_transform(dataset)

        #creamos el entrenamiento de la data
        train_data = set_entrenamiento_escalado[0:training_data_len, :]
        x_train = []
        y_train = []
        time_step = 60
        m = len(train_data)

        for i in range(time_step, m):
            x_train.append(train_data[i-time_step:i, 0])
            y_train.append(train_data[i, 0])        
        x_train, y_train = np.array(x_train), np.array(y_train) # transformamos a numpy array

        #remodelamos los datos, ya que la red LSTM espera que la entrada sea tridimensional y en estos momentos es bidimensional
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        #construimos el modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        #creamos el conjunto de datos de prueba
        prediction = set_entrenamiento_escalado[training_data_len - time_step: , :]
        x_predic = []
        y_predic = dataset[training_data_len:, :]

        for i in range(time_step, len(prediction)):
            x_predic.append(prediction[i-time_step:i, 0])

        #convertimos en numpy array y remodelamos los datos
        x_predic = np.array(x_predic)
        x_predic = np.reshape(x_predic, (x_predic.shape[0], x_predic.shape[1], 1))

        #obtenemos el precio y valores proyectados del modelo
        proyeccion = model.predict(x_predic)
        proyeccion = sc.inverse_transform(proyeccion)
        rmse = np.sqrt(np.mean(proyeccion - y_predic)**2)

        #----------------------------------------------------PASADO-----------------------------------------------------------
        #reusamos la data

        dataset2 = dataset
        dataset2 = dataset2.reshape(-1, 1)

        #scalamos la data
        scaler = sc
        scaler = scaler.fit(dataset2)
        dataset2 = scaler.transform(dataset2)

        #generamos los input y output
        pasado = 120
        futuro = 60

        xfut = []
        yfut = []

        for i in range (pasado, len(dataset2) - futuro + 1):
            xfut.append(dataset2[i - pasado: i])
            yfut.append(dataset2[i: i + futuro])

        xfut = np.array(xfut)
        yfut = np.array(yfut)

        #Generamos el modelo
        model2 = Sequential()
        model2.add(LSTM(units=50, return_sequences=True, input_shape=(pasado, 1)))
        model2.add(LSTM(units=50))
        model2.add(Dense(futuro))
        model2.compile(loss='mean_squared_error', optimizer='adam')
        model2.fit(xfut, yfut, epochs=1, batch_size=1)

        Xfuture = dataset2[- pasado:]
        Xfuture = Xfuture.reshape(1, pasado, 1)
        Yfuture = model2.predict(Xfuture).reshape(-1, 1)
        Yfuture = scaler.inverse_transform(Yfuture)

        #organizamos el resultado

        df_past = datos.reset_index()
        df_past.rename(columns={'index': 'Date'}, inplace=True)
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['futuro'] = np.nan
        df_past['futuro'].iloc[-1] = df_past['Close'].iloc[-1]
        

        df_future = pd.DataFrame(columns=['Date', 'Close', 'futuro'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=futuro)
        df_future['futuro'] = Yfuture.flatten()
        df_future['Close'] = np.nan
        

        #graficamos
        train = datos[:training_data_len]
        valid = datos[training_data_len:]
        valid['proyeccion'] = proyeccion
        valid2 = pd.concat([df_past, df_future]).set_index('Date') #futuro
        plt.figure(figsize=(11,7))
        plt.title('Copec proyeccion')
        plt.xlabel('Date', fontsize = 12)
        plt.ylabel('Close Price CLP', fontsize=12)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'proyeccion']])
        plt.plot(valid2['futuro'])
        plt.legend(['Train', 'Val', 'Proyecciones', 'futuro'], loc = 'lower right')        
        #mostramos el grafico
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        encoded = base64.b64encode(img.getvalue())
        my_html = format(encoded.decode('utf-8'))        
        return render(request, 'copec.html', {'image': my_html})
#-------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------

class LipigasPredictionView(LoginRequiredMixin,TemplateView):
    def get(self, request, **kwargs):

        #Generamos las cotizaciones
        df = web.DataReader('LIPIGAS.SN', data_source='yahoo', start='2019-01-01', end='2022-09-21')
        df.shape

        #Seleccionamos los datos para la prediccion usando solo el cierre de la base historica de datos y convertimos en un numpy array
        datos = df.filter(['Close'])
        dataset = datos.values        
        training_data_len = math.ceil(len(dataset) * .8) #numero de filas en el historico

        #normalizamos el set de entrenamiento
        sc = MinMaxScaler(feature_range=(0,1))
        set_entrenamiento_escalado = sc.fit_transform(dataset)
        
        #creamos el entrenamiento de la data
        train_data = set_entrenamiento_escalado[0:training_data_len, :]
        x_train = []
        y_train = []
        time_step = 60
        m = len(train_data)

        for i in range(time_step, m):
            x_train.append(train_data[i-time_step:i, 0])
            y_train.append(train_data[i, 0])        
        x_train, y_train = np.array(x_train), np.array(y_train) # transformamos a numpy array

        #remodelamos los datos, ya que la red LSTM espera que la entrada sea tridimensional y en estos momentos es bidimensional
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        #construimos el modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        #creamos el conjunto de datos de prueba
        prediction = set_entrenamiento_escalado[training_data_len - time_step: , :]
        x_predic = []
        y_predic = dataset[training_data_len:, :]

        for i in range(time_step, len(prediction)):
            x_predic.append(prediction[i-time_step:i, 0])

        #convertimos en numpy array y remodelamos los datos
        x_predic = np.array(x_predic)
        x_predic = np.reshape(x_predic, (x_predic.shape[0], x_predic.shape[1], 1))

        #obtenemos el precio y valores proyectados del modelo
        proyeccion = model.predict(x_predic)
        proyeccion = sc.inverse_transform(proyeccion)
        rmse = np.sqrt(np.mean(proyeccion - y_predic)**2)
        

        #----------------------------------------------------PASADO-----------------------------------------------------------
        #reusamos la data

        dataset2 = dataset
        dataset2 = dataset2.reshape(-1, 1)

        #scalamos la data
        scaler = sc
        scaler = scaler.fit(dataset2)
        dataset2 = scaler.transform(dataset2)

        #generamos los input y output
        pasado = 120
        futuro = 60

        xfut = []
        yfut = []

        for i in range (pasado, len(dataset2) - futuro + 1):
            xfut.append(dataset2[i - pasado: i])
            yfut.append(dataset2[i: i + futuro])

        xfut = np.array(xfut)
        yfut = np.array(yfut)

        #Generamos el modelo
        model2 = Sequential()
        model2.add(LSTM(units=50, return_sequences=True, input_shape=(pasado, 1)))
        model2.add(LSTM(units=50))
        model2.add(Dense(futuro))        
        model2.compile(loss='mean_squared_error', optimizer='adam')
        model2.fit(xfut, yfut, epochs=1, batch_size=1)

        Xfuture = dataset2[- pasado:]
        Xfuture = Xfuture.reshape(1, pasado, 1)
        Yfuture = model2.predict(Xfuture).reshape(-1, 1)
        Yfuture = scaler.inverse_transform(Yfuture)

        #organizamos el resultado

        df_past = datos.reset_index()
        df_past.rename(columns={'index': 'Date'}, inplace=True)
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['futuro'] = np.nan
        df_past['futuro'].iloc[-1] = df_past['Close'].iloc[-1]
        

        df_future = pd.DataFrame(columns=['Date', 'Close', 'futuro'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=futuro)
        df_future['futuro'] = Yfuture.flatten()
        df_future['Close'] = np.nan
        

        #graficamos ----------------------------------------------------------------------------------------------------------------------------------
        train = datos[:training_data_len]
        valid = datos[training_data_len:]
        valid['proyeccion'] = proyeccion
        valid2 = pd.concat([df_past, df_future]).set_index('Date') #futuro
        
        plt.figure(figsize=(11,7))
        plt.title('Lipigas proyeccion')
        plt.xlabel('Date', fontsize = 12)
        plt.ylabel('Close Price CLP', fontsize=12)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'proyeccion']]) 
        plt.plot(valid2['futuro']) #futuro        
        plt.legend(['Train', 'Val', 'Proyecciones', 'futuro'], loc = 'lower right')   #tambien 
        #mostramos el grafico
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        encoded = base64.b64encode(img.getvalue())
        my_html = format(encoded.decode('utf-8'))        
        return render(request, 'Lipigas.html', {'image': my_html})

class EnergyPredictionView(LoginRequiredMixin,TemplateView):
    def get(self, request, **kwargs):

        #Generamos las cotizaciones
        dataset = pd.read_csv('insecs\SPCLX_ENERGY_S_History.csv', index_col='FECHA', parse_dates=['FECHA'])
        dataset.head()

        plt.figure(figsize=(14,7))        
        plt.plot(dataset['CLOSE'])
        plt.xlabel('FECHA', fontsize = 12)
        plt.ylabel('Close Price CLP', fontsize = 12)
                
        #mostramos el grafico
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        encoded = base64.b64encode(img.getvalue())
        my_html = format(encoded.decode('utf-8'))        
        return render(request, 'energy.html', {'image': my_html})