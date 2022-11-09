#Insec/urls

from django.urls import re_path, path, include
from insecs import views

urlpatterns = [
    re_path(r'^$',views.HomePageView.as_view(),name="index"),
    re_path(r'insecs/', views.HomeInsecView.as_view(), name="insecs"),
    re_path(r'^insec/(?P<llave>ISC[0-9]{3})/$', views.DetalleInsecView.as_view(), name="detalle"),
    re_path(r'^copec/', views.CopecPredictionView.as_view(), name="copec"),
    re_path(r'^Lipigas/', views.LipigasPredictionView.as_view(), name="Lipigas.SN"),
    re_path(r'^energy/', views.EnergyPredictionView.as_view(), name="SPCLX-ENERGY.SN"),
    path('accounts/', include('accounts.urls')),
    path('accounts/', include('django.contrib.auth.urls'))
]