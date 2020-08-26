from django.urls import path
from .views import HomeView, TfaView, ThfView

urlpatterns = [
    path('tourist_hotspot_forecast/', ThfView.as_view(), name='thf'),
    path('tourist_flow_analysis/', TfaView.as_view(), name='tfa'),
    path('', HomeView.as_view(), name='home'),
]