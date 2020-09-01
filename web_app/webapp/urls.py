from django.urls import path

from . import views
from .views import HomeView, TfaView, ThfView, TrsView, ContactView

urlpatterns = [
    path('tourist_hotspot_forecast/', ThfView.as_view(), name='thf'),
    path('tourist_flow_analysis/', TfaView.as_view(), name='tfa'),
    path('tourism_recommendation_system/', TrsView.as_view(), name='trs'),
    path('contact/', ContactView.as_view(), name='contact'),
    path('', HomeView.as_view(), name='home'),
]
