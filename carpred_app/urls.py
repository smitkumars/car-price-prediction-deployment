from django.urls import path,include
from carpred_app import views

app_name= 'carpred_app'
urlpatterns= [
    path('',views.result,name='result'),
]
