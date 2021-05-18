from django.shortcuts import render
from django.http import HttpResponse

import joblib

# Create your views here.

def index(request):
    my_dict= {"hello":"Hello world"}
    return render(request, 'car_app/index.html',context=my_dict)

def result(request):
        rf_reg= joblib.load('rf_car_model.sav')

        new_lst=[]

        new_lst.append(request.GET.get('Present_Price'))
        new_lst.append(request.GET.get('Kms_Driven'))
        new_lst.append(request.GET.get('years'))
        new_lst.append(request.GET.get('Fuel'))
        new_lst.append(request.GET.get('Seller_Type'))
        new_lst.append(request.GET.get('Transmission'))
        new_lst.append(request.GET.get('Owner'))


        new_lst[2] = str(2021 - int(new_lst[2]))
        print(new_lst)
        sell_price= rf_reg.predict([new_lst])
        return render(request,'car_app/result.html',{'sell_price':sell_price})
