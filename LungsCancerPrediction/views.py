from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView

def homePageView(request):
    return render(request, 'index.html')

def output(request):
    return render(request, 'output.html')


