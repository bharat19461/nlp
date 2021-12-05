from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
from home.helper import *

# Create your views here.
def home(request):
	context = {'printresult' : False}
	return render(request, 'home.html', context)

def process(request):
	if request.method == 'POST':
		text = request.POST['text'] if 'text' in request.POST else ''

		if len(text) == 0:
			return HttpResponse('Empty String!!')

		#print(text)
		result = message_detection(text)
		context = {'message' : text, 'result' : result, 'printresult' : True}
		return render(request, 'home.html', context)

	else:
		return HttpResponse('You need to give input by clicking button')

	pass
