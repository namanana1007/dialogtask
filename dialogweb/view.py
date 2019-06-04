from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from . import diaagent

@csrf_exempt
def dia_response(request):
	return_value = {}
	if request.method == "POST":
		#print(request.POST)
		user_text = request.POST['user_text']
        #print("user_text",user_text)
		if diaagent.get_first_time_value():
			print("First time!")
			ansstr, episode_over = diaagent.reset_dialog_manager(user_text)
		else:
			ansstr, episode_over = diaagent.next_dialos_manager(user_text)
		return_value = {
			"ansstr":ansstr,
			"episode_over":episode_over
		}
	#print(HttpResponse(json.dumps(return_value))
	return HttpResponse(json.dumps(return_value))

@csrf_exempt
def chushihua(request):
	diaagent.set_first_time_value(True)
	my_ip = request.build_absolute_uri('/')
	#print("my IP:", my_ip)
	ctx = {}
	ctx['my_ip'] = my_ip
	return render(request, "ceshi.html", ctx)
