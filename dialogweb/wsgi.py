"""
WSGI config for HelloWorld project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from . import diaagent

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dialogweb.settings")

application = get_wsgi_application()

diaagent.init_dialog_manager()
