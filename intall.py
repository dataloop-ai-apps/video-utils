import os
from dotenv import load_dotenv
import dtlpy as dl

# from rfdetr.util.coco_classes import COCO_CLASSES

use_rc_env = False
if use_rc_env:
    dl.setenv('rc')
else:
    dl.setenv('prod')
if dl.token_expired():
    dl.login()

if use_rc_env:
    project = dl.projects.get(project_name='Husam Testing')
else:
    # project = dl.projects.get(project_name='ShadiDemo')
    project = dl.projects.get(project_name='IPM development')
    # project = dl.projects.get(project_name='Boston Dynamics POC')

models = project.models.list().print()

# Publish your app
# dpk = project.dpks.publish()
dpk = project.dpks.publish(local_path=os.getcwd(), manifest_filepath=r'splitting\\dataloop.json')

# Install or update the application
try:
    app = project.apps.get(app_name=dpk.display_name)
    app.dpk_version = dpk.version
    app.update()
except dl.exceptions.NotFound:
    print("installing ...")
    app = project.apps.install(dpk=dpk)


dpk = project.dpks.publish(local_path=os.getcwd(), manifest_filepath=r'stitching\\dataloop.json')

# Install or update the application
try:
    app = project.apps.get(app_name=dpk.display_name)
    app.dpk_version = dpk.version
    app.update()
except dl.exceptions.NotFound:
    print("installing ...")
    app = project.apps.install(dpk=dpk)
