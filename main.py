# Let's get this party started!
import falcon
import boto3
import os
import ml
import io
import requests
import shutil
from time import gmtime, strftime
import json
from threading import Thread
import pickle
import zipfile

from dotenv import load_dotenv
load_dotenv()

from falcon_multipart.middleware import MultipartMiddleware


MAIN_SERVER_KEY = os.getenv('MAIN_SERVER_KEY', "Test")
MAIN_SERVER_URL = os.getenv('MAIN_SERVER_URL', "http://localhost:3000")
AWS_BUCKET      = os.getenv('AWS_BUCKET', "NOBUCKET")

AWS_ACCESS_KEY  = os.getenv('AWS_ACCESS_KEY', "NOKEY")
AWS_SECRET_KEY  = os.getenv('AWS_SECRET_KEY', "NOSECRET")
AWS_REGION      = os.getenv('AWS_REGION', "NOREGION")

main_folder = "ml"

def train_process(id, url):
    print("Downloading:",id, url)
    path = download(id, url)
    print("Uploading:",id, url)
    if upload(id, path, "samples.zip") != True:
        print("upload failed", id, url)
        return
    print("Oppenning:",id, url)
    dfs = ml.open_zip(path)
    if dfs == None:
        print("open failed", id, url)
        return;
    print(dfs[0])
    print(len(dfs))

    #remove tmp file
    try_remove(path)

    print("Trainings:",id, url)
    agent = ml.train(dfs)
    print(agent)

    print("Exporting:",id, url)
    path = "./tmp/" + id + "_agents.zip"
    agent.export_file(path)
    print("Uploading export:",id, url)
    upload(id, path, "agents.zip")
    print("Done", id, url)

    #remove tmp file agents
    try_remove(path)


def try_remove(file_path):
    try: 
        os.remove(file_path) 
    except Exception as e:
        print("Remove failed", e)
        return False
    return True    
            

def s3client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )


#agents simple caching
max_agents = 10
agents = {}
def get_agent(agent_id):
    global agents;
    agent = {}
    if agent_id in agents:
        print("From cache", agent_id)
        agent = agents[agent_id][0]
        agents[agent_id] = (agent, gmtime())
        return agent

    print("From storage", agent_id)
    try:
        s3_response_object = s3client().get_object(Bucket=AWS_BUCKET, Key=main_folder + "/" + agent_id + "/agents.zip")
        agent = ml.open_zip_agent(io.BytesIO(s3_response_object['Body'].read()))
        if agent != None:
            agents[agent_id] = (agent, gmtime())
            if len(agents) > max_agents:
                min_key = min(agents.keys(), key=(lambda k: agents[k][1]))
                agents.pop(min_key, None)

        return agent  
    except Exception as e:
        print("Error getting agent", agent_id, e)
        return None    
          
def classify(agent_id, file):
    agent = get_agent(agent_id)
    if agent == None:
        return None

    print(agent.agents)
        
    return ml.classify(agent, file)

class TrainRequest(object):
    def on_post(self, req, resp):
        data = json.load(req.bounded_stream)

        if 'samples' not in data:
            resp.status = falcon.HTTP_400 
            resp.body = ('{}')
            return

        #assign some id
        id = "samples" + strftime("_%Y_%m_%d_%H-%M-%S", gmtime())

        #Spawn thread to process the data
        t = Thread(target=train_process, args=(id, data['samples'], ))
        t.start()

        resp.status = falcon.HTTP_200  
        resp.body = (json.dumps({'id': id}))


class ClassifyRequest(object):
    def on_post(self, req, resp):
        # Retrieve input_file
        input_file = req.get_param('sample')
        version = req.get_param('version')

        # Test if the file was uploaded
        if input_file == None or version == None:
            resp.status = falcon.HTTP_400 
            resp.body = ('{}')
            return

        print(input_file.filename, version)
        output = classify(version, input_file.file)

        if output == None:
            resp.status = falcon.HTTP_400
            resp.body = (json.dumps({}))
            return;

        resp.status = falcon.HTTP_200
        resp.body = (json.dumps(output))

class VersionsRequest(object):
    def on_get(self, req, resp):
        versions = []
        for key in s3client().list_objects(Bucket=AWS_BUCKET, Prefix=main_folder + "/", Delimiter='/').get('CommonPrefixes', []):
            folder_name = key['Prefix'][len(main_folder) + 1:-1]
            versions.append(folder_name)

        resp.status = falcon.HTTP_200
        resp.body = (json.dumps(versions))

def download(id, url):
    req = requests.get(url, headers={'x-api-key': MAIN_SERVER_KEY}, stream=True)
    
    folder = "./tmp"

    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = id + ".zip"
    file_path = folder + "/" + file_name

    with open(file_path, 'wb') as f:
        shutil.copyfileobj(req.raw, f)

    return file_path

def upload(id, file_path, name):
    filename = os.path.basename(file_path)

    s3_client = s3client()
    try:
        response = s3_client.upload_file(file_path, AWS_BUCKET, main_folder + "/" + id + "/" + name)
    except Exception as e:
        print("Error", e)
        return False
    return True

def update_versions():
    versions = []
    global main_version

    for key in s3client().list_objects(Bucket=AWS_BUCKET, Prefix=main_folder + "/", Delimiter='/').get('CommonPrefixes', []):
        folder_name = key['Prefix'][len(main_folder)+1:-1]
        versions.append(folder_name)
    main_version = max(versions)

    try:
        req = requests.get(MAIN_SERVER_URL + "/ml/version", headers={'x-api-key': MAIN_SERVER_KEY})
        result = req.json()
        version = result["version"]
        if version != None and version in versions:
            main_version = version
    except Exception as e:
        print("Error getting version", e)

    return main_version


# falcon.API instances are callable WSGI apps
app = falcon.API(middleware=[MultipartMiddleware()])

# things will handle all requests to the '/things' URL path
app.add_route('/train', TrainRequest())
app.add_route('/classify', ClassifyRequest())
app.add_route('/versions', VersionsRequest())

