# Let's get this party started!
import falcon
import boto3
import os
import ml

from dotenv import load_dotenv
load_dotenv()

def s3client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY', "NOKEY"),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY', "NOSECRET"),
        region_name=os.getenv('AWS_REGION', "NOREGION")
    )

class TrainRequest(object):
    def on_post(self, req, resp):
        resp.status = falcon.HTTP_200  # This is the default status
        resp.body = ('\nTwo things awe me most, the starry sky '
                     'above me and the moral law within me.\n'
                     '\n'
                     '    ~ Immanuel Kant\n\n')



# falcon.API instances are callable WSGI apps
app = falcon.API()

# things will handle all requests to the '/things' URL path
app.add_route('/train', TrainRequest())

main_folder = "ml/"

for key in s3client().list_objects(Bucket=os.getenv('AWS_BUCKET', "NOBUCKET"), Prefix=main_folder, Delimiter='/').get('CommonPrefixes', []):
    folder_name = key['Prefix'][len(main_folder):-1]
    print(folder_name)