# Let's get this party started!
import falcon
import boto3

def s3client():
    return boto3.client(
        's3',
        aws_access_key_id="AKIAXTALDU7HK57Y3TEN",
        aws_secret_access_key="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",
        region_name="eu-west-1"
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


for key in s3client().list_objects(Bucket='quana-prod', Delimiter='/').get('CommonPrefixes', []):
    folder_name = key['Prefix'][:-1]
    print(folder_name)