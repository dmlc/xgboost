import boto3
import json

lambda_client = boto3.client('lambda', region_name='us-west-2')

# Source code for the Lambda function is available at https://github.com/hcho3/xgboost-devops
r = lambda_client.invoke(
    FunctionName='XGBoostCICostWatcher',
    InvocationType='RequestResponse',
    Payload='{}'.encode('utf-8')
)

payload = r['Payload'].read().decode('utf-8')
if 'FunctionError' in r:
    msg = 'Error when invoking the Lambda function. Stack trace:\n'
    error = json.loads(payload)
    msg += f"    {error['errorType']}: {error['errorMessage']}\n"
    for trace in error['stackTrace']:
        for line in trace.split('\n'):
            msg += f'    {line}\n'
    raise RuntimeError(msg)
response = json.loads(payload)
if response['approved']:
    print(f"Testing approved. Reason: {response['reason']}")
else:
    raise RuntimeError(f"Testing rejected. Reason: {response['reason']}")
