from inference_sdk import InferenceHTTPClient, InferenceConfiguration


CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="bhE3MAgNEQBERjkCsH1T"
).configure(InferenceConfiguration(
    api_key_transport="header"  # header-based auth (inference v1.5.0+)
))

result = CLIENT.infer("pool.jpg", model_id="pool-billiard-n0ywn/1")

# print(result)