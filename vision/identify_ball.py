from inference_sdk import InferenceHTTPClient, InferenceConfiguration

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="API_KEY"
).configure(InferenceConfiguration(
    api_key_transport="header"  # header-based auth (inference v1.5.0+)
))

result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="pool-ball-detection-jhi6w/5")