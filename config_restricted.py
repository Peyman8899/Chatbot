"""Configuration for restricted environment"""
import json
import httpx

# Token-based authentication
VERIFY_WIN = "//ms/dist/sec/PRoJ/pki/prod/truststores/GenPop-PROD/combined/pem/bundle.pem"
token_json = httpx.get("http://mrmeed:23123/token", verify=VERIFY_WIN).text
token = json.loads(token_json)

# Configuration
config = {
    "api_base": "https://aigateway-dev.ms.com/openai/v1/",
    "api_key": token,
    "chroma_path": "C:/MSDE/peymana/chroma_repo",
    "embedding_model": "text-embedding-ada-002"
}
