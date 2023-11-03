import time
import os
from gptcache import cache, Config
from weaviate import EmbeddedOptions, Client
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.embedding.huggingface import Huggingface
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter import openai
WEAVIATE_URL = 'http://localhost:8080'
EMBED_PATH = "/alidata/models/moka-ai/m3e-base/"
CLASS_NAME = "GPTCache"
MODEL_NAME = "baichuan2-13b-int4"
PERSISTENCE_PATH = "/alidata/persistence/weawiate/"
# 必须以openai的格式提供服务
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://0.0.0.0:20000/v1"

cache.set_openai_key()

embed_model = Huggingface(model=EMBED_PATH)

cache_base = CacheBase(name="postgresql",
                       sql_url="postgresql+psycopg2://postgres:postgres@0.0.0.0:5432/gptcache",
                       table_name="gptcache",
                       table_len_config=None)

embed_option = EmbeddedOptions(persistence_data_path=PERSISTENCE_PATH)
vector_base = VectorBase(
    name="weaviate",
    url='http://localhost:8080',
    port="8080",
    startup_period=50,
    embeded_options=embed_option)


# data_manager = manager_factory("sqlite,faiss", vector_params={"dimension":embed_model.dimension})
data_manager = get_data_manager(cache_base=cache_base,vector_base=vector_base)
cache.init(
    embedding_func=embed_model.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=temperature_softmax
    )
cache.config = Config(similarity_threshold=0.8)

question = "what's github"

for _ in range(3):
    start = time.time()
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature = 1.0,  # Change temperature here
        messages=[{
            "role": "user",
            "content": question
        }],
    )
    print("Time elapsed:", round(time.time() - start, 3))
    print("Answer:", response["choices"][0]["message"]["content"])