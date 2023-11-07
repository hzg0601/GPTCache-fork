import time
import os
from gptcache import cache, Config
from weaviate import EmbeddedOptions, Client
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.embedding.huggingface import Huggingface
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
#GPTCache对openai的各个类(ChatCompletion,Completion,Audio,Image,Moderation)进行了改造
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
# name为必须字段
# 针对sql类，需要提供sql_url,table_name='gptcache',table_len_config
# 针对mongodb，需要mongo_host,mongo_port,mongo_db,usernanme,password
# 针对redis,需要提供redis_host,redis_port,global_key_prefix
# 针对dynamo，需要提供aws_access_key_id，aws_secret_access_key，region_name，aws_profile_name，endpoint_url

cache_base = CacheBase(name="postgresql",
                       sql_url="postgresql+psycopg2://postgres:postgres@0.0.0.0:5432/gptcache",
                       table_name="gptcache",
                       table_len_config=None)

embed_option = EmbeddedOptions(persistence_data_path=PERSISTENCE_PATH)
# name为必须字段，top_k=1是公共字段，其他根据具体数据库配置，见gptcache.manager.vector_data.manager.VectorBase：
#    `Milvus` (with , `host`, `port`, `password`, `secure`, `collection_name`, `index_params`, `search_params`, `local_mode`, `local_data` params),
#    `Faiss` (with , `index_path`, `dimension`, `top_k` params),
#    `Chromadb` (with `top_k`, `client_settings`, `persist_directory`, `collection_name` params),
#    `Hnswlib` (with `index_file_path`, `dimension`, `top_k`, `max_elements` params).
#    `pgvector` (with `url`, `collection_name`, `index_params`, `top_k`, `dimension` params).
#     docarray
#     usearch
#     redis
#     qdrant
#     weaviate ()
#   
vector_base = VectorBase(
    name="weaviate",
    top_k=3,
    url='http://localhost:8080',
    port="8080",
    startup_period=50,
    embeded_options=embed_option)


# data_manager = manager_factory("sqlite,faiss", vector_params={"dimension":embed_model.dimension})
# get_data_manager：
    # cache_base: 关系数据库管理，sql类,mongo,redis,dynamo
    # vector_base: 向量数据库管理，
    # object_base : 存储位置管理，local, s3(amazon)
    # eviction_base: 缓存数据淘汰管理，策略LRU=True,FIFO,LFU,RR,对象memorycache=True,rediscache,noop
        # FIFO：First In First Out，先进先出，淘汰最早被缓存的对象；
        # LRU：Least Recently Used，淘汰最长时间未被使用的数据，以时间作为参考；
        # LFU：Least Frequently Used，淘汰一定时期内被访问次数最少的数据，以次数作为参考；
        # RR: Random Replacement 
        # 如果关系数据库标记为delete的数目超过MAX_MARK_COUNT=1,
        # 或者标记delete的比例占全部数据的MAX_MARK_RATE=0.1，则执行淘汰策略
        # 先删关系数据库，再删向量数据库，如果执行淘汰的次数大于等于REBUILD_CONDITION=5,
        # 则执行向量数据库的重建。

data_manager = get_data_manager(cache_base=cache_base,vector_base=vector_base)
cache.init(
    embedding_func=embed_model.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=temperature_softmax
    )



def query_with_gpt_cache(question:str,temperature:float=2.0,threshold=0.8):
    cache.config = Config(similarity_threshold=threshold)
    start = time.time()
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature = temperature,  # Change temperature here
        messages=[{
            "role": "user",
            "content": question
        }],
        skip_cache=True
    )
    # print("本次查询耗时:", round(time.time() - start, 3))
    # print("Answer:", response["choices"][0]["message"]["content"])
    elapse = round(time.time() - start, 3)
    answer_str = response["choices"][0]["message"]["content"]
    return elapse, answer_str


import streamlit as st 
from streamlit_chat import message

st.title(f"GPTCache Demo,Powered by {MODEL_NAME}")
st.subheader(":four_leaf_clover::four_leaf_clover::point_right:欢迎试用、反馈:point_left::clap::clap:")

temperature = st.slider(label=(":rainbow[请选择参数temperature以控制回答的随机程度 "
                               "注意：若选择<0,则可能跳过大模型完全基于语义搜索返回结果；"
                                "若>2,则跳过语义搜索，直接基于大模型返回结果]"),
                                min_value=-1.0,
                                max_value=3.0,
                                value=1.7)
threshold = st.slider(label=":rainbow[请选择参数similary_threshold以控制向量相似度检索阈值]",min_value=0.0,max_value=1.0,value=0.8)
# # “会话状态是为每个用户会话在重新运行（rerun）中共享变量的方法。除了存储和持久化的能力，还公开了使用回调操作状态的能力。”
# # session state只可能在应用初始化时被整体重置。而rerun不会重置session state。可以理解成“状态缓存”。
# # 可以使用任何函数、包括回调函数去调用或修改它的值

# user_input = st.text_input(label="请输入您的问题",placeholder="示例：你是谁",key="input")

# if "generated" not in st.session_state:
#     st.session_state["generated"] = []
# if "past" not in st.session_state:
#     st.session_state["past"] = []
# if "elapse" not in st.session_state:
#     st.session_state["elapse"] = []

# if user_input:
#     elapse, ouput = query_with_gpt_cache(user_input,temperature=temperature)
#     st.session_state['past'].append(user_input)
#     st.session_state['generated'].append(ouput)
#     # st.info(f"耗时：{elapse}s")
#     st.session_state['elapse'].append(f"本次查询耗时：{elapse}s")

# if st.session_state['generated']:
#     for i in range(len(st.session_state["generated"])-1,-1,-1):
#         message(st.session_state["elapse"][i])
#         message(st.session_state["generated"][i],key=str(i))
#         message(st.session_state["past"][i],
#                 is_user=True,
#                 key=str(i)+"_user")

# 初始化session_state
if "messages" not in st.session_state:
    st.session_state['messages'] = []
if "elapse" not in st.session_state:
    st.session_state["elapse"] = []
# 在rerun时仍显示历史信息
for i, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if i%2 == 1:
        run = int(i//2)
        st.markdown(f":blue[{st.session_state.elapse[run]}]")
# 输入信息
if prompt := st.chat_input(placeholder="请输入对话内容，换行请使用Shift+Enter ",key="prompt"):
    # 将message加入chat history
    st.session_state.messages.append(
        {"role":"user","content":prompt}
    )
    # 在chat message中显示用户输入
    st.markdown(prompt)

    # 在chat_message中显示ai返回答案
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        elapse,response = query_with_gpt_cache(question=prompt,temperature=temperature,threshold=threshold)

        message_placeholder.markdown(response)
        st.info(f"本次对话耗时: {elapse}s")
    st.session_state.messages.append(
        {"role":"assistance","content":response}
    )
    st.session_state.elapse.append(f"本次对话耗时: {elapse}s")
# from streamlit_chatbox import ChatBox

# chat_box = ChatBox(chat_name="gptcache",
#                    session_key="chat_history",
#                    assistant_avatar="icon.png")
# if not chat_box.chat_inited:
#     st.toast(f"欢迎试用gptcache demo，当前使用{MODEL_NAME}为后端LLM")
#     chat_box.init_session()
# # Display chat messages from history on app rerun
# chat_box.output_messages()

# chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "

# if prompt := st.chat_input(chat_input_placeholder,key="prompt"):
#     chat_box.user_say(prompt)
#     chat_box.ai_say("正在思考，稍安勿躁...")
#     text = ""
#     elapse,response = query_with_gpt_cache(question=prompt,temperature=temperature)

#     chat_box.update_msg(text,streaming=True)



# question = st.text_input(label=":orange[请输入文本内容]",placeholder="你是谁？")
# temperature = st.slider(label=":orange[请选择输出参数temperature]",min_value=0.0,max_value=2.0)

# elapse, answer_str = query_with_gpt_cache(question=question,temperature=temperature)

# st.info(f"本次查询耗时：{elapse}")
# st.info(f"答案：{answer_str}")


