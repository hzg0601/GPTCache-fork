import time

import numpy as np

from gptcache import cache
from gptcache.processor.post import temperature_softmax
from gptcache.utils.error import NotInitError
from gptcache.utils.log import gptcache_log
from gptcache.utils.time import time_cal


def adapt(llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs):
    """Adapt to different llm

    :param llm_handler: LLM calling method, when the cache misses, this function will be called
    :param cache_data_convert: When the cache hits, convert the answer in the cache to the format of the result returned by llm
    :param update_cache_callback: If the cache misses, after getting the result returned by llm, save the result to the cache
    :param args: llm args
    :param kwargs: llm kwargs
    :return: llm result
    """
    start_time = time.time()
    search_only_flag = kwargs.pop("search_only", False) #  如果未设置cache_enable,或设定cache_skip，且未设置next_cache，search_only_flag为True时，直接返回None
    user_temperature = "temperature" in kwargs
    user_top_k = "top_k" in kwargs
    temperature = kwargs.pop("temperature", 0.0)
    chat_cache = kwargs.pop("cache_obj", cache) #* cache对象，即core.py的Cache类
    session = kwargs.pop("session", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."
    if not chat_cache.has_init:
        raise NotInitError()
    # # 设置了cache_enable和cache_skip两个参数，前者有chat_cache的cache_enable_func决定
    # 如果不设cache_enable则不进行embedding，且不进行语义搜索，
    # 如果设置cache_skip则进行embedding，但不进行语义搜索
    
    # 如果cache_enable为False,且cache_next为None，则直接基于大模型返回但不保存大模型的返回结果
    # 如果cache_enable为True,skip_cache为True,则进行embedding，但不执行语义搜索，直接返回大模型的结果，并保存结果；
    # 如果cache_enable为True,skip_cache为False,则进行embedding,执行语义搜索，基于语义搜索返回sql数据库的结果，如果未检索到答案，则仍基于大模型返回，并保存结果
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs) 
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    # you want to retry to send the request to chatgpt when the cache is negative
    # 如果temperature \in （0,2),则根据temperature的采样结果决定cache_skip是否为True
    # 若大于2，则cache_skip=True,若小于0,则cache_skip=False
    # 
    if 0 < temperature < 2:
        cache_skip_options = [True, False]
        prob_cache_skip = [0, 1]
        cache_skip = kwargs.pop(
            "cache_skip",
            temperature_softmax(
                messages=cache_skip_options,
                scores=prob_cache_skip,
                temperature=temperature,
            ),
        )
    elif temperature >= 2:
        cache_skip = kwargs.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = kwargs.pop("cache_skip", False)
    # 默认cache_factor为1
    # time_cal记录函数调用时间的装饰器
    cache_factor = kwargs.pop("cache_factor", 1.0)
    pre_embedding_res = time_cal(
        chat_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=chat_cache.report.pre,
    )(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
        cache_config=chat_cache.config,
    )
    # 如果pre_embedding_func的执行结果是一个tuple,
    # 则tuple的元素分别是pre_store_data和pre_embedding_data
    # 否则该返回值即是pre_store_data,也是pre_embedding_data
    # pre_embedding_data即查询的question
    # 如果设置了input_summary_len，调用_summarize_input对输入进行改写
    # 该函数会调用Huggingface_hub的facebook/bart-large-cn模型（1.63G）
    # 
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if chat_cache.config.input_summary_len is not None:
        pre_embedding_data = _summarize_input(
            pre_embedding_data, chat_cache.config.input_summary_len
        )
    # 针对pre_embedding_data执行embedding_fun，即对存入embedding库的question进行嵌入
    # 如果cache_skip不为True,则调用data_manager.search方法
    # 基于embedding_data在向量数据库中查找top_k个向量的id和distance
    # 并调用report.search方法记录耗时和操作数
    #
    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    # 如果设置cache_enable,且不设定cache_skip,则执行cache 保存，及基于向量搜索返回结果
    if cache_enable and not cache_skip:
        search_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", 5)
            if (user_temperature and not user_top_k)
            else kwargs.pop("top_k", -1),
        )
        if search_data_list is None:
            search_data_list = []
        cache_answers = []
        # 根据similairty的range,threhold,cache_factor计算rank_threhold
        # 并仍限制其在range内
        similarity_threshold = chat_cache.config.similarity_threshold
        min_rank, max_rank = chat_cache.similarity_evaluation.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor
        rank_threshold = (
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        # 针对embedding中召唤的数据，根据其id，调用get_data_by_id
        # 根据question table中的id查找answer,deps,session_ids，
        # 然后组装成CacheData返回，记录取数据的时间和操作次数
        # 
        for search_data in search_data_list:
            cache_data = time_cal(
                chat_cache.data_manager.get_scalar_data,
                func_name="get_data",
                report_func=chat_cache.report.data,
            )(
                search_data,
                extra_param=context.get("get_scalar_data", None),
                session=session,
            )
            if cache_data is None:
                continue

            # cache consistency check
            # 如果定义了检查函数，则检查数据是否一致
            if chat_cache.config.data_check:
                is_healthy = cache_health_check(
                    chat_cache.data_manager.v,
                    {
                        "embedding": cache_data.embedding_data,
                        "search_result": search_data,
                    },
                )
                if not is_healthy:
                    continue
            # 如果context和question都有deps属性或字段
            # 则返回eval_query_data的question为deps[0][data],embedding为None
            # 否则eval_query_data的question即pre_store_data, embedding为召回的embedding_data
            #
            # 可以通过eval_cache_data从向量数据库的匹配得分
            # 以及similarity_evaluation的max_distance,positive重新计算distance
            # 
            if "deps" in context and hasattr(cache_data.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None,
                }
                eval_cache_data = {
                    "question": cache_data.question.deps[0].data,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": cache_data.question,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": cache_data.embedding_data,
                }
            rank = time_cal(
                chat_cache.similarity_evaluation.evaluation,
                func_name="evaluation",
                report_func=chat_cache.report.evaluation,
            )(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            gptcache_log.debug(
                "similarity: [user question] %s, [cache question] %s, [value] %f",
                pre_store_data,
                cache_data.question,
                rank,
            )
            # 如果重新计算的rank_threhold <= 重新计算的distance,即相似度高于阈值
            # 则缓存相似度，答案，向量搜索结果，sql返回的session结果至cache_answers中
    
            if rank_threshold <= rank:
                cache_answers.append(
                    (float(rank), cache_data.answers[0].answer, search_data, cache_data)
                )
                chat_cache.data_manager.hit_cache_callback(search_data)
        # 对cache_answer按rank进行排序
        # 将answer和全部数据作为一个answers_dict
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d) for d in cache_answers)
        if len(cache_answers) != 0:
            hit_callback = kwargs.pop("hit_callback", None)
            if hit_callback and callable(hit_callback):
                factor = max_rank - min_rank
                hit_callback([(d[3].question, d[0] / factor if factor else d[0]) for d in cache_answers])
            # 如果定义的post_process_messages_func是temperature_softmax
            # 则根据temperature，score进行采样返回答案
            # 否则将答案取出进行处理，返回答案
            def post_process():
                if chat_cache.post_process_messages_func is temperature_softmax:
                    return_message = chat_cache.post_process_messages_func(
                        messages=[t[1] for t in cache_answers],
                        scores=[t[0] for t in cache_answers],
                        temperature=temperature,
                    )
                else:
                    return_message = chat_cache.post_process_messages_func(
                        [t[1] for t in cache_answers]
                    )
                return return_message

            return_message = time_cal(
                post_process,
                func_name="post_process",
                report_func=chat_cache.report.post,
            )()
            # 针对采样的答案，从answer_dict中取出其对应的全部数据
            # 如果定义了session,调用data_manager.add_session，
            # 将embeeding向量，sesssion_name, embedding的问题存入session表中
            chat_cache.report.hint_cache()
            cache_whole_data = answers_dict.get(str(return_message))
            if session and cache_whole_data:
                chat_cache.data_manager.add_session(
                    cache_whole_data[2], session.name, pre_embedding_data
                )
            # 将user_question,cache_question,cache_question_id,cache_answer,smiliar_value,time存入report表中
            # 调用cache_data_convert对return_message进行组装，然后返回前端
            if cache_whole_data:
                # user_question / cache_question / cache_question_id / cache_answer / similarity / consume time/ time
                report_cache_data = cache_whole_data[3] # cache_data
                report_search_data = cache_whole_data[2] # search_data
                chat_cache.data_manager.report_cache(
                    pre_store_data if isinstance(pre_store_data, str) else "",
                    report_cache_data.question
                    if isinstance(report_cache_data.question, str)
                    else "",
                    report_search_data[1],
                    report_cache_data.answers[0].answer
                    if isinstance(report_cache_data.answers[0].answer, str)
                    else "",
                    cache_whole_data[0],
                    round(time.time() - start_time, 6),
                )
            return cache_data_convert(return_message)
    # 如果cache_skip为True，则考虑next_cache,
    # 如果next_cache不为None，将定义的关键字存储，执行递归调用
    # 如果next_cache为None,则直接调用大模型
    next_cache = chat_cache.next_cache
    # 如果未设置cache_enable,或设定cache_skip,则考虑chat_cache是否设置了next_cache
    # 如果设置了next_cache，则保存所有参数，仍执行执行cache 保存，及基于向量搜索返回结果
    # 如果未设置next_cache，则考虑search_only_flag，若设置了search_only_flag,则返回None
    # 若未设置search_only_flag，则直接调用大模型
    if next_cache: 
        kwargs["cache_obj"] = next_cache
        kwargs["cache_context"] = context
        kwargs["cache_skip"] = cache_skip
        kwargs["cache_factor"] = cache_factor
        kwargs["search_only"] = search_only_flag
        llm_data = adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )
    else:
        if search_only_flag: # 
            # cache miss
            return None
        llm_data = time_cal(
            llm_handler, func_name="llm_request", report_func=chat_cache.report.llm
        )(*args, **kwargs)

    if not llm_data:
        return None
    # 如果直接调用大模型返回的结果不为None,且cache_enable不为None、False
    # 则调用data_manager.save保存大模型的返回结果，并最终返回大模型的结果
    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                time_cal(
                    chat_cache.data_manager.save,
                    func_name="save",
                    report_func=chat_cache.report.save,
                )(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                )
                if (
                    chat_cache.report.op_save.count > 0
                    and chat_cache.report.op_save.count % chat_cache.config.auto_flush
                    == 0
                ):
                    chat_cache.flush()

            llm_data = update_cache_callback(
                llm_data, update_cache_func, *args, **kwargs
            )
        except Exception as e:  # pylint: disable=W0703
            gptcache_log.warning("failed to save the data to cache, error: %s", e)
    return llm_data


async def aadapt(
    llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
):
    """Simple copy of the 'adapt' method to different llm for 'async llm function'

    :param llm_handler: Async LLM calling method, when the cache misses, this function will be called
    :param cache_data_convert: When the cache hits, convert the answer in the cache to the format of the result returned by llm
    :param update_cache_callback: If the cache misses, after getting the result returned by llm, save the result to the cache
    :param args: llm args
    :param kwargs: llm kwargs
    :return: llm result
    """
    start_time = time.time()
    user_temperature = "temperature" in kwargs
    user_top_k = "top_k" in kwargs
    temperature = kwargs.pop("temperature", 0.0)
    chat_cache = kwargs.pop("cache_obj", cache)
    session = kwargs.pop("session", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."
    if not chat_cache.has_init:
        raise NotInitError()
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    # you want to retry to send the request to chatgpt when the cache is negative

    if 0 < temperature < 2:
        cache_skip_options = [True, False]
        prob_cache_skip = [0, 1]
        cache_skip = kwargs.pop(
            "cache_skip",
            temperature_softmax(
                messages=cache_skip_options,
                scores=prob_cache_skip,
                temperature=temperature,
            ),
        )
    elif temperature >= 2:
        cache_skip = kwargs.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = kwargs.pop("cache_skip", False)
    cache_factor = kwargs.pop("cache_factor", 1.0)
    pre_embedding_res = time_cal(
        chat_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=chat_cache.report.pre,
    )(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
        cache_config=chat_cache.config,
    )
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if chat_cache.config.input_summary_len is not None:
        pre_embedding_data = _summarize_input(
            pre_embedding_data, chat_cache.config.input_summary_len
        )

    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if cache_enable and not cache_skip:
        search_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", 5)
            if (user_temperature and not user_top_k)
            else kwargs.pop("top_k", -1),
        )
        if search_data_list is None:
            search_data_list = []
        cache_answers = []
        similarity_threshold = chat_cache.config.similarity_threshold
        min_rank, max_rank = chat_cache.similarity_evaluation.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor
        rank_threshold = (
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        for search_data in search_data_list:
            cache_data = time_cal(
                chat_cache.data_manager.get_scalar_data,
                func_name="get_data",
                report_func=chat_cache.report.data,
            )(
                search_data,
                extra_param=context.get("get_scalar_data", None),
                session=session,
            )
            if cache_data is None:
                continue

            if "deps" in context and hasattr(cache_data.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None,
                }
                eval_cache_data = {
                    "question": cache_data.question.deps[0].data,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": cache_data.question,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": cache_data.embedding_data,
                }
            rank = time_cal(
                chat_cache.similarity_evaluation.evaluation,
                func_name="evaluation",
                report_func=chat_cache.report.evaluation,
            )(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            gptcache_log.debug(
                "similarity: [user question] %s, [cache question] %s, [value] %f",
                pre_store_data,
                cache_data.question,
                rank,
            )
            if rank_threshold <= rank:
                cache_answers.append(
                    (float(rank), cache_data.answers[0].answer, search_data, cache_data)
                )
                chat_cache.data_manager.hit_cache_callback(search_data)
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d) for d in cache_answers)
        if len(cache_answers) != 0:
            def post_process():
                if chat_cache.post_process_messages_func is temperature_softmax:
                    return_message = chat_cache.post_process_messages_func(
                        messages=[t[1] for t in cache_answers],
                        scores=[t[0] for t in cache_answers],
                        temperature=temperature,
                    )
                else:
                    return_message = chat_cache.post_process_messages_func(
                        [t[1] for t in cache_answers]
                    )
                return return_message

            return_message = time_cal(
                post_process,
                func_name="post_process",
                report_func=chat_cache.report.post,
            )()
            chat_cache.report.hint_cache()
            cache_whole_data = answers_dict.get(str(return_message))
            if session and cache_whole_data:
                chat_cache.data_manager.add_session(
                    cache_whole_data[2], session.name, pre_embedding_data
                )
            if cache_whole_data:
                # user_question / cache_question / cache_question_id / cache_answer / similarity / consume time/ time
                report_cache_data = cache_whole_data[3]
                report_search_data = cache_whole_data[2]
                chat_cache.data_manager.report_cache(
                    pre_store_data if isinstance(pre_store_data, str) else "",
                    report_cache_data.question
                    if isinstance(report_cache_data.question, str)
                    else "",
                    report_search_data[1],
                    report_cache_data.answers[0].answer
                    if isinstance(report_cache_data.answers[0].answer, str)
                    else "",
                    cache_whole_data[0],
                    round(time.time() - start_time, 6),
                )
            return cache_data_convert(return_message)

    next_cache = chat_cache.next_cache
    if next_cache:
        kwargs["cache_obj"] = next_cache
        kwargs["cache_context"] = context
        kwargs["cache_skip"] = cache_skip
        kwargs["cache_factor"] = cache_factor
        llm_data = adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )
    else:
        llm_data = await llm_handler(*args, **kwargs)

    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                time_cal(
                    chat_cache.data_manager.save,
                    func_name="save",
                    report_func=chat_cache.report.save,
                )(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                )
                if (
                    chat_cache.report.op_save.count > 0
                    and chat_cache.report.op_save.count % chat_cache.config.auto_flush
                    == 0
                ):
                    chat_cache.flush()
            llm_data = update_cache_callback(
                llm_data, update_cache_func, *args, **kwargs
            )
        except Exception:  # pylint: disable=W0703
            gptcache_log.error("failed to save the data to cache", exc_info=True)
    return llm_data


_input_summarizer = None


def _summarize_input(text, text_length):
    if len(text) <= text_length:
        return text

    # pylint: disable=import-outside-toplevel
    from gptcache.processor.context.summarization_context import (
        SummarizationContextProcess,
    )

    global _input_summarizer
    if _input_summarizer is None:
        _input_summarizer = SummarizationContextProcess()
    summarization = _input_summarizer.summarize_to_sentence([text], text_length)
    return summarization


def cache_health_check(vectordb, cache_dict):
    """This function checks if the embedding
    from vector store matches one in cache store.
    If cache store and vector store are out of
    sync with each other, cache retrieval can
    be incorrect.
    If this happens, force the similary score
    to the lowerest possible value.
    """
    emb_in_cache = cache_dict["embedding"]
    _, data_id = cache_dict["search_result"]
    emb_in_vec = vectordb.get_embeddings(data_id)
    flag = np.all(emb_in_cache == emb_in_vec)
    if not flag:
        gptcache_log.critical("Cache Store and Vector Store are out of sync!!!")
        # 0: identical, inf: different
        cache_dict["search_result"] = (
            np.inf,
            data_id,
        )
        # self-healing by replacing entry
        # in the vec store with the one
        # from cache store by the same
        # entry_id.
        vectordb.update_embeddings(
            data_id,
            emb=cache_dict["embedding"],
        )
    return flag
