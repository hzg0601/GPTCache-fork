from datetime import datetime
from typing import List, Optional, Dict

import numpy as np

from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    Question,
    QuestionDep,
)
from gptcache.utils import import_sqlalchemy

import_sqlalchemy()

# pylint: disable=C0413
import sqlalchemy
from sqlalchemy import func, create_engine, Column, Sequence
from sqlalchemy.types import (
    String,
    DateTime,
    LargeBinary,
    Integer,
    Float,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DEFAULT_LEN_DOCT = {
    "question_question": 3000,
    "answer_answer": 3000,
    "session_id": 1000,
    "dep_name": 1000,
    "dep_data": 3000,
}


def _get_table_len(config: Dict, column_alias: str) -> int:
    if config and column_alias in config and config[column_alias] > 0:
        return config[column_alias]
    return DEFAULT_LEN_DOCT.get(column_alias, 1000)


def get_models(table_prefix, db_type, table_len_config):
    DynamicBase = declarative_base(class_registry={})  # pylint: disable=C0103
    """ 
    建立了answer,question,question_report,session,report四张表

    """

    class QuestionTable(DynamicBase):
        """
        question table，建立了
        create_on,lass_access,embedding_data，deleted,question等四字段

        """

        __tablename__ = table_prefix + "_question"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            question_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(Integer, question_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=False,
        )
        create_on = Column(DateTime, default=datetime.now)
        last_access = Column(DateTime, default=datetime.now)
        embedding_data = Column(LargeBinary, nullable=True)
        deleted = Column(Integer, default=0)

    class AnswerTable(DynamicBase):
        """
        answer table建立了id,question_id,answer,aswer_type字段
        """

        __tablename__ = table_prefix + "_answer"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            answer_id_seq = Sequence(f"{__tablename__}_id_seq")
            id = Column(Integer, answer_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        answer = Column(
            String(_get_table_len(table_len_config, "answer_answer")), nullable=False
        )
        answer_type = Column(Integer, nullable=False)

    class SessionTable(DynamicBase):
        """
        session table建立了id,question_id,session_id,session_question字段
        session每次会话都记录？
        """

        __tablename__ = table_prefix + "_session"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            session_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(
                Integer,
                session_id_seq,
                primary_key=True,
                autoincrement=True,
            )
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        session_id = Column(
            String(_get_table_len(table_len_config, "session_id")), nullable=False
        )
        session_question = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=False,
        )

    class QuestionDepTable(DynamicBase):
        """
        answer table建立了id,qestion_id,dep_name,dep_data,dep_type
        dep?
        """

        __tablename__ = table_prefix + "_question_dep"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            question_dep_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(
                Integer, question_dep_id_seq, primary_key=True, autoincrement=True
            )
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        dep_name = Column(
            String(_get_table_len(table_len_config, "dep_name")), nullable=False
        )
        dep_data = Column(
            String(_get_table_len(table_len_config, "dep_data")), nullable=False
        )
        dep_type = Column(Integer, nullable=False)

    class ReportTable(DynamicBase):
        """
        report table建立了id,user_question,cache_question_id,cache_question_id,cache_answer,
        similarity,cache_delta_time,cache_time,extra
        """

        __tablename__ = table_prefix + "_report"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            question_dep_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(
                Integer, question_dep_id_seq, primary_key=True, autoincrement=True
            )
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)

        user_question = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=False,
        )
        cache_question_id = Column(
            Integer,
            nullable=False,
        )
        cache_question = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=False,
        )
        cache_answer = Column(
            String(_get_table_len(table_len_config, "answer_answer")), nullable=False
        )
        similarity = Column(Float, nullable=False)
        cache_delta_time = Column(Float, nullable=False)
        cache_time = Column(DateTime, default=datetime.now)
        extra = Column(
            String(_get_table_len(table_len_config, "question_question")),
            nullable=True,
        )

    return QuestionTable, AnswerTable, QuestionDepTable, SessionTable, ReportTable


class SQLStorage(CacheStorage):
    """
    Using sqlalchemy to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.

    :param name: the name of the cache storage, it is support 'sqlite', 'postgresql', 'mysql', 'mariadb', 'sqlserver' and  'oracle' now.
    :type name: str
    :param sql_url: the url of the sql database for cache, such as '<db_type>+<db_driver>://<username>:<password>@<host>:<port>/<database>',
                    and the default value is related to the `cache_store` parameter,
                    'sqlite:///./sqlite.db' for 'sqlite',
                    'duckdb:///./duck.db' for 'duckdb',
                    'postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres' for 'postgresql',
                    'mysql+pymysql://root:123456@127.0.0.1:3306/mysql' for 'mysql',
                    'mariadb+pymysql://root:123456@127.0.0.1:3307/mysql' for 'mariadb',
                    'mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server' for 'sqlserver',
                    'oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8' for 'oracle'.
    :type sql_url: str
    :param table_name: the table name for sql database, defaults to 'gptcache'.
    :type table_name: str
    """

    def __init__(
        self,
        db_type: str = "sqlite",
        url: str = "sqlite:///./sqlite.db",
        table_name: str = "gptcache",
        table_len_config=None,
    ):
        if table_len_config is None:
            table_len_config = {}
        self._url = url
        # 返回五张表的类，每个类规定了表的字段
        self._ques, self._answer, self._ques_dep, self._session, self._report = get_models(
            table_name, db_type, table_len_config
        )
        self._engine = create_engine(self._url)
        # sessionmarker用于构建session对象
        self.Session = sessionmaker(bind=self._engine)  # pylint: disable=invalid-name
        self.create()
    # 建表
    def create(self):
        self._ques.__table__.create(bind=self._engine, checkfirst=True)
        self._answer.__table__.create(bind=self._engine, checkfirst=True)
        self._ques_dep.__table__.create(bind=self._engine, checkfirst=True)
        self._session.__table__.create(bind=self._engine, checkfirst=True)
        self._report.__table__.create(bind=self._engine, checkfirst=True)

    def _insert(self, data: CacheData, session: sqlalchemy.orm.Session) -> Column:
        # CacheData包含question,answer,embedding_data,session_id,create_on,last_access
        # session为sql的session
        # ORM (对象关系映射)，ORM 将数据库中的表与面向对象语言中的类建立了一种对应关系。
        # 若要操作数据库，数据库中的表或者表中的一条记录就可以直接通过操作类或者类实例来完成。
        # ques_data 为sql里qeustion_table的类
        ques_data = self._ques(
            question=data.question
            if isinstance(data.question, str)
            else data.question.content,
            embedding_data=data.embedding_data.tobytes()
            if data.embedding_data is not None
            else None,
        )
        # 在sql的session里添加对象
        #* deps是question的属性，question包含了deps和content两种属性
        #* dep又包含了name,data,dep_type三种属性
        session.add(ques_data)
        session.flush()
        if isinstance(data.question, Question) and data.question.deps is not None:
            all_deps = []
            for dep in data.question.deps:
                all_deps.append(
                    self._ques_dep(
                        question_id=ques_data.id,
                        dep_name=dep.name,
                        dep_data=dep.data,
                        dep_type=dep.dep_type,
                    )
                )
            session.add_all(all_deps)
        answers = data.answers if isinstance(data.answers, list) else [data.answers]
        all_data = []
        for answer in answers:
            answer_data = self._answer(
                question_id=ques_data.id,
                answer=answer.answer,
                answer_type=int(answer.answer_type),
            )
            all_data.append(answer_data)
        session.add_all(all_data)
        if data.session_id:
            session_data = self._session(
                question_id=ques_data.id,
                session_id=data.session_id,
                session_question=data.question
                if isinstance(data.question, str)
                else data.question.content,
            )
            session.add(session_data)
        return ques_data.id
    #  定义批量提交表格至数据库的函数
    def batch_insert(self, all_data: List[CacheData]):
       
        ids = []
        with self.Session() as session:
            for data in all_data:
                ids.append(self._insert(data, session))
           
            session.commit()
        return ids
    
    # 根据question table中的id查找qeustion
    # 并将last_access更新至当前时间
    # 根据question table中的id查找answer,deps,session_ids，
    # 然后组装成CacheData返回
    def get_data_by_id(self, key: int) -> Optional[CacheData]:
        with self.Session() as session:
            qs = (
                session.query(self._ques)
                .filter(self._ques.id == key)
                .filter(self._ques.deleted == 0)
                .first()
            )
            if qs is None:
                return None
            last_access = qs.last_access
            qs.last_access = datetime.now()
            ans = (
                session.query(self._answer.answer, self._answer.answer_type)
                .filter(self._answer.question_id == qs.id)
                .all()
            )
            deps = (
                session.query(
                    self._ques_dep.dep_name,
                    self._ques_dep.dep_data,
                    self._ques_dep.dep_type,
                )
                .filter(self._ques_dep.question_id == qs.id)
                .all()
            )
            session_ids = (
                session.query(self._session.session_id)
                .filter(self._session.question_id == qs.id)
                .all()
            )
            res_ans = [(item.answer, item.answer_type) for item in ans]
            res_deps = [
                QuestionDep(item.dep_name, item.dep_data, item.dep_type)
                for item in deps
            ]
            session.commit()

            return CacheData(
                question=qs.question if not deps else Question(qs.question, res_deps),
                answers=res_ans,
                embedding_data=np.frombuffer(qs.embedding_data, dtype=np.float32),
                session_id=session_ids,
                create_on=qs.create_on,
                last_access=last_access,
            )
    # 获取所有question的id
    def get_ids(self, deleted=True):
        state = -1 if deleted else 0
        with self.Session() as session:
            res = session.query(self._ques.id).filter(self._ques.deleted == state).all()
            return [item.id for item in res]
    # 将已删除问题的状态标记为-1
    def mark_deleted(self, keys):
        with self.Session() as session:
            session.query(self._ques).filter(self._ques.id.in_(keys)).update(
                {"deleted": -1}
            )
            session.commit()
    # 清空已删除数据
    def clear_deleted_data(self):
        with self.Session() as session:
            objs = session.query(self._ques).filter(self._ques.deleted == -1)
            q_ids = [obj.id for obj in objs]
            session.query(self._answer).filter(
                self._answer.question_id.in_(q_ids)
            ).delete()
            session.query(self._ques_dep).filter(
                self._ques_dep.question_id.in_(q_ids)
            ).delete()
            session.query(self._session).filter(
                self._session.question_id.in_(q_ids)
            ).delete()
            objs.delete()
            session.commit()
    # 记录question的数量
    def count(self, state: int = 0, is_all: bool = False):
        with self.Session() as session:
            if is_all:
                return session.query(func.count(self._ques.id)).scalar()
            return (
                session.query(func.count(self._ques.id))
                .filter(self._ques.deleted == state)
                .scalar()
            )
    # 增加session数据，向session表中写入数据
    def add_session(self, question_id, session_id, session_question):
        with self.Session() as session:
            session_data = self._session(
                question_id=question_id,
                session_id=session_id,
                session_question=session_question,
            )
            session.add(session_data)
            session.commit()
    # 根据session表的id删除session数据
    def delete_session(self, keys):
        with self.Session() as session:
            session.query(self._session).filter(self._session.id.in_(keys)).delete()
            session.commit()
    # 根据question_id或session_id取出session中的数据
    def list_sessions(self, session_id=None, key=None):
        with self.Session() as session:
            query = session.query(self._session)
            if session_id:
                query = query.filter(self._session.session_id == session_id)
            elif key:
                query = query.filter(self._session.question_id == key)
            return query.all()
    # 写入report数据，report 
    # 数据包含用户的新问题,已缓存的问题，已缓存问题的id,缓存的answer,相似度值，缓存的时间差
    # 向report
    #* 用来向前端报告的数据？
    def report_cache(self, user_question, cache_question, cache_question_id, cache_answer, similarity_value, cache_delta_time):
        with self.Session() as session:
            report_data = self._report(
                user_question=user_question,
                cache_question=cache_question,
                cache_question_id=cache_question_id,
                cache_answer=cache_answer,
                similarity=similarity_value,
                cache_delta_time=cache_delta_time,
            )
            session.add(report_data)
            session.commit()

    def close(self):
        pass

    def count_answers(self):
        # for UT
        with self.Session() as session:
            return session.query(func.count(self._answer.id)).scalar()
