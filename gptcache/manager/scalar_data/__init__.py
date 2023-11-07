__all__ = ["CacheBase"]

from gptcache.utils.lazy_import import LazyImport

scalar_manager = LazyImport(
    "scalar_manager", globals(), "gptcache.manager.scalar_data.manager"
)

# 
# 下游脚本调用时先进入__init__中，调用本函数
# 本函数调用gptcache.manager.scalar_data.manager.CacheBase.get函数
# .get函数基于name等args返回对应的*Storage类
def CacheBase(name: str, **kwargs):
    """Generate specific CacheStorage with the configuration. For example, setting for
       `SQLDataBase` (with `name`, `sql_url` and `table_name` params) to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.
    """
    return scalar_manager.CacheBase.get(name, **kwargs)
