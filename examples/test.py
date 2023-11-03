import re
print(type(key),type(keywords),type(json))
JSON_MAP = {'层数': "floor", "名称": "name"}
JOSN_MAP_REV = {value_: key_ for key_, value_ in JSON_MAP.items()}


# def test(keywords, key, json):
if isinstance(key, str):
    key = [key]
keywords = {key_: value_ 
            for key_, value_ in keywords.items()
            if key_ in key}
keywords = {key_: (re.split("[，、]", value_) if
                    re.search("[、，]", value_) else value_)
                    for key_, value_ in keywords.items()
                    }
if len(keywords) == 0:
    return json
if JOSN_MAP_REV['floor'] in keywords:
    json['data']['floor'] = keywords.pop(JOSN_MAP_REV['floor'])
    JOSN_MAP_REV.pop("floor")
if len(keywords) > 0:
    json['data']['object'] = [{JSON_MAP[key_]: value_ for key_, value_ 
                                        in keywords.items()}]
    
return json