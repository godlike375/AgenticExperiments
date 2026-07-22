ENVIRONMENT_PREFIX = '[[SYS ENV]]'


def tool(description="", short_description="", requires_confirmation=False, **params):
    def decorator(func):
        func._is_tool = True
        func._tool_name = func.__name__
        func._requires_confirmation = requires_confirmation
        func._short_description = short_description

        # Маппинг имён типов Python в стандартные JSON Schema-типы,
        _PY_TO_JSON_TYPE = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }

        properties = {}
        required = []
        for pname, (ptype, pdesc) in params.items():
            json_type = _PY_TO_JSON_TYPE.get(ptype, ptype)
            properties[pname] = {"type": json_type, "description": pdesc}
            if not pdesc.lower().startswith("optional"):
                required.append(pname)

        func._tool_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description or (func.__doc__ or "").split("\n")[0].strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        return func
    return decorator