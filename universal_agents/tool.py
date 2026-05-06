def tool(description="", requires_confirmation=False, **params):
    def decorator(func):
        func._is_tool = True
        func._tool_name = func.__name__
        func._requires_confirmation = requires_confirmation

        properties = {}
        required = []
        for pname, (ptype, pdesc) in params.items():
            properties[pname] = {"type": ptype, "description": pdesc}
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