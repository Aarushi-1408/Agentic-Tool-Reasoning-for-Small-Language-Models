from typing import Dict, Any, List, Tuple
import jsonschema


def validate_arguments(parameters_schema: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate tool arguments against the JSON Schema defined in parameters_schema.
    Returns (is_valid, error_message).
    
    The harness performs validation purely against schema semantics and does NOT
    silently fix or alter the model's argument types/decisions.
    """
    if not isinstance(arguments, dict):
        return False, f"Arguments must be a dictionary/object, received {type(arguments).__name__}."

    # Default schema wrapper if missing top-level type
    schema = dict(parameters_schema)
    if "type" not in schema:
        schema["type"] = "object"

    try:
        jsonschema.validate(instance=arguments, schema=schema)
        return True, ""
    except jsonschema.exceptions.ValidationError as err:
        # Return exact validation failure reason for structured observation
        return False, f"Invalid arguments according to tool schema: {err.message}"
    except Exception as err:
        return False, f"Schema validation error: {str(err)}"


def generate_tool_schema(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate standard JSON representation for a tool schema.
    """
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }
