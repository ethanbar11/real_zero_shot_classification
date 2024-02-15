# import json
#
# class CompactJSONEncoder(json.JSONEncoder):
#     """A JSON Encoder that puts small lists on single lines."""
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.indentation_level = 0
#         self.max_length_in_one_line = 50
#
#     def encode(self, o):
#         """Encode JSON object *o* with respect to single line lists."""
#
#         if isinstance(o, (list, tuple)):
#             if self._is_single_line_list(o):
#                 return "[" + ", ".join(json.dumps(el) for el in o) + "]"
#             else:
#                 self.indentation_level += 1
#                 output = [self.indent_str + self.encode(el) for el in o]
#                 self.indentation_level -= 1
#                 return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
#
#         elif isinstance(o, dict):
#             self.indentation_level += 1
#             output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
#             self.indentation_level -= 1
#             return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
#
#         else:
#             return json.dumps(o)
#
#     def _is_single_line_list(self, o):
#         if isinstance(o, (list, tuple)):
#             return not any(isinstance(el, (list, tuple, dict)) for el in o) \
#                 and len(o) <= self.max_length_in_one_line \
#                 and len(str(o)) - 2 <= 60
#
#     @property
#     def indent_str(self) -> str:
#         return " " * self.indentation_level * self.indent
#
#     def iterencode(self, o, **kwargs):
#         """Required to also work with `json.dump`."""
#         return self.encode(o)
#
#
import json
from typing import Union


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 10000 #70
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 100 #2
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            return self._encode_list(o)
        if isinstance(o, dict):
            return self._encode_object(o)
        if isinstance(o, float):  # Use scientific notation for floats
            return format(o, "g")
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_list(self, o):
        if self._put_on_single_line(o):
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    def _encode_object(self, o):
        if not o:
            return "{}"
        if self._put_on_single_line(o):
            return (
                "{ "
                + ", ".join(
                    f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()
                )
                + " }"
            )
        self.indentation_level += 1
        output = [
            f"{self.indent_str}{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()
        ]

        self.indentation_level -= 1
        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        return (
            self._primitives_only(o)
            and len(o) <= self.MAX_ITEMS
            and len(str(o)) - 2 <= self.MAX_WIDTH
        )

    def _primitives_only(self, o: Union[list,tuple,dict]):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )
