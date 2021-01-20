import numpy as np
from collections import namedtuple
from typing import Callable, List, Dict, Optional, Union
from codelets.adl.backups.operand import Operand, NullOperand
from codelets.adl.flex_param import FlexParam
from itertools import count
from dataclasses import dataclass, asdict, field
from types import FunctionType
import sys
import inspect
import traceback
field_cnt = count()
INSTR_FN_NAME_TEMPLATE = """param_fn{FN_ID}"""
INSTR_FN_TEMPLATE = """def param_fn{FN_ID}(program, hag, relocation_table, cdlt, op{ITER_ARGS}):\n\treturn {FN_BODY}"""
FIELD_HEADER = """<%def field_name="{field_name}(hag, op, cdlt, relocation_table, program, iter_arg=None, cond_arg=None)">"""
FIELD_END = "</%def>"
FIELD_TEMPLATE = """
<%def field_name="{field_name}(cdlt, op, output_type, iter_arg=None, cond_arg=None)">
    % if 
</%def>
"""

@dataclass
class Field:
    field_name: str
    bitwidth: int
    field_id: int = field(default_factory=lambda: next(field_cnt))
    value: int = field(default=None)
    value_names: Dict[str, int] = field(default_factory=dict)
    value_str: str = field(default=None)
    param_fn: FlexParam = field(default=None, init=False)


    @property
    def isset(self) -> bool:
        return self.value is not None

    @property
    def value_name_list(self) -> List[str]:
        return list(self.value_names.keys())

    def set_param_fn(self, fn: FlexParam):
        assert isinstance(fn, FlexParam)
        self.param_fn = fn

    def set_value(self, value):
        if self.value is not None:
            raise ValueError(f"{self.field_name} has already been set to {self.value}")
        self.value = value

    def set_value_by_string(self, value_name) -> None:
        if value_name not in self.value_names:
            raise ValueError(f"{value_name} is not mapped to any values for this field.\n"
                             f"Possible values: {self.value_name_list}")
        elif self.value is not None and self.value_str != value_name:
            raise ValueError(f"Value for {self.field_name} has already been set to {self.value_str}, cannot set to {value_name}")
        else:
            self.value_str = value_name
            self.value = self.value_names[value_name]

    def get_string_value(self) -> Union[str]:
        if self.value_str is not None:
            return self.value_str
        return str(self.value)

    # def get_param_fnc(self, **iter_args):
    #     if len(iter_args) > 0:
    #         iter_arg_str = ", " + ", ".join(list(iter_args.keys()))
    #     else:
    #         iter_arg_str = ""
    #
    #     param_fnc_str = INSTR_FN_TEMPLATE.format(FN_ID=self.field_id, FN_BODY=self.param_fn, ITER_ARGS=iter_arg_str)
    #     param_fnc_code = compile(param_fnc_str, "<string>", "exec")
    #     param_fnc = FunctionType(param_fnc_code.co_consts[0], globals(), name=INSTR_FN_NAME_TEMPLATE.format(FN_ID=self.field_id))
    #     return param_fnc, param_fnc_str

    # def run_param_fnc(self, *args, **iter_args):
    #     param_fnc, param_func_str = self.get_param_fnc(**iter_args)
    #     # TODO: Important--> this assumes that iter_args are iterated over in the correct order
    #     kwargs_as_args = tuple([v for _, v in iter_args.items()])
    #     try:
    #         args = args + kwargs_as_args
    #         result = param_fnc(*(args))
    #     except Exception as e:
    #         raise RuntimeError(f"Error while trying to execute param func:\n"
    #                            f"Func: {param_func_str}\n"
    #                            f"Args: {args}\n"
    #                            f"Error: {e}")
    #     return result

    def set_value_from_param_fn(self, *args, **iter_args):
        param_fn_args = list(args)
        for k, v in iter_args.items():
            if k not in self.param_fn.fn_args:
                self.param_fn.add_fn_arg(k)
                param_fn_args.append(v)
        param_fn_args = tuple(param_fn_args)
        self.value = self.param_fn.evaluate_fn(*param_fn_args)
        # self.value = self.run_param_fnc(*args, **iter_args)

    def template_header(self):
        return FIELD_HEADER.format(field_name=self.field_name)

    def emit(self, output_type):
        if output_type == "string_final":
            if self.isset:
                return self.get_string_value()
            else:
                return f"${{{self.param_fn}}}"
        elif output_type == "string_placeholders":
            if self.isset:
                return self.get_string_value()
            else:
                return f"$({self.field_name})"
        elif output_type == "decimal":
            assert self.isset
            return f"{self.value}"
        else:
            assert output_type == "binary"
            assert self.isset
            bin_rep = np.binary_repr(self.value, self.bitwidth)
            return f"{bin_rep}"

    def copy(self):
        field = Field(self.field_name, self.bitwidth, self.field_id, self.value, self.value_names.copy(),
                     self.value_str)

        flex_param = None if not self.param_fn else self.param_fn.copy()
        field.param_fn = flex_param
        return field
