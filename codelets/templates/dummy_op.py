from typing import Dict, Any, Tuple, List, TYPE_CHECKING, Union
from dataclasses import dataclass, field
from functools import singledispatchmethod, singledispatch
from . import TEMPLATE_CLASS_ARG_MAP
from itertools import count
from typing import ClassVar
import inspect


from codelets.adl.flex_param import FlexParam

if TYPE_CHECKING:
    from .operation_template import LoopTemplate

flex_param_cnt = count()

@dataclass
class DummyOp:
    # This is a placeholder for templates. Each time a node method is called on this,
    # a function is generated to preserve the function call for when the value can be computed during
    # instantiation.
    template_types: List[str]
    flex_param: FlexParam
    obj_instance: Any = field(default=None)
    op_count: ClassVar[int] = 0

    def __post_init__(self):
        DummyOp.op_count += 1

    @property
    def name(self):
        return self.flex_param.name

    @property
    def value(self):
        return self.flex_param.value

    def update(self, dummy_op: 'DummyOp'):
        self.template_types = dummy_op.template_types
        self.flex_param.update_fn_code_args(dummy_op.flex_param.fn_args,
                                            dummy_op.flex_param.fn_body_str)
        self.obj_instance = dummy_op.obj_instance
        return dummy_op

    def __getattr__(self, name):
        # TODO: Add assertion here to make sure its a valid attribute
        new_code = f"{self.flex_param.fn_body_str}.{name}"
        self.flex_param.update_fn_code(new_code)
        return self

    def __getitem__(self, key):
        if isinstance(key, str):
            key = f"'{key}'"
        new_code = f"{self.flex_param.fn_body_str}[{key}]"
        self.flex_param.update_fn_code(new_code)
        return self

    def __len__(self):
        new_code = f"len({self.flex_param.fn_body_str})"
        self.flex_param.update_fn_code(new_code)
        return self

    def evaluate(self, instance_args):
        obj_instances = []
        for t in self.template_types:
            if instance_args[t] not in obj_instances:
                obj_instances.append(instance_args[t])
        obj_instances = tuple(obj_instances)
        return self.flex_param.evaluate_fn(*obj_instances, force_evaluate=True)

    def __mul__(self, other):
        return dummy_op(other, self, '*')

    def __rmul__(self, other):
        return dummy_op(other, self, '*', reflected=True)

    def __add__(self, other):
        return dummy_op(other, self, '+')

    def __radd__(self, other):
        return dummy_op(other, self, '+', reflected=True)

    def __sub__(self, other):
        return dummy_op(other, self, '-')

    def __rsub__(self, other):
        return dummy_op(other, self, '-', reflected=True)

    def __truediv__(self, other):
        return dummy_op(other, self, '/')

    def __rdiv__(self, other):
        return dummy_op(other, self, '/', reflected=True)

    def __floordiv__(self, other):
        return dummy_op(other, self, '//')

    def __rfloordiv__(self, other):
        return dummy_op(other, self, '//', reflected=True)

    def __mod__(self, other):
        return dummy_op(other, self, '%')

    def __rmod__(self, other):
        return dummy_op(other, self, '%', reflected=True)

@dataclass
class DummyParam:
    # This is a placeholder for templates. Each time a node method is called on this,
    # a function is generated to preserve the function call for when the value can be computed during
    # instantiation.
    flex_param: FlexParam
    dummy_args: Tuple[Union[DummyOp, 'LoopTemplate', 'DummyParam']]
    obj_instance: Any = field(default=None)
    op_count: ClassVar[int] = 0

    def __post_init__(self):
        DummyParam.op_count += 1

    @property
    def name(self):
        return self.flex_param.name

    @property
    def value(self):
        return self.flex_param.value

    def evaluate(self, instance_args):
        args = []
        for d in self.dummy_args:
            res = d.evaluate(instance_args)
            args.append(res)
        return self.flex_param.evaluate_fn(*tuple(args), force_evaluate=True)

    def get_full_obj_type(self, obj):
        obj_mro = inspect.getmro(obj.__class__)
        assert len(obj_mro) >= 2
        base = obj_mro[-2]
        name = f"{base.__module__}.{base.__name__}"
        return name

# TODO: Need to fix the argument names here, this is extremely confusing
@singledispatch
def dummy_op(op1, op2, op_str, reflected=False):
    raise NotImplementedError(f"No implementation for loop {op_str} op {type(op1)}.")

@dummy_op.register
def _(op1: DummyOp, op2: DummyOp, op_str: str, reflected=False):
    template_type_str = list(set(op1.template_types + op2.template_types))
    arg_str = [TEMPLATE_CLASS_ARG_MAP[t][0] for t in template_type_str]
    if reflected:
        fp_name = f"({op1.name}{op_str}{op2.name})"
        fn_str = f"({op1.flex_param.fn_body_str}{op_str}{op2.flex_param.fn_body_str})"
    else:
        fp_name = f"({op2.name}{op_str}{op1.name})"
        fn_str = f"({op2.flex_param.fn_body_str}){op_str}({op1.flex_param.fn_body_str})"

    fp = FlexParam(fp_name, arg_str, fn_str)
    return DummyOp(template_type_str, fp)

@dummy_op.register
def _(op1: int, op2: DummyOp, op_str: str, reflected=False):
    if reflected:
        new_code = f"({op1}{op_str}{op2.flex_param.fn_body_str})"
    else:
        new_code = f"({op2.flex_param.fn_body_str}{op_str}{op1})"
    op2.flex_param.update_fn_code(new_code)
    return op2

# @singledispatch
# def dummy_mul(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_rmul(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_add(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_radd(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_div(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_rdiv(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_floor_div(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_rfloor_div(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_sub(op2, op1):
#     raise NotImplementedError
#
# @singledispatch
# def dummy_rsub(op2, op1):
#     raise NotImplementedError
#
# @dummy_mul.register
# def _(op2: DummyOp, op1: DummyOp):
#     return dummy_dummy_op(op1, op2, "*")
#
# @dummy_mul.register
# def _(op2: int, op1: DummyOp):
#     return int_dummy_op(op2, op1, "*")
#
# @dummy_rmul.register
# def _(op1: DummyOp, op2: DummyOp):
#     return dummy_dummy_op(op2, op1, "*")
#
# @dummy_rmul.register
# def _(op1: int, op2: DummyOp):
#     return int_dummy_op(op1, op2, "*", reflected=True)
#
# @dummy_sub.register
# def _(op1: DummyOp, op2: DummyOp):
#     return dummy_dummy_op(op2, op1, "-")
#
# @dummy_sub.register
# def _(op1: int, op2: DummyOp):
#     return int_dummy_op(op1, op2, "-")
#
# @dummy_rsub.register
# def _(op2: DummyOp, op1: DummyOp):
#     return dummy_dummy_op(op1, op2, "-")
#
# @dummy_rsub.register
# def _(op2: int, op1: DummyOp):
#     return int_dummy_op(op2, op1, "-", reflected=True)
#
# @dummy_add.register
# def _(op1: DummyOp, op2: DummyOp):
#     return dummy_dummy_op(op2, op1, "+")
#
# @dummy_add.register
# def _(op1: int, op2: DummyOp):
#     return int_dummy_op(op1, op2, "+")
#
# @dummy_radd.register
# def _(op2: DummyOp, op1: DummyOp):
#     return dummy_dummy_op(op1, op2, "+")
#
# @dummy_radd.register
# def _(op2: int, op1: DummyOp):
#     return int_dummy_op(op2, op1, "+", reflected=True)
#
# @dummy_div.register
# def _(op1: DummyOp, op2: DummyOp):
#     return dummy_dummy_op(op2, op1, "/")
#
# @dummy_div.register
# def _(op1: int, op2: DummyOp):
#     return int_dummy_op(op1, op2, "/")
#
# @dummy_rdiv.register
# def _(op2: DummyOp, op1: DummyOp):
#     return dummy_dummy_op(op1, op2, "/")
#
# @dummy_rdiv.register
# def _(op2: int, op1: DummyOp):
#     return int_dummy_op(op2, op1, "/", reflected=True)
#
# @dummy_floor_div.register
# def _(op1: DummyOp, op2: DummyOp):
#     return dummy_dummy_op(op2, op1, "//")
#
# @dummy_floor_div.register
# def _(op1: int, op2: DummyOp):
#     return int_dummy_op(op1, op2, "//")
#
# @dummy_rfloor_div.register
# def _(op2: DummyOp, op1: DummyOp):
#     return dummy_dummy_op(op1, op2, "//")
#
# @dummy_rfloor_div.register
# def _(op2: int, op1: DummyOp):
#     return int_dummy_op(op2, op1, "//", reflected=True)
#
# def dummy_dummy_op(op1: DummyOp, op2: DummyOp, op_str: str):
#     template_type_str = list(set(op1.template_types + op2.template_types))
#     arg_str = [TEMPLATE_CLASS_ARG_MAP[t][0] for t in template_type_str]
#     fn_str = f"{op1.flex_param.fn_body_str}{op_str}{op2.flex_param.fn_body_str}"
#     fp = FlexParam(f"({op1.name}{op_str}{op1.name})", arg_str, fn_str)
#     return DummyOp(template_type_str, fp)
#
# def int_dummy_op(op1: int, op2: DummyOp, op_str: str, reflected=False):
#     if reflected:
#         new_code = f"({op1}{op_str}{op2.flex_param.fn_body_str})"
#     else:
#         new_code = f"({op2.flex_param.fn_body_str}{op_str}{op1})"
#     op2.flex_param.update_fn_code(new_code)
#     return op2