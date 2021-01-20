from sympy import Basic, Idx, IndexedBase, Lambda
from sympy.utilities.lambdify import lambdastr
from typing import Union, List, Dict
from itertools import count
import re
from types import FunctionType, LambdaType, CodeType
import inspect
from dataclasses import dataclass, field
flex_param_cnt = count()

@dataclass
class FlexParam:
    name: str
    fn_args: List[str] = field(default_factory=list)
    fn_body_str: str = field(default=None)
    fn_code_str: str = field(default=None)
    fn: LambdaType = field(default=None)
    fn_code: CodeType = field(default=None, init=False)
    value_type: str = field(default='NA', init=False)
    _value: Union[str, int] = field(default=None, init=False)
    flex_id: int = field(default_factory=lambda: next(flex_param_cnt))

    def __post_init__(self):
        if len(self.fn_args) > 0:
            self.value_type = "function"
            if self.fn_body_str is not None:
                assert self.fn_body_str is not None
                assert self.fn is None
                self.create_function_from_str(self.fn_args, self.fn_body_str)
            else:
                assert self.fn is not None and isinstance(self.fn, LambdaType)
                assert self.fn_body_str is None
                self.fn_code = self.fn.__code__
                self.fn_code_str, self.fn_body_str = get_lambda_source(self.fn)

        elif self.fn is not None and isinstance(self.fn, LambdaType):
            self.value_type = "function"
            assert self.fn_body_str is None
            self.fn_args = list(self.fn.__code__.co_varnames)
            self.fn_code = self.fn.__code__
            self.fn_code_str, self.fn_body_str = get_lambda_source(self.fn)
        else:
            self.value_type = "static"

    def create_function_from_str(self, arg_names, fn_body):
        self.fn_code_str = f"lambda {','.join(arg_names)}: {fn_body}"
        self.fn_code = compile(self.fn_code_str, "<string>", "exec")
        self.fn = LambdaType(self.fn_code.co_consts[0], globals())

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def is_set(self):
        return self.value is not None

    def add_fn_arg(self, arg):
        self.fn_args.append(arg)
        self.create_function_from_str(self.fn_args, self.fn_body_str)


    def evaluate_fn(self, *fn_args):
        assert len(fn_args) == len(self.fn_args)

        # TODO: Important--> this assumes that iter_args are iterated over in the correct order
        try:
            result = self.fn(*(fn_args))
        except Exception as e:
            raise RuntimeError(f"Error while trying to execute param func:\n"
                               f"Func: {self.name}: {self.fn_body_str}\n"
                               f"Arg names: {self.fn_args}\n"
                               f"Args: {fn_args}\n"
                               f"Error: {e}")
        if not self.is_set():
            self.value = result

        return result

    def copy(self):
        flex_param = FlexParam(self.name,
                               self.fn_args.copy(),
                               self.fn_body_str,
                               flex_id=self.flex_id)
        flex_param.value = self.value
        flex_param.fn_code = self.fn_code
        return flex_param



def lambda_to_str(fn):
    fn_str = str(inspect.getsourcelines(fn)[0])
    fn_str = fn_str.strip("['\\n']").split(" = ")[1]
    return fn_str

def get_lambda_body(lambda_str):
    groups = re.search("lambda .*: (.*)", lambda_str)
    return groups[0]

def get_lambda_source(lambda_func):
    import ast
    import os
    """Return the source of a (short) lambda function.
    If it's impossible to obtain, returns None.
    """
    try:
        source_lines, _ = inspect.getsourcelines(lambda_func)
    except (IOError, TypeError):
        print(f"IO/Type Error")
        return None

    # skip `def`-ed functions and long lambdas
    if len(source_lines) != 1:
        return None

    source_text = os.linesep.join(source_lines).strip()

    # find the AST node of a lambda definition
    # so we can locate it in the source code
    source_ast = ast.parse(source_text)
    lambda_node = next((node for node in ast.walk(source_ast)
                        if isinstance(node, ast.Lambda)), None)
    if lambda_node is None:  # could be a single line `def fn(x): ...`
        return None

    # HACK: Since we can (and most likely will) get source lines
    # where lambdas are just a part of bigger expressions, they will have
    # some trailing junk after their definition.
    #
    # Unfortunately, AST nodes only keep their _starting_ offsets
    # from the original source, so we have to determine the end ourselves.
    # We do that by gradually shaving extra junk from after the definition.
    lambda_text = source_text[lambda_node.col_offset:]
    lambda_body_text = source_text[lambda_node.body.col_offset:]
    min_length = len('lambda:_')  # shortest possible lambda expression
    while len(lambda_text) > min_length:
        try:
            # What's annoying is that sometimes the junk even parses,
            # but results in a *different* lambda. You'd probably have to
            # be deliberately malicious to exploit it but here's one way:
            #
            #     bloop = lambda x: False, lambda x: True
            #     get_short_lamnda_source(bloop[0])
            #
            # Ideally, we'd just keep shaving until we get the same code,
            # but that most likely won't happen because we can't replicate
            # the exact closure environment.
            code = compile(lambda_body_text, '<unused filename>', 'eval')
            # Thus the next best thing is to assume some divergence due
            # to e.g. LOAD_GLOBAL in original code being LOAD_FAST in
            # the one compiled above, or vice versa.
            # But the resulting code should at least be the same *length*
            # if otherwise the same operations are performed in it.
            if len(code.co_code) == len(lambda_func.__code__.co_code):
                return lambda_text, lambda_body_text
        except SyntaxError:
            pass
        lambda_text = lambda_text[:-1]
        lambda_body_text = lambda_body_text[:-1]
    return None