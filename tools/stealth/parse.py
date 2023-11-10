from lark import ParseTree, Lark, indenter


GRAMMAR = """
start: function

function: "def " NAME "(" parameters ")" ":" suite

parameters: parameter ("," parameter)*
parameter: NAME ":" type ["@" location]

suite: _NEWLINE _INDENT stmt+ _DEDENT
stmt: "assert" expr _NEWLINE -> assert
    | (one_split | multiple_split) _NEWLINE -> split
    | [NAME "="] NAME "(" call_args ")" _NEWLINE -> call
    | "for" NAME "in" "loop" "(" expr ["," expr] ")" ":" suite -> for_loop
    | "return" NAME ("," NAME)* _NEWLINE -> return_output

one_split: split_pair "=" split_call
multiple_split: "(" split_pair ")" ("," "(" split_pair ")")+ "=" split_call ("," split_call)+
split_pair: dimension "," dimension
split_call: "split" "(" dimension ")"

call_args: call_arg ("," call_arg)*
call_arg: indexed_variable | size | expr 

indexed_variable: NAME "[" expr ("," expr)* "]"

size: "[" expr ("," expr)* "]"

type: NAME "[" dimensions "]" | NAME
dtype: NAME

dimensions: dimension ("," dimension)*
dimension: NAME | INT

location: NAME

expr: "(" expr ")"
    | expr "or" expr  -> or_expr
    | expr "and" expr -> and_expr
    | expr "==" expr    -> eq_expr
    | expr "!=" expr    -> ne_expr
    | expr "<" expr     -> lt_expr
    | expr "<=" expr    -> le_expr
    | expr ">" expr     -> gt_expr
    | expr ">=" expr    -> ge_expr
    | expr "*" expr     -> mul_expr
    | expr "/" expr     -> div_expr
    | expr "//" expr    -> floordiv_expr
    | expr "+" expr     -> add_expr
    | expr "-" expr     -> sub_expr
    | NAME
    | "True"
    | "False"
    | INT

NAME: CNAME | CNAME ("." CNAME)+

_NEWLINE: ( /\\r?\\n[ \\t]*/ | COMMENT )+
COMMENT: /#[^\\n]*/

%import common.LETTER
%import common.DIGIT
%import common.INT
%import common.CNAME
%import common.WS
%import common.WS_INLINE
%declare _INDENT _DEDENT
%ignore WS
%ignore WS_INLINE
%ignore COMMENT
"""


class TreeIndenter(indenter.Indenter):
    NL_type = '_NEWLINE'
    OPEN_PAREN_types = ["LPAR"]
    CLOSE_PAREN_types = ["RPAR"]
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 4


def parse_stealth_codelet(codelet_string: str, verbose: bool = False) -> ParseTree:
    if verbose:
        print("Beginning parsing codelet...")
    parser = Lark(GRAMMAR, start='start', parser='lalr', postlex=TreeIndenter())
    tree = parser.parse(codelet_string)
    if verbose:
        print("Finished parsing codelet.")
        print(f"Codelet Parse Tree:\n{tree.pretty()}")
    return tree
