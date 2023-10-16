from typing import NoReturn
from lark import Tree, Token


class CodeletParseError(Exception):
    pass


class CodeletError(Exception):
    pass


def raise_codelet_parse_error(message: str, obj, input_text) -> NoReturn:    
    def get_line(lines, index):
        return lines[index] if 0 <= index < len(lines) else None
    
    if isinstance(obj, Token):
        line = input_text.splitlines()[obj.line - 1]
        indicator = ' ' * obj.column + '^'
        full_message = (f"{message} at line {obj.line}, column {obj.column}:\n"
                        f"{obj.line:4} | {line}\n"
                        f"     | {indicator}")
    elif isinstance(obj, Tree):
        min_leaf = None
        max_leaf = None
        
        # Recursive function to find the min and max positions in a tree
        def traverse_tree(tree):
            nonlocal min_leaf, max_leaf
            for child in tree.children:
                if isinstance(child, Token):
                    if min_leaf is None:
                        min_leaf = child
                    if max_leaf is None:
                        max_leaf = child
                    
                    if child.start_pos < min_leaf.start_pos:
                        min_leaf = child
                    if child.end_pos > max_leaf.end_pos:
                        max_leaf = child
                else:
                    traverse_tree(child)
        
        traverse_tree(obj)

        lines = input_text.splitlines()
        start_line = obj.children[0].line if isinstance(obj.children[0], Token) else min_leaf.line 
        end_line = obj.children[-1].line if isinstance(obj.children[-1], Token) else max_leaf.line
        
        num_initial_chars_to_remove = None
        for i in range(start_line - 3, end_line + 2):
            line = get_line(lines, i)
            if line is not None:
                if num_initial_chars_to_remove is None:
                    num_initial_chars_to_remove = len(line) - len(line.lstrip())
                else:
                    num_initial_chars_to_remove = min(num_initial_chars_to_remove, len(line) - len(line.lstrip()))
        
        context_lines = []
        for i in range(start_line - 3, end_line + 2):
            line = get_line(lines, i)
            if line is not None:
                line_str = line[num_initial_chars_to_remove:]
                context_lines.append(f"{i + 1:4} | {line_str}")
        context = "\n".join(line for line in context_lines if line is not None)
        
        if start_line == end_line:
            full_message = f"{message} at line {start_line}:\n\n{context}"
        else:
            full_message = f"{message} between lines {start_line} and {end_line}:\n\n{context}"
    else:
        full_message = f"{message}. Unable to determine exact location."
    
    raise CodeletParseError(full_message)
