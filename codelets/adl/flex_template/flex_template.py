from typing import List
from dataclasses import dataclass, field
from .instruction import Instruction
from codelets.adl.flex_param import FlexParam
from types import LambdaType, FunctionType

@dataclass
class FlexTemplate:
    base_instruction: Instruction
    instructions: List[Instruction] = field(default_factory=list)
    conditional: FlexParam = field(default=None)
    iter_args: List[str] = field(default_factory=list)
    iterables: List[FlexParam] = field(default_factory=list)
    num_instructions: int = field(init=False, default=1)

    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def add_condition(self, condition_str: str):
        base_args = Instruction.DEFAULT_FN_ARGS + self.iter_args
        self.conditional = FlexParam("conditional", base_args, condition_str)

    def add_iterable(self, arg_name: str, iterable):
        iterable_param = FlexParam(arg_name, Instruction.DEFAULT_FN_ARGS, iterable)
        self.iter_args.append(arg_name)
        self.iterables.append(iterable_param)

    def set_instructions(self, instruction_list: List[Instruction]):
        if len(self.instructions) > 0:
            raise RuntimeError(f"Instructions have already been evaluated!:\n"
                               f"Base instruction: {self.base_instruction}")
        self.instructions = instruction_list

    def evaluate(self, program, hag, op_idx, cdlt_id):
        fn_args = self.create_fn_args(program, hag, cdlt_id, op_idx)
        instructions = self.evaluate_iterable_instruction(fn_args, [], 0, {})
        self.set_instructions(instructions)

    def set_field_by_name(self, field_name, field_value):
        self.base_instruction.set_field_by_name(field_name, field_value)

    def set_field_flex_param(self, field_name, param_fn_str):
        self.base_instruction.set_field_flex_param(field_name, param_fn_str)

    def set_field_value(self, field_name, value, value_str=None):
        self.base_instruction.set_field_value(field_name, value, value_str=value_str)

    def set_instruction_length(self, program, hag, op_idx, cdlt_id):
        instr_size = 1
        fn_args = self.create_fn_args(program, hag, cdlt_id, op_idx)

        for idx in range(len(self.iterables)):
            iterable = self.iterables[idx].evaluate_fn(*fn_args)
            instr_size *= len(iterable)
        self.num_instructions = instr_size

    def evaluate_conditional(self, fn_args, iter_args):

        if self.conditional is not None:
            fn_args = fn_args + tuple(iter_args.values())
            condition = self.conditional.evaluate_fn(*fn_args)
        else:
            condition = True
        return condition

    # TODO: Add
    def evaluate_iterable_instruction(self, fn_args: tuple, instructions: List[Instruction], iter_idx: int, iter_args: dict):
        instruction = self.base_instruction.instruction_copy()

        if iter_idx >= len(self.iterables):
            condition = self.evaluate_conditional(fn_args, iter_args)
            if condition:
                instruction.evaluate_fields(fn_args, iter_args)
                instructions.append(instruction)
        else:
            iter_arg_name = self.iter_args[iter_idx]
            iterable_fnc = self.iterables[iter_idx]
            # TODO: Add checks for validation here
            iterable = iterable_fnc.evaluate_fn(*fn_args)
            iter_idx += 1
            for i in iterable:
                iter_args[iter_arg_name] = i
                instructions = self.evaluate_iterable_instruction(fn_args, instructions, iter_idx, iter_args)
        return instructions

    def template_copy(self):
        return FlexTemplate(self.base_instruction.instruction_copy(),
                            iter_args=self.iter_args.copy(),
                            iterables=self.iterables.copy(),
                            conditional=None if not self.conditional else self.conditional.copy()
                            )


    def create_fn_args(self, program, hag, cdlt_id, op_id):
        cdlt = program.get_codelet(cdlt_id)
        op = cdlt.get_op(op_id)
        args = [program, hag, program.relocatables, cdlt, op]
        return tuple(args)

    def emit(self, output_type="string_final"):
        if self.conditional and not self.conditional.value:
            return []
        if len(self.instructions) == 0:
            print(self.base_instruction)

        assert len(self.instructions) > 0
        instr_strings = []
        for i in self.instructions:
            instr_strings.append(i.emit(output_type))
        return instr_strings