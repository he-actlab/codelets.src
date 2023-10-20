from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Union
from ..core import *
from .compute_operation_group import ComputeOperationGroup, build_compute_operation_groups
from ..expression import *
from .utils import input_output_dimension_to_name
from codelets.templates.codelet_template import CodeletTemplate, LoopTemplate, DummyOp, FlexParam, OperandTemplate
from codelets.examples.genesys import DTYPE_MAP, OP_DTYPES
from codelets.adl.graph import ArchitectureGraph


class CodeletTemplateLoopBodyBuilder:
    _ON_CHIP_LOCATIONS: set[str] = {"IBUF", "WBUF", "BBUF", "OBUF", "VMEM1", "VMEM2"}
    _COMPUTE_UNITS: set[str] = {"SIMD", "PE_ARRAY"}
    _CODELET_COMPUTE_NAME_TO_COMPUTE_OPERATION_NAME: dict[str, str] = {
        "mvmul": "MVMUL",
        "mvmul_bias": "MVMUL",
        "add": "ADD",
        "sub": "SUB",
        "mul": "MUL",
        "div": "DIV",
        "max": "MAX",
        "min": "MIN",
        "relu": "RELU",
        "tanh": "TANH",
        "sigmoid": "SIGMOID",
        "sqrt": "SQRT",
        "inv_sqrt": "INV_SQRT"
    }
    _COMPUTE_UNIT_LOCATION_MAP = {
        "SIMD": "SIMD",
        "PE_ARRAY": "pe_array"
    }

    _codelet: StealthCodelet
    _dummy_ops: dict[str, DummyOp]
    _operands: dict[str, OperandTemplate]

    def __init__(self, codelet: StealthCodelet, dummy_ops: dict[str, DummyOp], operands: dict[str, OperandTemplate]) -> None:
        self._codelet = codelet
        self._dummy_ops = dummy_ops.copy()
        self._operands = operands.copy()
    
    def build(self, codelet_template: CodeletTemplate, compute_operation_group: ComputeOperationGroup, compute_operation_output_operands: list[OperandTemplate], loop_indices: dict[str, LoopTemplate]) -> None:
        compute_operation_index: int = 0
        for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
            if isinstance(transfer_or_compute_operation, StealthLoad) and transfer_or_compute_operation.location in self._ON_CHIP_LOCATIONS:
                codelet_template.transfer(self._operands[transfer_or_compute_operation.source_operand_name], ["DRAM", transfer_or_compute_operation.location])
            elif isinstance(transfer_or_compute_operation, StealthStore) and self._codelet._operands[transfer_or_compute_operation.destination_operand_name].location in self._ON_CHIP_LOCATIONS:
                codelet_template.transfer(self._operands[transfer_or_compute_operation.destination_operand_name], [self._codelet._operands[transfer_or_compute_operation.destination_operand_name].location, "DRAM"])
            elif isinstance(transfer_or_compute_operation, StealthCompute):
                write_location: str = self._get_compute_operation_output_write_location(transfer_or_compute_operation, compute_operation_group)
                compute_operation_output_operands[compute_operation_index].set_write_destination(write_location)
                input_offsets, output_offsets = self._get_compute_operation_input_and_output_offsets(transfer_or_compute_operation, compute_operation_group, loop_indices)
                codelet_template.compute(
                    self._CODELET_COMPUTE_NAME_TO_COMPUTE_OPERATION_NAME[transfer_or_compute_operation.operation_name], 
                    [self._operands[operand_name][input_offsets[operand_name]] if operand_name in input_offsets else self._operands[operand_name] for operand_name in transfer_or_compute_operation.operands],
                    [compute_operation_output_operands[compute_operation_index][output_offsets[transfer_or_compute_operation.destination_operand_name]]],
                    target=self._COMPUTE_UNIT_LOCATION_MAP[transfer_or_compute_operation.location]
                )
                compute_operation_index += 1
    
    def _get_compute_operation_output_write_location(self, compute_operation: StealthCompute, compute_operation_group: ComputeOperationGroup) -> str:
        is_next_store: bool = False
        for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
            if isinstance(transfer_or_compute_operation, StealthCompute) and transfer_or_compute_operation == compute_operation:
                is_next_store = True
            elif isinstance(transfer_or_compute_operation, StealthStore) and is_next_store and transfer_or_compute_operation.source_operand_name == compute_operation.destination_operand_name:
                return self._codelet._operands[transfer_or_compute_operation.destination_operand_name].location
        raise RuntimeError(f"Cannot find the write location for compute operation {compute_operation}.")
    
    def _get_compute_operation_input_and_output_offsets(self, compute_operation: StealthCompute, compute_operation_group: ComputeOperationGroup, loop_indices: dict[str, LoopTemplate]) -> tuple[dict[str, tuple[Any]], dict[str, tuple[Any]]]:
        input_offsets: dict[str, tuple[Any]] = {}
        output_offsets: dict[str, tuple[Any]] = {}
        is_compute_occurred: bool = False
        for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
            if isinstance(transfer_or_compute_operation, StealthLoad) and transfer_or_compute_operation.location in self._COMPUTE_UNITS and transfer_or_compute_operation.destination_operand_name in compute_operation.operands:
                input_offsets[transfer_or_compute_operation.destination_operand_name] = tuple(map(partial(self._expression_to_loop_index_expression, loop_indices=loop_indices), transfer_or_compute_operation.source_operand_offset))
            elif isinstance(transfer_or_compute_operation, StealthCompute) and transfer_or_compute_operation == compute_operation:
                is_compute_occurred = True
            elif isinstance(transfer_or_compute_operation, StealthStore) and self._codelet._operands[transfer_or_compute_operation.source_operand_name].location in self._COMPUTE_UNITS and transfer_or_compute_operation.source_operand_name == compute_operation.destination_operand_name and is_compute_occurred:
                output_offsets[transfer_or_compute_operation.source_operand_name] = tuple(map(partial(self._expression_to_loop_index_expression, loop_indices=loop_indices), transfer_or_compute_operation.destination_operand_offset))
                break
        return (input_offsets, output_offsets)
    
    def _expression_to_loop_index_expression(self, input_expression: StealthExpression, loop_indices: dict[str, LoopTemplate]) -> Any:
        def helper(expression: StealthExpression):
            if isinstance(expression, StealthLiteral):
                return expression.value
            elif isinstance(expression, StealthVariableName):
                if expression.name in loop_indices:
                    return loop_indices[expression.name]
                elif expression.name in self._dummy_ops:
                    return self._dummy_ops[expression.name]
                else:
                    raise RuntimeError(f"Unknown variable name: {expression.name}")
            elif isinstance(expression, StealthUnaryExpression):
                if expression.operation == "-":
                    return -helper(expression.operand)
                else:
                    raise RuntimeError(f"Unknown unary operation: {expression.operation}")
            elif isinstance(expression, StealthBinaryExpression):
                lhs = helper(expression.lhs)
                rhs = helper(expression.rhs)
                if expression.operation == "+":
                    return lhs + rhs
                elif expression.operation == "-":
                    return lhs - rhs
                elif expression.operation == "*":
                    return lhs * rhs
                elif expression.operation == "/":
                    return lhs / rhs
                elif expression.operation == "//":
                    return lhs / rhs
                else:
                    raise RuntimeError(f"Unknown binary operation for conversion to loop index expression: {expression.operation}")
            else:
                raise RuntimeError(f"Cannot convert expression to loop index expression: {type(expression)}")

        input_expression = evaluate_expression(input_expression, {})
        return helper(input_expression)


class CodeletTemplateBuilder:
    _codelet: StealthCodelet

    _dummy_ops: dict[str, DummyOp]
    _operands: dict[str, OperandTemplate]

    def __init__(self, codelet: StealthCodelet) -> None:
        self._codelet = codelet

        self._dummy_ops = {}
        self._operands = {}
    
    def _add_dummy_op(self, dummy_op_name: str, dummy_op: DummyOp) -> None:
        self._dummy_ops[dummy_op_name] = dummy_op
    
    def _get_dummy_op(self, dummy_op_name: str) -> DummyOp:
        return self._dummy_ops[dummy_op_name]

    def _is_dummy_op_exist(self, dummy_op_name: str) -> bool:
        return dummy_op_name in self._dummy_ops
    
    def _add_operand(self, operand_name: str, operand: OperandTemplate) -> None:
        self._operands[operand_name] = operand
    
    def _get_operand(self, operand_name: str) -> OperandTemplate:
        return self._operands[operand_name]
    
    def _is_operand_exist(self, operand_name: str) -> bool:
        return operand_name in self._operands
    
    def build(self, codelet_template: CodeletTemplate, input_dtype, acc_dtype) -> None:
        self._create_dimension_dummy_ops(codelet_template)
        self._create_argument_operands(codelet_template, input_dtype, acc_dtype)
        self._build_codelet_template_body(codelet_template, input_dtype, acc_dtype)
    
    def _create_dimension_dummy_ops(self, codelet_template: CodeletTemplate) -> None:
        self._dimension_dummy_op_creation_helper(codelet_template, self._codelet.inputs, is_input=True)
        self._dimension_dummy_op_creation_helper(codelet_template, self._codelet.outputs, is_input=False)

    def _dimension_dummy_op_creation_helper(self, codelet_template: CodeletTemplate, stealth_operands: list[StealthOperand], is_input=True) -> None:
        for i, stealth_operand in enumerate(stealth_operands):
            for j, stealth_codelet_operand_dimension in enumerate(stealth_operand.shape):
                dimension_name = input_output_dimension_to_name(stealth_codelet_operand_dimension)
                if not self._is_dummy_op_exist(dimension_name):
                    fn_body_str = f"node.inputs[{i}].shape[{j}]" if is_input else f"node.outputs[{i}].shape[{j}]"
                    new_flex_param = FlexParam(
                        name=dimension_name,
                        fn_args=["node"],
                        fn_body_str=fn_body_str,
                    )
                    new_flex_param.create_function_from_str(["node"], fn_body_str)
                    new_dummy_op = DummyOp(
                        ["NodePlaceholder"],
                        new_flex_param,
                        dtype=None
                    )
                    codelet_template._dummy_ops[dimension_name] = new_dummy_op
                    self._add_dummy_op(dimension_name, new_dummy_op)
    
    def _create_argument_operands(self, codelet_template: CodeletTemplate, input_dtype, acc_dtype) -> None:
        input_operands: list[OperandTemplate] = self._argument_operand_creation_helper(codelet_template, self._codelet.inputs, input_dtype, acc_dtype)
        output_operands: list[OperandTemplate] = self._argument_operand_creation_helper(codelet_template, self._codelet.outputs, input_dtype, acc_dtype)
        codelet_template.set_inputs(input_operands)
        codelet_template.set_outputs(output_operands)
    
    def _argument_operand_creation_helper(self, codelet_template: CodeletTemplate, stealth_operands: list[StealthOperand], input_dtype, acc_dtype) -> list[OperandTemplate]:
        ret: list[OperandTemplate] = []
        for stealth_operand in stealth_operands:
            ret.append(self._create_codelet_template_operand(codelet_template, stealth_operand, input_dtype, acc_dtype))
        return ret
    
    def _create_codelet_template_operand(self, codelet_template: CodeletTemplate, stealth_operand: StealthOperand, input_dtype, acc_dtype) -> OperandTemplate:
        data_type = input_dtype if stealth_operand.dtype == "i8" else acc_dtype
        operand_template: OperandTemplate = codelet_template.create_operand_template(stealth_operand.name, OP_DTYPES, list(map(lambda d: self._constant_expression_to_dummy_op(d), stealth_operand.shape)), default_dtype=data_type)
        self._add_operand(stealth_operand.name, operand_template)
        return operand_template
    
    def _constant_expression_to_dummy_op(self, expression: StealthExpression) -> Union[DummyOp, int]:
        assert is_expression_constant(expression, set(map(lambda p: p.name, self._codelet.params))), f"Expression {expression} is not constant."
        if isinstance(expression, StealthLiteral) and isinstance(expression.value, int):
            return expression.value
        elif isinstance(expression, StealthVariableName):
            return self._get_dummy_op(expression.name) 
        elif isinstance(expression, StealthBinaryExpression):
            if expression.operation == "+":
                return self._constant_expression_to_dummy_op(expression.lhs) + self._constant_expression_to_dummy_op(expression.rhs)
            elif expression.operation == "-":
                return self._constant_expression_to_dummy_op(expression.lhs) - self._constant_expression_to_dummy_op(expression.rhs)
            elif expression.operation == "*":
                return self._constant_expression_to_dummy_op(expression.lhs) * self._constant_expression_to_dummy_op(expression.rhs)
            elif expression.operation == "/":
                return self._constant_expression_to_dummy_op(expression.lhs) / self._constant_expression_to_dummy_op(expression.rhs)
            elif expression.operation == "//":
                return self._constant_expression_to_dummy_op(expression.lhs) // self._constant_expression_to_dummy_op(expression.rhs)
            else:
                raise RuntimeError(f"Unsupported binary operation for conversion to dummy op: {expression.operation}")
        else:
            raise RuntimeError(f"Cannot convert {type(expression)} to dummy op.")
    
    def _build_codelet_template_body(self, codelet_template: CodeletTemplate, input_dtype, acc_dtype) -> None:
        compute_operation_groups: list[ComputeOperationGroup] = build_compute_operation_groups(self._codelet)
        self._create_top_level_allocated_operands(codelet_template, compute_operation_groups, input_dtype, acc_dtype)
        self._link_tile_and_element_stealth_operands_to_top_level_codelet_template_operands(compute_operation_groups)
        compute_operation_output_operands: list[list[OperandTemplate]] = self._create_compute_operation_output_operands(codelet_template, compute_operation_groups, input_dtype, acc_dtype)
        loop_body_builder = CodeletTemplateLoopBodyBuilder(self._codelet, self._dummy_ops, self._operands)

        for compute_operation_group, compute_operation_group_output_operands in zip(compute_operation_groups, compute_operation_output_operands):
            print(compute_operation_group)
            start_config, end_config = self._create_config_functions(compute_operation_group)

            start_config(codelet_template)
            loop_templates: OrderedDict[str, LoopTemplate] = self._create_loop_templates(codelet_template, compute_operation_group)
            loop_body_builder.build(codelet_template, compute_operation_group, compute_operation_group_output_operands, loop_templates)
            self._close_loop_templates(loop_templates)
            end_config(codelet_template)
            
    def _create_top_level_allocated_operands(self, codelet_template: CodeletTemplate, compute_operation_groups: list[ComputeOperationGroup], input_dtype, acc_dtype) -> None:
        top_level_allocations: set[StealthAllocation] = set()
        for compute_operation_group in compute_operation_groups:
            top_level_allocations.update(compute_operation_group.top_level_allocations)
        for top_level_allocation in top_level_allocations:
            if top_level_allocation.operand_name not in map(lambda o: o.name, self._codelet.outputs):
                new_operand: OperandTemplate = self._create_codelet_template_operand(codelet_template, self._get_operand(top_level_allocation.operand_name), input_dtype, acc_dtype) 
                codelet_template.add_temp_operand(new_operand)
    
    def _link_tile_and_element_stealth_operands_to_top_level_codelet_template_operands(self, compute_operation_groups: list[ComputeOperationGroup]) -> None:
        for compute_operation_group in compute_operation_groups:
            for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
                if isinstance(transfer_or_compute_operation, StealthLoad):
                    self._add_operand(transfer_or_compute_operation.destination_operand_name, self._get_operand(transfer_or_compute_operation.source_operand_name))
            for transfer_or_compute_operation in reversed(compute_operation_group.transfer_and_compute_operations):
                if isinstance(transfer_or_compute_operation, StealthStore):
                    self._add_operand(transfer_or_compute_operation.source_operand_name, self._get_operand(transfer_or_compute_operation.destination_operand_name))
        
    def _create_compute_operation_output_operands(self, codelet_template: CodeletTemplate, compute_operation_groups: list[ComputeOperationGroup], input_dtype, acc_dtype) -> list[list[OperandTemplate]]:
        ret: list[list[OperandTemplate]] = [[]]
        for compute_operation_group in compute_operation_groups:
            for compute_operation in compute_operation_group.operations:
                if self._is_operand_exist(compute_operation.destination_operand_name):
                    ret[-1].append(self._get_operand(compute_operation.destination_operand_name))
                else:
                    raise NotImplementedError("TODO: Create intermediate operands automatically.")
            ret.append([])
        return ret
    
    def _create_config_functions(self, compute_operation_group: ComputeOperationGroup) -> tuple[Callable[[CodeletTemplate], None], Callable[[CodeletTemplate], None]]:
        if compute_operation_group.is_systolic_array_operation:
            return (self._systolic_array_start_config, self._systolic_array_end_config)
        elif compute_operation_group.is_simd_operation:
            return (partial(self._simd_start_config, immediates=[(immediate_name, self._constant_expression_to_dummy_op(self._codelet.get_immediate(immediate_name))) for immediate_name in compute_operation_group.immediates]), self._simd_end_config)
        else:
            raise RuntimeError(f"Somehow got a compute operation group that is neither a systolic array operation nor a SIMD operation.")

    def _systolic_array_start_config(self, codelet_template: CodeletTemplate) -> None:
        codelet_template.configure("start", "systolic_array")
        codelet_template.configure("start", "WBUF")
        codelet_template.configure("start", "BBUF")
        codelet_template.configure("start", "IBUF")
        codelet_template.configure("start", "OBUF")
    
    def _systolic_array_end_config(self, codelet_template: CodeletTemplate) -> None:
        codelet_template.configure("end", "WBUF")
        codelet_template.configure("end", "BBUF")
        codelet_template.configure("end", "IBUF")
        codelet_template.configure("end", "OBUF")
        codelet_template.configure("end", "systolic_array")
    
    def _simd_start_config(self, codelet_template: CodeletTemplate, immediates: list[tuple[str, Any]]) -> None:
        immediate_dummy_ops = []
        for immediate_name, immediate_value in immediates:
            immediate_dummy_ops.append(codelet_template.dummy_op(immediate_name, immediate_value))
            temp_operand: OperandTemplate = codelet_template.create_temp_operand([self._codelet.get_simd_width()], "IMM", name=immediate_name)
            self._add_operand(str(immediate_value), temp_operand)
        codelet_template.configure("start", "SIMD")
        for immediate_dummy_op in immediate_dummy_ops:
            codelet_template.configure("start", immediate_value=immediate_dummy_op)
            
    def _simd_end_config(self, codelet_template: CodeletTemplate) -> None:
        codelet_template.configure("end", "SIMD")
    
    def _create_loop_templates(self, codelet_template: CodeletTemplate, compute_operation_group: ComputeOperationGroup) -> OrderedDict[str, LoopTemplate]:
        loop_templates: OrderedDict[str, LoopTemplate] = OrderedDict()
        assert len(compute_operation_group.outer_loop_index_variables) == len(compute_operation_group.inner_loop_index_variables), f"Number of outer loop index variables ({len(compute_operation_group.outer_loop_index_variables)}) does not match number of inner loop index variables ({len(compute_operation_group.inner_loop_index_variables)})."
        for outer_loop_index_variable, inner_loop_index_variable in zip(compute_operation_group.outer_loop_index_variables, compute_operation_group.inner_loop_index_variables):
            tile_size = evaluate_expression(outer_loop_index_variable.stride, {})
            if isinstance(tile_size, StealthBinaryExpression):
                assert isinstance(tile_size.lhs, StealthVariableName)
                dimension = self._get_dummy_op(tile_size.lhs.name)
            elif isinstance(tile_size, StealthVariableName):
                dimension = self._get_dummy_op(tile_size.name)
            else:
                raise RuntimeError(f"Invalid tile size for outer loop {outer_loop_index_variable}")
            loop_template: LoopTemplate = codelet_template.loop(dimension).__enter__()
            loop_templates[inner_loop_index_variable.name] = loop_template
        return loop_templates

    def _close_loop_templates(self, loop_templates: OrderedDict[str, LoopTemplate]) -> None:
        for loop_template in reversed(loop_templates.values()):
            loop_template.__exit__(None, None, None)


def build_codelet_template(codelet: StealthCodelet) -> Callable[[ArchitectureGraph], CodeletTemplate]:
    def codelet_template_func(hag: ArchitectureGraph) -> CodeletTemplate:
        input_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
        acc_dtype = DTYPE_MAP[ f"FXP{hag.meta_cfg['ACC_WIDTH']}"] 
        builder = CodeletTemplateBuilder(codelet)
        with CodeletTemplate(codelet.operation_name) as codelet_template:
            builder.build(codelet_template, input_dtype, acc_dtype)
        return codelet_template
    return codelet_template_func
