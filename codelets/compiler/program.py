import json
from typing import List, Callable, Dict, List, Any
from collections import defaultdict
from time import time
from codelets.adl.flex_param import FlexParam
from codelets.adl.flex_template.instruction import Instruction
from codelets.templates.codelet_template import CodeletTemplate
from codelets.adl.operation import Operand, Loop, Compute, Transfer, Configure, Operation
from codelets.adl.graph import ArchitectureNode
from codelets.codelet_impl import Codelet
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import polymath as pm
from sympy import Basic
from .relocation_table import RelocationTable
import networkx as nx

EMIT_OPTIONS = ["decimal", "operations", "string_final", "string_placeholders", "binary"]

@dataclass
class CompilationStage:
    name: str
    level: str
    compilation_fn: Callable
    dependencies: List[str]
    fn_kwargs: Dict[str, Any] = field(default_factory=dict)
    skip_noops: bool = field(default=True)

    def __post_init__(self):
        # TODO: Check function signature
        pass

    def run(self, *args):
        return self.compilation_fn(*args, **self.fn_kwargs)

@dataclass
class OperandDataflow:
    node_name: str
    node_type: str
    cdlt_read: List[int] = field(default_factory=list)
    cdlt_write: List[int] = field(default_factory=list)
    read_operand_names: List[str] = field(default_factory=list)
    write_operand_names: List[str] = field(default_factory=list)

    def add_read(self, cdlt: Codelet, operand: Operand):
        self.read_operand_names.append(operand.name)
        self.cdlt_read.append(cdlt.instance_id)

    def add_write(self, cdlt: Codelet, operand: Operand):
        self.write_operand_names.append(operand.name)
        self.cdlt_write.append(cdlt.instance_id)

class CodeletProgram(object):

    def __init__(self, graph: pm.Node, hag: ArchitectureNode, program_mode: str="inference"):
        Operation.reset()
        Codelet.reset()
        self._name = graph.name
        self._hag = hag
        self._graph = graph
        self._codelets = []
        self._codelet_templates = {}
        self._relocatables = RelocationTable(hag.get_off_chip_storage())
        self._compilation_pipeline = defaultdict(list)
        self._preproc_stages = defaultdict(list)
        self._template_stages = defaultdict(list)
        self._instruction_stages = defaultdict(list)
        self._program_mode = program_mode
        self._side_effect_params = {'program': {}, 'codelet': {}, 'op': {}}
        self._operand_mapping = {}
        self._codelet_instructions = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def hag(self) -> ArchitectureNode:
        return self._hag

    @property
    def program_mode(self):
        return self._program_mode

    @property
    def graph(self):
        return self._graph

    @property
    def operand_mapping(self) -> Dict[str, OperandDataflow]:
        return self._operand_mapping

    @property
    def codelets(self) -> List[Codelet]:
        return self._codelets

    @property
    def relocatables(self) -> RelocationTable:
        return self._relocatables

    @property
    def codelet_instructions(self) -> Dict[str, List[Instruction]]:
        return self._codelet_instructions

    @property
    def compilation_pipeline(self) -> Dict[int, List[CompilationStage]]:
        return self._compilation_pipeline

    @property
    def preproc_stages(self) -> Dict[int, List[CompilationStage]]:
        return self._preproc_stages

    @property
    def template_stages(self) -> Dict[int, List[CompilationStage]]:
        return self._template_stages

    @property
    def instruction_stages(self) -> Dict[int, List[CompilationStage]]:
        return self._instruction_stages

    @property
    def side_effect_params(self):
        return self._side_effect_params

    @property
    def codelet_templates(self):
        return self._codelet_templates

    def add_side_effect_param(self, name, scope, init_val):
        if scope == 'program':
            self._side_effect_params[scope][name] = init_val
        elif scope == 'codelet':
            for c in self.codelets:
                self._side_effect_params[scope][c.cdlt_uid][name] = init_val
        else:
            raise RuntimeError(f"Currently no other scopes are supported for side effects other than 'program'"
                               f" and 'codelet'. Request side effect: {scope}")

    def set_relocation_ns_offsets(self, offsets: Dict[str, int], offset_type="address"):
        assert self.relocatables.is_empty
        mem_layout = list(offsets.keys())
        assert offset_type in ["address", "bits"]
        if offset_type == "address":
            nbanks = self.relocatables.storage_node.banks
            width = self.relocatables.storage_node.width
            offsets = {k: v*nbanks*width for k, v in offsets.items()}
        self._relocatables = RelocationTable(self.hag.get_off_chip_storage(),
                                             mem_layout=mem_layout,
                                             offsets=offsets)

    def update_side_effect_param(self, name, scope, value, codelet_id=None, operation_id=None):
        if scope == 'program':
            self._side_effect_params[scope][name] = value
        elif scope == 'codelet':
            assert codelet_id in self.side_effect_params['codelet']
            self._side_effect_params[scope][codelet_id][name] = value

    def extract_bits(self, value: int, num_bits: int, pos: int):
        return ((((1 << num_bits) - 1) << pos) & value) >> pos

    def add_codelet(self, cdlt: Codelet):
        self._codelets.append(cdlt)

    def get_codelet(self, cdlt_id: int):
        for cdlt in self.codelets:
            if cdlt.instance_id == cdlt_id:
                return cdlt
        raise KeyError(f"Unable to get codelet with id {cdlt_id} in codelet list")

    def save(self, output_path=None, save_format="json"):
        if output_path:
            if output_path[-1] != "/":
                output_path = output_path + "/"
        else:
            output_path = str(Path.cwd()) + "/"

        if save_format == "json":
            full_path = f"{output_path}{self.name}.json"
            self.save_json(full_path)
        elif save_format == "text":
            full_path = f"{output_path}{self.name}.txt"
            self.save_text(full_path)
        else:
            raise ValueError(f"Invalid file output: {save_format}")

    def save_binary(self, full_path):
        raise NotImplementedError

    def get_tiling_dims(self, inputs: List[Operand], outputs: List[Operand]):
        assert all([i.is_instantiated() for i in inputs])
        assert all([o.is_instantiated() for o in outputs])

    def add_compilation_step(self, name: str,
                             compilation_fn: Callable,
                             level='codelet',
                             dependencies=None,
                             stage_kwargs=None,
                             skip_noops=True,
                             insert_idx=-1,
                             preproc=False,
                             template=False,
                             instruction=False
                             ):
        if not callable(compilation_fn):
            raise TypeError(f"Compilation step must be a callable function:\n"
                            f"Name: {name}\n"
                            f"Compilation arg: {compilation_fn}, Type: {type(compilation_fn)}")
        elif name in self.compilation_pipeline[level]:
            raise KeyError(f"Compilation for compilation stage already exists:\n"
                           f"Name: {name}")
        stage_kwargs = stage_kwargs or {}
        dependencies = dependencies or []
        level_names = [comp_stage.name for comp_stage in self.compilation_pipeline[level]]
        # for d in dependencies:
        #     assert d in level_names

        fn_obj = CompilationStage(name, level, compilation_fn, dependencies, fn_kwargs=stage_kwargs, skip_noops=skip_noops)
        assert not (preproc and template)
        if preproc:
            if insert_idx >= 0:
                self._preproc_stages[level].insert(insert_idx, fn_obj)
            else:
                self._preproc_stages[level].append(fn_obj)
        elif template:
            if insert_idx >= 0:
                self._template_stages[level].insert(insert_idx, fn_obj)
            else:
                self._template_stages[level].append(fn_obj)
        elif instruction:
            if insert_idx >= 0:
                self._instruction_stages[level].insert(insert_idx, fn_obj)
            else:
                self._instruction_stages[level].append(fn_obj)
        else:
            if insert_idx >= 0:
                self._compilation_pipeline[level].insert(insert_idx, fn_obj)
            else:
                self._compilation_pipeline[level].append(fn_obj)

    INSTR_FN_TEMPLATE = """def param_fn{FN_ID}(hag, op, cdlt, relocation_table, program, fixed_val=None): return {FN_BODY}"""

    def set_instruction_templates(self, cdlt: Codelet):
        for o in cdlt.ops:

            template = self.hag.get_operation_template(o)
            template_copy = [ft.template_copy() for ft in template]

            o.set_template(template_copy)

    def instantiate_instructions(self, cdlt: Codelet, fixed_val=None):

        for o in cdlt.ops:
            args = (self, self.hag, o.global_op_id, cdlt.instance_id)
            for ft in o.instructions:
                ft.evaluate(*args)

    def instantiate_codelet(self, node):
        cdlt_template = self.codelet_templates[node.op_name]
        assert isinstance(cdlt_template, CodeletTemplate), f"Invalid template: {cdlt_template}"
        cdlt = cdlt_template.instantiate({"HAGPlaceholder": self.hag, "NodePlaceholder": node})
        self.add_codelet(cdlt)

        for i, operand in enumerate(cdlt.inputs):
            n = node.inputs[i]
            self.add_operand_mapping(cdlt, node, n, operand, "input")
            if operand.node_name is None:
                operand.node_name = n.name

        for o, operand in enumerate(cdlt.outputs):
            n = node.outputs[o]
            self.add_operand_mapping(cdlt, node, n, operand, "output")
            if operand.node_name is None:
                operand.node_name = n.name

        return cdlt

    def emit_compute_op(self, op, output_type):

        if output_type == "operations":
            source_names = [s.name for s in op.sources]
            dst_names = [d.name for d in op.dests]
            op_str = f"{op.op_str}: {op.target}-{op.op_name}({source_names})->{dst_names}"
        else:
            assert output_type == "json"
            sources = [s.name for s in op.sources]
            dests = [d.name for d in op.dests]
            op_str = {"op_type": op.op_type,
                      "op_id": op.global_op_id,
                      "operation_name": op.op_name,
                      "target": op.target,
                      "sources": sources,
                      "destinations": dests
                      }
        return op_str

    def emit_config_op(self, op, output_type):
        if output_type == "operations":
            op_str = f"{op.op_str}: {op.start_or_finish}-{op.target}"
        else:
            assert output_type == "json"
            op_str = {"op_type": op.op_type,
                      "op_id": op.global_op_id,
                      "start_or_finish": op.start_or_finish,
                      "target": op.target}
        return op_str

    def emit_transfer_op(self, op, output_type):
        if output_type == "operations":
            op_str = f"{op.op_str}: OPERAND: {op.operand.name}[{'->'.join(op.path)}], SIZES: {op.sizes}"
        else:
            assert output_type == "json"
            transfer_info = {}
            for move in op.operand.data_moves:
                text_key = f"{move.src_node}->{move.dst_node}"
                transfer_info[text_key] = {}
                transfer_info[text_key]['size'] = move.size()
                transfer_info[text_key]['offset'] = [str(o) for o in move.domain_offsets()]

            op_str = {"op_type": op.op_type,
                      "op_id": op.global_op_id,
                      "operand": op.operand.name,
                      "transfer_path": op.path,
                      "transfers": transfer_info}
        return op_str

    def emit_loop_op(self, op, output_type):
        if output_type == "operations":
            op_str = f"{op.op_str}[{op.loop_level}]: START={op.start}; STOP={op.end}; STRIDE={op.stride}; OFFSET:{op.offset}"
        else:
            assert output_type == "json"
            op_str = {"op_type": op.op_type,
                      "op_id": op.global_op_id,
                      "start": op.start,
                      "end": op.end,
                      "offset": op.offset,
                      "stride": op.stride
                      }
        return op_str

    def emit_operation(self, op: Operation, output_type):
        if output_type not in ["json", "operations"]:
            op_str = []
            for ft in op.instructions:
                ft_out = ft.emit(output_type)
                if len(ft_out) == 0:
                    continue
                op_str += ft_out
        elif op.op_type == "compute":
            op_str = self.emit_compute_op(op, output_type)
        elif op.op_type == "transfer":
            op_str = self.emit_transfer_op(op, output_type)
        elif op.op_type == "config":
            op_str = self.emit_config_op(op, output_type)
        else:
            assert op.op_type == "loop"
            op_str = self.emit_loop_op(op, output_type)
        return op_str

    def emit_single_codelet(self, cdlt, output_type):

        if output_type in ["operations", "operations_idx"]:
            input_str = ", ".join([f"{i.name}{i.shape_list}" for i in cdlt.inputs])
            out_str = ", ".join([f"{o.name}{o.shape_list}" for o in cdlt.outputs])
            operand_str = f"inputs={input_str}\n" \
                          f"outputs={out_str}\n"
            op_str = f"// CODELET:\t{cdlt.op_name}{cdlt.instance_id}\n"
            op_str += operand_str

            for i, o in enumerate(cdlt.ops):
                ostr = f"\t" * (o.loop_level + 1)
                if output_type == "operations_idx":
                    ostr = f"{i}" + f"{ostr}"
                ostr += f"{o.emit('operations')}\n"
                op_str += ostr
        elif output_type == "json":
            op_params = {}
            operand_dim_map = cdlt.operand_dim_mapping()
            for k, v in cdlt.required_params.items():
                if k not in operand_dim_map:
                    assert isinstance(v, FlexParam)
                    op_params[k] = v.value

            op_str = {}
            loop_order = cdlt.get_loop_order()
            op_str['operation'] = cdlt.op_name
            op_str['instance_id'] = cdlt.instance_id
            op_str['iterable_dimensions'] = {k: operand_dim_map[k] for k in loop_order}
            op_str['operation_parameters'] = op_params
            op_str['inputs'] = [i.emit(output_type) for i in cdlt.inputs]
            op_str['outputs'] = [o.emit(output_type) for o in cdlt.outputs]
            op_str['operation_sequence'] = [op.emit(output_type) for op in cdlt.ops]
        elif output_type == "json_no_ops":
            op_params = {}
            operand_dim_map = cdlt.operand_dim_mapping()

            loop_order = cdlt.get_loop_order()

            for k, v in cdlt.required_params.items():
                if k not in operand_dim_map:
                    assert isinstance(v, FlexParam)
                    op_params[k] = v.value

            op_str = {}
            op_str['operation'] = cdlt.op_name
            op_str['instance_id'] = cdlt.instance_id
            op_str['iterable_dimensions'] = {k: operand_dim_map[k] for k in loop_order}
            op_str['operation_parameters'] = op_params
            op_str['inputs'] = [i.emit("json") for i in cdlt.inputs]
            op_str['outputs'] = [o.emit("json") for o in cdlt.outputs]

        elif output_type not in ["decimal", "binary"]:
            op_str = f""
            for o in cdlt.ops:
                instr_list = self.emit_operation(o, output_type)
                if len(instr_list) > 0:
                    instr_list = f"\n".join(instr_list) + "\n"
                    op_str += instr_list
        else:
            op_str = []
            for o in cdlt.ops:
                instr_list = self.emit_operation(o, output_type)
                op_str += instr_list
            op_str = "\n".join(op_str)
        return op_str

    def emit_codelets_reevaluate(self, output_type):
        codelet_strings = []
        for c in self.codelets:
            # Emit codelet start
            if self.hag.has_op_template("codelet", "start"):
                cdlt_start = self.hag.get_cdlt_op_template("start")
                for ft in cdlt_start.instructions:
                    codelet_strings += ft.emit(output_type)
            codelet_strings.append(self.emit_single_codelet(c, output_type))

            # Emit codelet end
            if self.hag.has_op_template("codelet", "end"):
                cdlt_end = self.hag.get_cdlt_op_template("end")
                for ft in cdlt_end.instructions:
                    codelet_strings += ft.emit(output_type)
        return codelet_strings

    def emit_codelets(self, output_type):
        codelet_strings = []
        for c in self.codelets:
            if output_type in ["string_final", "decimal", "binary"]:
                instr = self.codelet_instructions[c.cdlt_uid]
                if len(instr) == 0:
                    continue
                # Change the output type for "emit" to output_type once finalized
                instr_str = [i.emit(output_type) for i in instr]
                codelet_strings.append(f"\n".join(instr_str) + "\n")
            else:
                codelet_strings.append(self.emit_single_codelet(c, output_type))

        return codelet_strings

    def emit(self, output_type: str):
        program_strings = []

        if self.hag.has_op_template("program", "start"):
            start_instrs = self.hag.get_program_template("start")
            for ft in start_instrs.instructions:
                program_strings += ft.emit(output_type)

        program_strings += self.emit_codelets(output_type)

        if self.hag.has_op_template("program", "end"):
            end_instrs = self.hag.get_program_template("end")
            for ft in end_instrs.instructions:
                program_strings += ft.emit(output_type)

        if output_type not in ["json", "json_no_ops"]:
            return "\n".join(program_strings)
        else:
            res = {"mode": self.program_mode, "program": program_strings}
            res = json.loads(json.dumps(res, cls=CodeletJSONEncoder))
            return res

    def instantiate_instructions_templates(self, node, cdlt):
        self.set_instruction_templates(cdlt)
        self.relocatables.add_data_relocation(node, cdlt)
        self.instantiate_instructions(cdlt)

        if self.hag.has_op_template("codelet", "start"):
            cdlt_start = self.hag.get_cdlt_op_template("start")
            args = (self, self.hag, -1, cdlt.instance_id)
            for ft in cdlt_start.instructions:
                ft.evaluate(*args)

        if self.hag.has_op_template("codelet", "end"):
            cdlt_end = self.hag.get_cdlt_op_template("end")
            args = (self, self.hag, -1, cdlt.instance_id)
            for ft in cdlt_end.instructions:
                ft.evaluate(*args)

    def evaluate_lazy_instruction_templates(self, cdlt):
        for o in cdlt.ops:
            args = (self, self.hag, o.global_op_id, cdlt.instance_id)
            for ft in o.instructions:
                ft.lazy_evaluate(*args)

    def add_operand_mapping(self, cdlt: Codelet, parent_node: pm.Node, node: pm.Node, operand: Operand, operand_type):

        if node.name not in self.operand_mapping:
            self.operand_mapping[node.name] = OperandDataflow(node.name, node.__class__.__name__)

        if operand_type == "output":
            self.operand_mapping[node.name].add_write(cdlt, operand)
        else:
            if node.op_name not in ["state", "input"] and len(self.operand_mapping[node.name].cdlt_write) == 0 and \
                    node not in parent_node.inputs:
                if node.graph is None:
                    raise RuntimeError
                node_dfgs = node.name.split("/")
                if len(node_dfgs) > 1:
                    node_dfgs = node.name.split("/")
                    if node_dfgs[-1] in self.operand_mapping:
                        self.operand_mapping[node.name] = self.operand_mapping[node_dfgs[-1]]
                else:
                    raise RuntimeError(f"No write node for {parent_node.op_name}/{parent_node.name}: {node.name} - {node.op_name} - {cdlt.op_name} - Graph: {node.graph}")

            self.operand_mapping[node.name].add_read(cdlt, operand)

    def create_cdlt_dfg(self):
        dfg = nx.DiGraph()

        for cdlt in self.codelets:
            dfg.add_node(cdlt.instance_id, label=f"op[{cdlt.op_name}{cdlt.instance_id}]", color="blue")

        for node_name, dataflow in self.operand_mapping.items():
            dfg.add_node(node_name, label=f"data[{node_name}]", color="red", operand_type=dataflow.node_type)

            for rcdlt in dataflow.cdlt_read:
                dfg.add_edge(node_name, rcdlt)

            for wcdlt in dataflow.cdlt_write:
                dfg.add_edge(wcdlt, node_name)

        return dfg

    def check_connectivity(self):
        for node_name, dataflow in self.operand_mapping.items():
            if len(dataflow.cdlt_write) == 0 and dataflow.node_type not in ["state", "input"]:
                print(f"Initializer: {dataflow.node_name} - {dataflow.node_type}")

    # TODO: Fix these
    def save_json(self, full_path):
        json_blob = []
        with open(full_path, 'w') as outfile:
            json.dump(json_blob, outfile, indent=4)

    def save_text(self, full_path):
        instructions = []
        instructions = "\n".join(instructions)
        with open(full_path, 'w') as outfile:
            outfile.write(instructions)

    def sequence_nodes(self, sequence_algorithm, validate_lowered=True, verbose=False, **sequence_kwargs):
        # TODO: Add support for different sequencing algos

        if verbose:
            print(f"Sequencing nodes")

        start = time()
        node_list = []
        all_ops = []
        missing_ops = []
        if sequence_algorithm == "default":
            for name, node in self.graph.nodes.items():
                if not isinstance(node, (pm.write, pm.placeholder)) and node.op_name not in all_ops:
                    all_ops.append(node.op_name)
                if self.hag.has_codelet(node.op_name):
                    node_list.append(node)
                elif not isinstance(node, (pm.write, pm.placeholder)) and node.op_name not in missing_ops:
                    missing_ops.append(node.op_name)

            if len(missing_ops) > 0 and validate_lowered:
                raise RuntimeError(
                    f"Input graph includes operations which are unsupported by the target architecture.\n"
                    f"Unsupported Operations: {[mo for mo in missing_ops]}\n"
                    f"HAG-supported Operations: {self.hag.all_codelet_names}")
        elif sequence_algorithm == 'filtered':
            assert 'filtered_layers' in sequence_kwargs
            filter_layers = sequence_kwargs['filtered_layers']
            layers_found = {k: False for k in filter_layers}
            for name, node in self.graph.nodes.items():
                if not isinstance(node, (pm.write, pm.placeholder)) and node.op_name not in all_ops:
                    all_ops.append(node.op_name)
                if node.op_name in filter_layers and self.hag.has_codelet(node.op_name) and not layers_found[node.op_name]:
                    node_list.append(node)
                    layers_found[node.op_name] = True
                elif not isinstance(node, (pm.write, pm.placeholder)) and node.op_name not in missing_ops:
                    missing_ops.append(node.op_name)
            print(f"Skipping {missing_ops}")
        else:
            raise RuntimeError(f"{sequence_algorithm} is not a valid sequencing algorithm")

        if verbose:
            print(f"Sequencing took {time() - start} seconds")

        return node_list

    def store_tiling(self, path):
        tiling_info = {}
        for c in self.codelets:
            tiling_info[f"{c.op_name}{c.instance_id}"] = c.domain_tiling

        with open(f"{path}/{self.name}_tiling_info.json", "w") as outfile:
            json.dump(tiling_info, outfile, indent=4)


    def load_tiling(self, filename):

        with open(f'{filename}') as f:
            tiling = json.load(f)

        for c in self.codelets:
            tile_key = f"{c.op_name}{c.instance_id}"
            c._domain_tiling = {}

            if tile_key not in tiling:
                print(f"{tile_key} not found in tiling. Leaving tiling to empty")
            else:
                for level, tiling_values in tiling[tile_key].items():
                    c._domain_tiling[int(level)] = tiling_values

    def get_required_templates(self, nodes: List[pm.Node]):
        node_ops = list(set([n.op_name for n in nodes]))
        cdlt_templates = {name: self.hag.get_codelet_template(name) for name in node_ops}
        return cdlt_templates

    def run_template_stages(self, node_sequence, verbose=False):

        if verbose and len(self.template_stages.keys()) > 0:
            print(f"Running template stages")

        self._codelet_templates = self.get_required_templates(node_sequence)
        for level, fns in self.template_stages.items():
            for template_name in list(self.codelet_templates.keys()):
                cdlt_tmplt = self.codelet_templates[template_name]
                for fn in fns:
                    cdlt_tmplt = fn.run(self, cdlt_tmplt)
                    if not isinstance(cdlt_tmplt, CodeletTemplate):
                        raise RuntimeError(f"Compilation stage {fn.name}"
                                           f" does not return codelet template.")
                self.codelet_templates[template_name] = cdlt_tmplt

    def run_preprocessing_stages(self, node_sequence, codelets, verbose=False):
        if verbose:
            print(f"\nRunning Preprocessing functions")

        stage_start = time()
        for level, fns in self.preproc_stages.items():
            for n in node_sequence:
                cdlt = codelets[n.name]
                for fn in fns:
                    if cdlt.is_noop() and fn.skip_noops:
                        if verbose:
                            print(f"Skipping NOOP {cdlt.op_name}")
                        continue
                    if verbose:
                        print(f"Preprocessing with {fn.name} on codelet {cdlt.op_name}{cdlt.instance_id}")
                    cdlt = fn.run(self, n, cdlt)

                assert n.name in codelets and codelets[n.name].instance_id == cdlt.instance_id
                codelets[n.name] = cdlt

        if verbose:
            print(f"\nPreprocessing took {time() - stage_start} seconds")

        return codelets

    def instantiate_all_codelets(self, node_sequence, verbose=False):
        codelets = {}

        stage_start = time()
        if verbose:
            print(f"\nInstantiating codelets")

        for n in node_sequence:
            if verbose:
                print(f"Instantiating {n.op_name}")

            cdlt = self.instantiate_codelet(n)
            assert n.name not in codelets
            codelets[n.name] = cdlt

        if verbose:
            print(f"\nInstantiating codelets took {time() - stage_start} seconds")

        return codelets

    def instantiate_all_operations(self, node_sequence, codelets, verbose=False):
        if verbose:
            print(f"\nInstantiating Codelet Operations")

        stage_start = time()
        for n in node_sequence:
            cdlt = codelets[n.name]
            if cdlt.is_noop():
                if verbose:
                    print(f"Skipping NOOP codelet {cdlt.op_name}{cdlt.instance_id}")
                continue
            if verbose:
                print(f"Instantiating codelet {cdlt.op_name}{cdlt.instance_id}")
            cdlt.instantiate_operations(n, self.hag)
            # TODO: Check if certain optimizations are necessary
            codelets[n.name] = cdlt

        if verbose:
            print(f"\nCodelet instantiation took {time() - stage_start}")
        return codelets

    def run_compilation_stages(self, node_sequence, codelets, verbose=False):
        if verbose:
            print(f"\nRunning compilation stages")

        stage_start = time()
        for level, fns in self.compilation_pipeline.items():
            for n in node_sequence:
                cdlt = codelets[n.name]
                for fn in fns:
                    if cdlt.is_noop() and fn.skip_noops:
                        if verbose:
                            print(f"Skipping NOOP codelet {cdlt.op_name}{cdlt.instance_id}")
                        continue
                    if verbose:
                        print(f"Applying stage {fn.name} on codelet {cdlt.op_name}{cdlt.instance_id}")
                    cdlt = fn.run(self, n, cdlt)
                codelets[n.name] = cdlt

        if verbose:
            print(f"\nCompilation stages took {time() - stage_start} seconds")

        return codelets

    def finalize_instructions(self, node_sequence, codelets, verbose=False):
        if verbose:
            print(f"\nFinalizing instruction templates")

        if self.hag.has_op_template("program", "start"):
            start_instrs = self.hag.get_program_template("start")
            args = (self, self.hag, -1, -1)
            for ft in start_instrs.instructions:
                ft.evaluate(*args)

        for n in node_sequence:
            cdlt = codelets[n.name]
            if codelets[n.name].is_noop():
                if verbose:
                    print(f"Skipping NOOP codelet {cdlt.op_name}{cdlt.instance_id}")
                continue
            if verbose:
                print(f"Instantiating template for {cdlt.op_name}{cdlt.instance_id}")
            self.instantiate_instructions_templates(n, codelets[n.name])

        # Generate program end, if it exists
        if self.hag.has_op_template("program", "end"):
            end_instrs = self.hag.get_program_template("end")
            args = (self, self.hag, -1, -1)
            for ft in end_instrs.instructions:
                ft.evaluate(*args)

    def finalize_instruction_memory(self, node_sequence, codelets, verbose=False):
        if verbose:
            print(f"Finalizing Instruction memory")

        for n in node_sequence:
            cdlt = codelets[n.name]
            if codelets[n.name].is_noop():
                if verbose:
                    print(f"Skipping NOOP codelet {cdlt.op_name}{cdlt.instance_id}")
                continue
            self.relocatables.update_relocation_offset('INSTR_MEM',
                                                       cdlt.cdlt_uid,
                                                       cdlt.num_instr * self.hag.instr_length)

    def finalize_flex_params(self, node_sequence, codelets, verbose=False):
        if verbose:
            print(f"\nEvaluating post-processed FlexParams")

        for n in node_sequence:
            cdlt = codelets[n.name]
            if codelets[n.name].is_noop():
                if verbose:
                    print(f"Skipping NOOP codelet {cdlt.op_name}{cdlt.instance_id}")
                continue

            if verbose:
                print(f"Evaluating lazy FlexParams for {cdlt.op_name}{cdlt.instance_id}")
            self.evaluate_lazy_instruction_templates(codelets[n.name])

    def run_instruction_stages(self, codelets, verbose=False):
        if verbose and len(self.instruction_stages.keys()) > 0:
            print(f"Running instruction stages")

        for name, cdlt in codelets.items():
            instr = []
            if self.hag.has_op_template("codelet", "start"):
                cdlt_start = self.hag.get_cdlt_op_template("start")
                for ft in cdlt_start.instructions:
                    instr += ft.instructions

            for o in cdlt.ops:
                for ft in o.instructions:
                    instr += ft.instructions

            if self.hag.has_op_template("codelet", "end"):
                cdlt_end = self.hag.get_cdlt_op_template("end")
                for ft in cdlt_end.instructions:
                    instr += ft.instructions
            self.codelet_instructions[cdlt.cdlt_uid] = instr
        finalized_codelets = {}

        stage_start = time()
        for level, fns in self.instruction_stages.items():
            for name, cdlt in codelets.items():
                for fn in fns:
                    if cdlt.is_noop() and fn.skip_noops:
                        if verbose:
                            print(f"Skipping NOOP {cdlt.op_name}")
                        continue
                    if verbose:
                        print(f"Preprocessing with {fn.name} on codelet {cdlt.op_name}{cdlt.instance_id}")
                    assert cdlt.cdlt_uid in self.codelet_instructions
                    cdlt = fn.run(self, self.codelet_instructions[cdlt.cdlt_uid])

                finalized_codelets[name] = cdlt

        if verbose:
            print(f"\nPreprocessing took {time() - stage_start} seconds")

        return codelets

    def compile(self, verbose=False, sequence_algorithm="default", tiling_path=None,
                finalize=True,
                **compile_kwargs):
        # This function performs breadth-first compilation, with coarsest abstractions first:
        # 1. Generate codelets from nodes
        # 2. Generate operands/operations within codelets
        # 3. Generate instruction templates within operations
        start = time()
        node_sequence = self.sequence_nodes(sequence_algorithm, verbose=verbose, **compile_kwargs)
        self.run_template_stages(node_sequence, verbose=verbose)

        codelets = self.instantiate_all_codelets(node_sequence, verbose=verbose)

        if tiling_path is not None:
            if verbose:
                print(f"\nLoading predefined tiling at {tiling_path}")
            self.load_tiling(tiling_path)
        codelets = self.run_preprocessing_stages(node_sequence, codelets, verbose=verbose)
        codelets = self.instantiate_all_operations(node_sequence, codelets, verbose=verbose)

        codelets = self.run_compilation_stages(node_sequence, codelets, verbose=verbose)

        if finalize:
            self.finalize_instructions(node_sequence, codelets, verbose=verbose)
            self.finalize_instruction_memory(node_sequence, codelets, verbose=verbose)
            self.finalize_flex_params(node_sequence, codelets, verbose=verbose)
            self.run_instruction_stages(codelets, verbose=verbose)

        if verbose:
            print(f"\nTotal compilation time was {time() - start} seconds")


def generate_possible_tilings(shape_dict, memory_paths):
    possible_tilings = {}
    for k, v in shape_dict.items():
        tile_permutations = []

def tiling_constraint(shapes, node_capacities, tile_sizes):

    for i, t in enumerate(tile_sizes):
        data_size = np.prod([t[s] for s in shapes])
        if data_size >= node_capacities[i]:
            return False
    return True

class CodeletJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, Basic):
            return str(o)
        return json.JSONEncoder.default(self, o)