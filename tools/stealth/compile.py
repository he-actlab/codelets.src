import os
import polymath as pm
import json
from stealth.stealth_codelet import StealthCodelet, StealthVariableName, build_codelet_template, collect_tiling
from codelets.examples.genesys import compile_genesys, load_config, DataGen
from stealth.utils import UniqueNameGenerator


class DummyTemplate(pm.Template):
    def __init__(self, *args, **kwargs):
        assert "number_of_inputs" in kwargs
        assert "number_of_outputs" in kwargs
        assert "custom_operation_name" in kwargs
        self._number_of_inputs = kwargs["number_of_inputs"]
        self._number_of_outputs = kwargs["number_of_outputs"]
        custom_operation_name = kwargs["custom_operation_name"]
        kwargs.pop("number_of_inputs")
        kwargs.pop("number_of_outputs")
        kwargs.pop("custom_operation_name")
        super().__init__(*args, **kwargs)
        self.op_name =custom_operation_name
    
    def define_graph(self, *args):
        pass

    @property
    def inputs(self):
        return tuple(self.args[0][i] for i in range(self._number_of_inputs))

    @property
    def outputs(self):
        return tuple(self.args[0][self._number_of_inputs + i] for i in range(self._number_of_outputs))


def _create_dummy_polymath_node_from_codelet(codelet: StealthCodelet, dimension_sizes: dict[str, int]) -> pm.Graph:
    unique_name_generator = UniqueNameGenerator()
    with pm.Node(name="test") as graph:
        top_inputs = []
        for operand in codelet.inputs:
            input_name = unique_name_generator.get_unique_name("input")
            top_inputs.append(pm.input(name=input_name, shape=tuple(dimension_sizes[s.name] if isinstance(s, StealthVariableName) else s.value for s in operand.shape)))
        top_outputs = []
        for output_operand in codelet.outputs:
            output_name = unique_name_generator.get_unique_name("output")
            top_outputs.append(pm.output(name=output_name, shape=tuple(dimension_sizes[s.name] if isinstance(s, StealthVariableName) else s.value for s in output_operand.shape)))
        
        args = top_inputs + top_outputs
        DummyTemplate(*args, number_of_inputs=len(top_inputs), number_of_outputs=len(top_outputs), custom_operation_name=codelet.operation_name) 
    return graph


def write_tiling(operation_name: str, tiling_path: str, tiling: dict[str, int]) -> None:
    if not os.path.exists(os.path.dirname(tiling_path)):
        os.makedirs(os.path.dirname(tiling_path))
    with open(tiling_path, "w") as f:
        file_contents = {operation_name + "1": {"1": tiling}}
        json.dump(file_contents, f, indent=4)


def compile(config_path: str, layer: StealthCodelet, dimension_sizes: dict[str, int], thread_id: int = 0) -> None:
    graph = _create_dummy_polymath_node_from_codelet(layer, dimension_sizes)
    cfg = load_config(config_path)
    tiling: dict[str, int] = collect_tiling(layer)
    write_tiling(layer.operation_name, f"stealth_outputs/tiling_info/tiling_{thread_id}.json", tiling)
    program = compile_genesys(
        model_name=layer.operation_name,
        graph=graph,
        genesys_cfg=cfg,
        custom_codelets={layer.operation_name: build_codelet_template(layer)},
        print_config=False,
        benchmark_path="stealth_outputs",
        # store_tiling=True,
        tiling_path=f"tiling_{thread_id}.json"
    )

    sys_array_size = cfg['ARRAY_M']
    dgen = DataGen(program,
                    single_codelets=False,
                    shared_datagen=False,
                    dir_ext=f"benchmark{sys_array_size}x{sys_array_size}_{thread_id}",
                    identifier="stealth",
                    generate_data=False,
                    verbose=False,
                    out_path=f"stealth_outputs/compilation_output",
                    store_whole_program=False)
    dgen.generate()
