
import polymath as pm
from codelets.micro_templates import ComputeTemplate
from codelets.codelet_template import CodeletTemplate
from codelets.adl.graph import ArchitectureNode, StorageNode

def identify_operand_targets(cdlt: CodeletTemplate, hag: ArchitectureNode):
    start_storage = hag.get_off_chip_storage()
    assert isinstance(start_storage, StorageNode)

    for o in cdlt.operands:
        if not o.is_location_set():
            o.set_location(start_storage.name)
        else:
            assert o.location == start_storage.name

    for o in cdlt.ops:
        if isinstance(o, ComputeTemplate):
            options = hag.primitive_targets(o.op_name)
            if len(options) == 0:
                raise RuntimeError(f"{o.op_name} is not a supported operation for the target architecture")
            elif o.is_target_set():
                assert o.target in options
            elif len(options) == 1:
                o.set_parameter('target', options[0])
            else:
                o.set_param_options('target', options)

    return cdlt

def collect_unset_paths(cdlt: CodeletTemplate, hag: ArchitectureNode):
    unset_paths = {'in': {},
                   'out': {}}

    # for i in cdlt.inputs:


