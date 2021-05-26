from codelets.codelet_template import CodeletTemplate
from codelets.adl.graph import ArchitectureNode


def set_targets(cdlt: CodeletTemplate, hag: ArchitectureNode):
    for o in cdlt.ops:
        if o.op_type == "compute":
            # Search which target supports the current op
            compute_ops = hag.operation_mappings['compute']
            for target in compute_ops.keys():
                if o.op_name in compute_ops[target]:
                    o.target = target
                    break
            if not o.is_target_set():
                raise RuntimeError(f"Unable to find target supporting op {o.op_name} in model")
