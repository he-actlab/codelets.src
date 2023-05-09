from typing import Dict
from .codelets.reference_impls.ref_op import OperandData
from codelets.codelet_impl import Codelet
from codelets.compiler.program import CodeletProgram
from collections import defaultdict
import numpy as np
import os
from pathlib import Path
import json
from .genesys import get_arch
import git
BENCH_BASE_ADDR = {"INSTR": 0, "OBUF": 0, "BBUF": 4096, "WBUF": 24576, "IBUF": 4259840}



def save_array(path, data):
    with open(path, 'w') as f:
        f.write('\n'.join([str(i) for i in data.flatten().tolist()]))

OUTPUT_TYPES = ["arch_cfg", "operations_idx", "json", "string_final", "decimal", "binary"]
OUT_DIR = Path(f"{Path(__file__).parent}/../../tools/compilation_output")

class DataGen(object):
    def __init__(self, program,
                 single_codelets=True,
                 dir_ext=None,
                 identifier=None,
                 output_types=None,
                 generate_data=False,
                 verbose=False,
                 store_partials=False,
                 store_whole_program=False,
                 shared_datagen=False,
                 propagate_outputs=False,
                 print_datagen_range=False,
                 out_path=None,
                 datagen_vrange=None):
        self.print_datagen_range = print_datagen_range
        self.datagen_vrange = datagen_vrange
        self.store_whole_program = store_whole_program
        self.shared_datagen = shared_datagen
        self._storage_info = {}
        self._propagate_outputs = propagate_outputs
        self._program = program
        self._inouts = {"inputs": [], "outputs": []}
        self._value_dict : Dict[str, Dict[str, OperandData]] = {"inputs": {},
                  "intermediate": {},
                  "outputs": {}}
        self._store_partials = store_partials
        self._single_codelets = single_codelets
        self._verbose = verbose
        self._generate_data = generate_data
        self.output_types = output_types
        if self.output_types is not None:
            assert all([o in OUTPUT_TYPES for o in output_types])
        else:
            self.output_types = OUTPUT_TYPES
        if out_path is not None:
            output_dir = f"{out_path}/{program.name}"
        else:
            output_dir = f"{OUT_DIR}/{program.name}"
        if dir_ext:
            output_dir = f"{output_dir}_{dir_ext}"
        self.output_dir = f"{output_dir}_{identifier}"
        self.arch_cfg = get_arch(None, self.program.hag.meta_cfg, None)

        if not Path(self.output_dir).exists():
            try:
                os.makedirs(self.output_dir)
            except OSError as e:
                print(f"Creation of directory {self.output_dir} failed:\n {e}")
            else:
                print(f"Successfully created of directory {self.output_dir}")

    @property
    def single_codelets(self):
        return self._single_codelets

    @property
    def storage_info(self):
        return self._storage_info

    @property
    def propagate_outputs(self):
        return self._propagate_outputs

    @property
    def program(self) -> CodeletProgram:
        return self._program

    @property
    def verbose(self):
        return self._verbose

    @property
    def generate_data(self):
        return self._generate_data

    @property
    def value_dict(self):
        return self._value_dict

    @property
    def inouts(self):
        return self._inouts

    def initialize_value_dict(self, cdlt):
        inouts = {"inputs": [], "outputs": []}
        for i in cdlt.inputs:
            if i.node_name in self.value_dict['outputs']:
                operand = self.value_dict['outputs'].pop(i.node_name)
                assert isinstance(operand, OperandData), f"Not operand: {operand.name}"
                assert operand.data.shape == i.shape, f"Operand and input shapes are not equal in {cdlt.cdlt_uid}:\n" \
                                                      f"Data: {operand.data.shape}\n" \
                                                      f"Operand: {i.shape}"
                inouts['inputs'].append(operand)
                self.value_dict['intermediate'][i.node_name] = operand
            elif i.node_name in self.value_dict['intermediate']:
                operand = self.value_dict['intermediate'][i.node_name]
                assert operand.data.shape == i.shape, "Operand and input shapes are not equal:\n" \
                                                      f"Data: {operand.data.shape}\n" \
                                                      f"Operand: {i.shape}"
                inouts['inputs'].append(operand)
        return inouts


    def store_inputs(self, base_path, inouts):
        # for n, i in self.value_dict['inputs'].items():
        for inp in inouts['inputs']:
            # node_name = i.node_name.replace("/", "_")
            if inp.node_name in self.value_dict['inputs']:
                i = self.value_dict['inputs'][inp.node_name]
            elif inp.node_name in self.value_dict['intermediate']:
                i = self.value_dict['intermediate'][inp.node_name]
            else:
                raise RuntimeError(f"No value found found for cdlt operand: {inp.node_name}\n" \
                                                   f"UID: {inp.node_name}\n" \
                                                   f"Value inputs: {list(self.value_dict['inputs'].keys())}\n"
                                   f"Value intermediates: {list(self.value_dict['intermediate'].keys())}\n")

            assert isinstance(i, OperandData)
            assert isinstance(i.data, np.ndarray)

            node_name = i.node_name

            assert node_name in self.storage_info, f"No storage info found for cdlt operand: {node_name}\n" \
                                                   f"UID: {i.node_name}\n" \
                                                   f"Storage keys: {list(self.storage_info.keys())}"

            if Path(f"{base_path}/{node_name}").exists():
                save_array(f'{base_path}/{node_name}/{node_name}.txt', i.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}/'
            else:
                save_array(f'{base_path}/{node_name}.txt', i.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}.txt'

    def initialize_storage(self, cdlt, inouts):
        formatted = []
        assert all([isinstance(i, OperandData) for i in inouts['inputs']])
        assert all([isinstance(o, OperandData) for o in inouts['outputs']])
        for i in inouts['inputs']:
            if i.fmt is None and i.node_name not in self.value_dict['inputs'] and \
                    i.node_name not in self.value_dict['intermediate']:
                self.value_dict['inputs'][i.node_name] = i
                node_name = i.node_name
                self.storage_info[node_name] = {"cdlt": cdlt.cdlt_uid,
                                                "path": None,
                                                'cdlt_name': i.idx.name,
                                                'operand_type': 'input'}

            elif i.fmt is not None:
                formatted.append(i)
        return formatted

    def store_formatted(self, formatted, base_path):
        for f in formatted:
            if f.node_name in self.value_dict['inputs']:
                assert f.fmt is not None
                node_name = f.node_name
                if not Path(f"{base_path}/{node_name}").exists():
                    os.makedirs(f"{base_path}/{node_name}")
                save_array(f'{base_path}/{node_name}/{node_name}_{f.fmt}.txt', f.data)

    def store_outputs(self, cdlt, inouts, base_path):

        for idx in range(len(inouts['outputs'])):
            o = inouts['outputs'][idx]
            if o.fmt is None:
                node_name = o.node_name
                assert isinstance(o, OperandData)
                assert isinstance(o.data, np.ndarray)
                if o.data.dtype != np.int64:
                    # o.data = o.data.astype(np.int64)
                    o = o._replace(data=o.data.astype(np.int64))
                    inouts['outputs'][idx] = o

                self.value_dict['outputs'][node_name] = o
                self.storage_info[node_name] = {"cdlt": cdlt.cdlt_uid,
                                             "path": None,
                                             'cdlt_name': o.idx.name,
                                             'operand_type': 'output'}



        for o in inouts['outputs']:

            node_name = o.node_name
            assert node_name in self.storage_info, f"No storage info found for output cdlt operand: {node_name}\n" \
                                                   f"UID: {o.node_name}\n" \
                                                   f"Storage keys: {list(self.storage_info.keys())}"

            if Path(f"{base_path}/{node_name}").exists():
                save_array(f'{base_path}/{node_name}/{node_name}.txt', o.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}/'
            else:
                save_array(f'{base_path}/{node_name}.txt', o.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}.txt'

        if 'csv_data' in inouts:
            partials = inouts.pop('csv_data')
            with open(f"{base_path}/debug_coords.csv", 'w') as f:
                f.write(f'N, (m/n/p), O_idx, I_idx, W_idx, I_val, W_val, partial\n')

                for k, v in partials.items():
                    for l in v:
                        f.write(f"N={k}, " + "," + l + "\n")


    def generate_cdlt_data(self, cdlt: Codelet, base_path):
        inouts = self.initialize_value_dict(cdlt)
        opgen = self.program.metadata['GENESYS_IMPLS'][cdlt.op_name](cdlt, self.program)
        inouts = opgen.compute_outputs(inouts, print_range=self.print_datagen_range, vrange=self.datagen_vrange)
        formatted = self.initialize_storage(cdlt, inouts)
        self.store_inputs(base_path, inouts)
        self.store_outputs(cdlt, inouts, base_path)
        self.store_formatted(formatted, base_path)


        with open(f"{base_path}/data_locations.json", "w") as outf:
            outf.write(json.dumps(self.storage_info, indent=2))

        if self.single_codelets:
            self.storage_info.clear()
            self.reset_value_dict()

        assert all([isinstance(i.data, np.ndarray) for i in self.value_dict['inputs'].values()])
        assert all([isinstance(o.data, np.ndarray) for o in self.value_dict['outputs'].values()])
        assert all([isinstance(i.data, np.ndarray) for i in self.value_dict['intermediate'].values()])

    def reset_value_dict(self):
        self._value_dict: Dict[str, Dict[str, OperandData]] = {"inputs": {},
                                                               "intermediate": {},
                                                               "outputs": {}}

    def generate_codelet_data(self):
        for layer_id, cdlt in enumerate(self.program.codelets):
            if cdlt.is_noop():
                if self.verbose:
                    print(f"Skipping generation for {cdlt.cdlt_uid}")
                continue
            if self.verbose:
                print(f"Storing codelet {cdlt.cdlt_uid}")
            output_location = f"{self.output_dir}/layer{layer_id}_{cdlt.cdlt_uid}"
            if not Path(output_location).exists():
                try:
                    os.makedirs(output_location)
                except OSError as e:
                    raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")

            if 'operations_idx' in self.output_types:
                otype = 'operations_idx'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'string_final' in self.output_types:
                otype = 'string_final'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'decimal' in self.output_types:
                otype = 'decimal'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'binary' in self.output_types:
                otype = 'binary'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'json' in self.output_types:
                otype = 'json'
                ext = 'json'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                res = json.dumps(res, indent=2)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if self.generate_data:
                base_path = f"{output_location}/data"
                if not Path(base_path).exists():
                    try:
                        os.makedirs(base_path)
                    except OSError as e:
                        raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")

                self.generate_cdlt_data(cdlt, base_path)
                if self.verbose:
                    print(f"Generating data to be stored in {base_path}")




    def store_program(self):
        if self.verbose:
            print(f"Storing program {self.program.name}")
        output_location = f"{self.output_dir}/program"
        if not Path(output_location).exists():
            try:
                os.makedirs(output_location)
            except OSError as e:
                raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")

        if 'operations_idx' in self.output_types:
            otype = 'operations_idx'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'string_final' in self.output_types:
            otype = 'string_final'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'decimal' in self.output_types:
            otype = 'decimal'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'binary' in self.output_types:
            otype = 'binary'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'json' in self.output_types:
            otype = 'json'
            ext = 'json'
            res = self.program.emit(otype)
            res = json.dumps(res, indent=2)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if self.generate_data and self.shared_datagen:
            base_path = f"{output_location}/data"
            if not Path(base_path).exists():
                try:
                    os.makedirs(base_path)
                except OSError as e:
                    raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")
            self.generate_whole_program_data()
        elif self.generate_data:
            base_path = f"{output_location}/data"
            if not Path(base_path).exists():
                try:
                    os.makedirs(base_path)
                except OSError as e:
                    raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")
            self.generate_whole_program_data_unshared()

    def generate_operand_storage_info(self, cdlt, operand, operand_type, offset, path):
        is_shuffled = False
        if Path(f"{path}/").exists():
            assert Path(f"{path}/").is_dir()
            path = path + f"/{operand.node_name}_shuffled.txt"
            is_shuffled = True
        elif Path(f"{path}.txt").exists():
            path = f"{path}.txt"
        else:
            raise RuntimeError(f"Path for codelet {cdlt.cdlt_uid}, {operand_type} {operand.node_name} ({operand.name}) does not exist:\n"
                               f"path: {path}")

        dtype = str(operand.dtype)

        assert 'DRAM' in operand.tiling, f"Operand {operand.node_name} is not tiled for DRAM in codelet {cdlt.cdlt_uid}"
        if is_shuffled:
            assert 'IBUF' in operand.tiling or 'WBUF' in operand.tiling
        dram_shape = operand.tiling['DRAM']
        operand_size = operand.dtype.bytes() * np.prod(list(dram_shape.values()))
        buffer = [k for k in operand.tiling.keys() if k not in ['DRAM', 'pe_array', 'SIMD']]
        assert len(buffer) > 0
        buffer = buffer[0]
        info_blob = {
            "path": path,
            "dtype": dtype,
            "layer": cdlt.cdlt_uid,
            "offset": offset,
            "shape": dram_shape,
            "size_in_bytes": operand_size,
            "buffer": buffer
        }
        return info_blob

    def generate_whole_program_data(self):
        last_cdlt_id = len(self.program.codelets)
        input_offset = self.program.get_instr_mem_end()

        output_offset = 0
        info_map = {"inputs": {}, "outputs": {}, "instructions": {}}

        def check_operand_info(prev_info, new_info, operand_type):
            pretty_prev = json.dumps(prev_info, indent=2, cls=NPEncoder)
            pretty_new = json.dumps(new_info, indent=2, cls=NPEncoder)
            if prev_info['path'] != new_info['path']:
                raise RuntimeError(f"Found previous {operand_type} with same name but different path:\n"
                                   f"Previous: {pretty_prev}\n"
                                   f"New: {pretty_new}")
            if prev_info['dtype'] != new_info['dtype']:
                raise RuntimeError(f"Found previous {operand_type} with same name but different datatypes:\n"
                                   f"Previous: {pretty_prev}\n"
                                   f"New: {pretty_new}")
            if prev_info['layer'] != new_info['layer']:
                raise RuntimeError(f"Found previous {operand_type} with same name but different layers:\n"
                                   f"Previous: {pretty_prev}\n"
                                   f"New: {pretty_new}")
            if prev_info['size_in_bytes'] != new_info['size_in_bytes']:
                raise RuntimeError(f"Found previous {operand_type} with same name but different sizes:\n"
                                   f"Previous: {pretty_prev}\n"
                                   f"New: {pretty_new}")

        for layer_id, cdlt in enumerate(self.program.codelets):
            base_cdlt_path = f"{self.output_dir}/layer{layer_id}_{cdlt.cdlt_uid}"
            cdlt_path = f"{base_cdlt_path}/data/"
            instr_offset = self.program.get_codelet_instr_offset(cdlt.instance_id)
            info_map['instructions'][cdlt.cdlt_uid] = {
                "path": f"{base_cdlt_path}/{cdlt.cdlt_uid}_string_final.txt",
                "bitwidth": 32,
                "size_in_bytes": self.program.cdlt_num_instr(cdlt) * 4,
                "offset": instr_offset
            }
            for i in cdlt.inputs:
                input_path = f"{cdlt_path}{i.node_name}"
                # offset = self.program.
                inp_offset = self.program.get_input_operand_offset(i)
                info_blob = self.generate_operand_storage_info(cdlt, i, "input", inp_offset, input_path)

                if i.node_name in info_map['inputs']:
                    check_operand_info(info_map['inputs'][i.node_name], info_blob, 'input')
                    continue
                info_map["inputs"][i.node_name] = info_blob
                input_offset += info_blob['size_in_bytes']

            for o in cdlt.outputs:
                output_path = f"{cdlt_path}{o.node_name}"
                out_offset = self.program.get_output_operand_offset(o)

                info_blob = self.generate_operand_storage_info(cdlt, o, "output", out_offset, output_path)

                if o.node_name in info_map['outputs']:
                    check_operand_info(info_map['outputs'][o.node_name], info_blob, 'output')
                    continue
                info_map["outputs"][o.node_name] = info_blob
                output_offset += info_blob['size_in_bytes']


        output_location = f"{self.output_dir}/program"
        if not Path(output_location).exists():
            raise RuntimeError("directory for storying whole program does not exist!")
        res = json.dumps(info_map, indent=2, cls=NPEncoder)
        with open(f"{output_location}/{self.program.name}_operand_storage_info.json", "w") as outfile:
            outfile.write(res)


    def generate_whole_program_data_unshared(self):
        last_cdlt_id = len(self.program.codelets)
        input_offset = self.program.get_instr_mem_end()

        output_offset = 0
        info_map = {}


        for layer_id, cdlt in enumerate(self.program.codelets):
            if cdlt.is_noop():
                continue
            layer_key = f"layer{layer_id}_{cdlt.cdlt_uid}"
            base_cdlt_path = f"{self.output_dir}/{layer_key}"
            cdlt_path = f"{base_cdlt_path}/data/"
            instr_offset = self.program.get_codelet_instr_offset(cdlt.instance_id)
            info_map[layer_key] = {
                'inputs': {},
                'outputs': {}
            }
            info_map[layer_key]['instructions'] = {
                "path": f"{base_cdlt_path}/{cdlt.cdlt_uid}_string_final.txt",
                "bitwidth": 32,
                "size_in_bytes": self.program.cdlt_num_instr(cdlt) * 4,
                "offset": instr_offset
            }
            for i in cdlt.inputs:
                input_path = f"{cdlt_path}{i.node_name}"
                # offset = self.program.
                inp_offset = self.program.get_input_operand_offset(i)
                info_blob = self.generate_operand_storage_info(cdlt, i, "input", inp_offset, input_path)

                info_map[layer_key]["inputs"][i.node_name] = info_blob
                input_offset += info_blob['size_in_bytes']

            for o in cdlt.outputs:
                output_path = f"{cdlt_path}{o.node_name}"
                out_offset = self.program.get_output_operand_offset(o)

                info_blob = self.generate_operand_storage_info(cdlt, o, "output", out_offset, output_path)

                info_map[layer_key]["outputs"][o.node_name] = info_blob
                output_offset += info_blob['size_in_bytes']


        output_location = f"{self.output_dir}/program"
        if not Path(output_location).exists():
            raise RuntimeError("directory for storying whole program does not exist!")
        res = json.dumps(info_map, indent=2, cls=NPEncoder)
        with open(f"{output_location}/{self.program.name}_operand_storage_info.json", "w") as outfile:
            outfile.write(res)


    def generate(self):
        if 'arch_cfg' in self.output_types:
            assert self.arch_cfg is not None
            self.arch_cfg['IBUF_END'] = int(BENCH_BASE_ADDR['IBUF'] + np.prod(self.program.codelets[0].inputs[0].shape))
            repo = git.Repo(search_parent_directories=True)
            self.arch_cfg['COMPILER_COMMIT_HASH'] = repo.head.object.hexsha
            res = json.dumps(self.arch_cfg, indent=2)
            with open(f"{self.output_dir}/{self.program.name}_arch_cfg.json", "w") as outfile:
                outfile.write(res)

        self.generate_codelet_data()

        if self.store_whole_program:
            self.store_program()


class NPEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NPEncoder, self).default(obj)
