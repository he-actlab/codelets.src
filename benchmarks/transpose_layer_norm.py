import onnx
from onnx import helper, ModelProto, AttributeProto
from onnxruntime.transformers.onnx_model import OnnxModel
from pathlib import Path
from typing import Dict, List, cast
MODEL_DIR = Path(f"{Path(__file__).parent}/models")


class LayerNormTranspose(object):
    def __init__(self, model, model_type):

        assert model_type in ["bert", "vit"]
        self.model_type = model_type
        fused_op_type = "LayerNormalization"
        search_op_types = "ReduceMean"
        description = None
        self.search_op_types: List[str] = [search_op_types] if isinstance(search_op_types, str) else search_op_types
        self.fused_op_type: str = fused_op_type
        self.description: str = f"{fused_op_type}({description})" if description else fused_op_type
        if isinstance(model, ModelProto):
            self.model: OnnxModel = OnnxModel(model)
        else:
            assert isinstance(model, str)
            model_proto = onnx.load(f"{MODEL_DIR}/{model}.onnx")
            self.model: OnnxModel = OnnxModel(model_proto)
        self.nodes_to_remove: List = []
        self.nodes_to_add: List = []
        self.prune_graph: bool = False
        self.node_name_to_graph_name: dict = {}
        self.this_graph_name: str = None
        # It is optional that subclass updates fused_count since we will also check nodes_to_add to get counter.
        self.fused_count: int = 0

    def apply(self):
        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        # This assumes that two search ops will not be fused at same time!
        for search_op_type in self.search_op_types:
            for node in self.model.get_nodes_by_op_type(search_op_type):
                graph = self.model.get_graph_by_node(node)
                if graph is None:
                    raise Exception("Can not find node in any graphs")
                self.this_graph_name = graph.name
                self.fuse(node, input_name_to_nodes, output_name_to_node)

        op_list = [node.op_type for node in self.nodes_to_add]
        count = max(self.fused_count, op_list.count(self.fused_op_type))


        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add, self.node_name_to_graph_name)
        if self.prune_graph:
            self.model.prune_graph()
        elif self.nodes_to_remove or self.nodes_to_add:
            self.model.update_graph()
        self.model.topological_sort()



    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        if self.model_type == "bert":
            self.fuse_bert(node, input_name_to_nodes, output_name_to_node)
        else:
            self.fuse_vit(node, input_name_to_nodes, output_name_to_node)

    def fuse_vit(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
            Fuse Layer Normalization subgraph into one node LayerNormalization:
                  +----------------------+
                  |                      |
                  |                      v
              [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                         (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0)    ^
                                         |                                               |
                                         +-----------------------------------------------+
             It also handles cases of duplicated sub nodes exported from older version of PyTorch:
                  +----------------------+
                  |                      v
                  |           +-------> Sub-----------------------------------------------+
                  |           |                                                           |
                  |           |                                                           v
              [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul --> Add
                  |                      ^
                  |                      |
                  +----------------------+
            """
        # All target nodes begin with an "Add" node with three children, so if we arent there, skip this
        children = self.model.get_children(node, input_name_to_nodes)

        if len(children) != 1:
            return
        root_input = node.input[0]
        child = children[0]

        # If the child node's input is not shared with the current node, return
        if child.op_type != 'Sub' or child.input[0] != root_input:
            return


        div_node = self.model.find_first_child_by_type(child, 'Div', input_name_to_nodes, recursive=False)
        if div_node is None:
            return

        path_id, parent_nodes, _ = self.model.match_parent_paths(
            div_node, [(['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Sub'], [1, 0, 0, 0, 0]),
                       (['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Cast', 'Sub'], [1, 0, 0, 0, 0, 0])], output_name_to_node)
        # No match found
        if path_id < 0:
            return

        # last node is sub-node and is the same one we found earlier
        sub_node = parent_nodes[-1]
        if sub_node != child:
            return

        second_add_node = parent_nodes[1]
        i, add_weight = self.model.get_constant_input(second_add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0E-4:
            print(f"epsilon value is not expeced: {add_weight}")
            return

        pow_node = parent_nodes[3]
        if not self.model.find_constant_input(pow_node, 2.0) == 1:
            return

        mul_node = input_name_to_nodes[div_node.output[0]][0]
        if mul_node.op_type != 'Mul':
            return

        last_add_node = input_name_to_nodes[mul_node.output[0]][0]
        if last_add_node.op_type != 'Add':
            return

        subgraph_nodes = [node]
        subgraph_nodes.extend(children)
        subgraph_nodes.extend(parent_nodes[:-1])

        subgraph_nodes.extend([last_add_node, mul_node, div_node])
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, last_add_node.output, input_name_to_nodes,
                                                output_name_to_node):
            print(f"It is not safe to insert transpose node. Skip")
            return

        weight_input = mul_node.input[1 - self.model.input_index(div_node.output[0], mul_node)]
        if not self.model.is_constant_with_specified_dimension(weight_input, 1, "layernorm weight"):
            return

        bias_input = last_add_node.input[1 - self.model.input_index(mul_node.output[0], last_add_node)]
        if not self.model.is_constant_with_specified_dimension(bias_input, 1, "layernorm bias"):
            return

        ## First, insert the transpose which starts the layer normalization
        transpose_output1 = node.input[0] + '_transposed'
        inpt_transpose_node = helper.make_node('Transpose',
                                               inputs=[node.input[0]],
                                               outputs=[transpose_output1],
                                               name=self.model.create_node_name("Transpose",
                                                                                name_prefix="Transpose"))
        inpt_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 2, 1])])
        self.nodes_to_add.append(inpt_transpose_node)
        self.node_name_to_graph_name[inpt_transpose_node.name] = self.this_graph_name

        # First, we need to get the "ReduceMean" parent, which should be an "Add" node, and then get the child "Add" node

        # This is where VIT differs from BERT: We only want to replace two uses with the transposed values
        self.model.replace_input_of_all_nodes(node.input[0], transpose_output1)
        transpose_children = self.model.get_children(inpt_transpose_node)
        if len(transpose_children) == 3:
            transpose_parent = self.model.get_parent(inpt_transpose_node, 0)
            add_child = None
            for c in transpose_children:
                if c.op_type == "Add":
                    add_child = c
                    break
            if add_child is None:
                raise RuntimeError(f"Unable to find correct child for transpose operation for node {node.name}")
            elif transpose_output1 not in add_child.input:
                raise RuntimeError(f"Transpose operation is not found in the parent of the add node for {node.name}")
            self.model.replace_node_input(add_child, transpose_output1, transpose_parent.output[0])

        for attr in node.attribute:
            if attr.name == "axes":
                assert attr.type == AttributeProto.AttributeType.INTS, f"Invalid type for attribute: {helper.printable_type(attr.type)}"
                attr.ints.pop()
                attr.ints.extend(int(i) for i in [1])

        ## Next, we need to update the axis for any reducemean operations
        last_reduce_mean = None
        for n in subgraph_nodes:
            if n.op_type == "ReduceMean" and n.name != node.name:
                last_reduce_mean = n
                break
        assert last_reduce_mean is not None

        for attr in last_reduce_mean.attribute:
            if attr.name == "axes":
                assert attr.type == AttributeProto.AttributeType.INTS, f"Invalid type for attribute: {helper.printable_type(attr.type)}"
                attr.ints.pop()
                attr.ints.extend(int(i) for i in [1])

        ## Lastly, convert the tensor back to the correct shape with an addition transpose after the
        ## last addition operation
        last_node = div_node
        transpose_output2 = last_node.output[0] + '_transposed'
        out_transpose_node = helper.make_node('Transpose',
                                              inputs=[last_node.output[0]],
                                              outputs=[transpose_output2],
                                              name=self.model.create_node_name("Transpose",
                                                                               name_prefix="Transpose"))
        out_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 2, 1])])
        self.nodes_to_add.append(out_transpose_node)
        self.node_name_to_graph_name[out_transpose_node.name] = self.this_graph_name
        self.model.replace_input_of_all_nodes(last_node.output[0], transpose_output2)

    def fuse_bert(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse Layer Normalization subgraph into one node LayerNormalization:
              +----------------------+
              |                      |
              |                      v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                     (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0)    ^
                                     |                                               |
                                     +-----------------------------------------------+
         It also handles cases of duplicated sub nodes exported from older version of PyTorch:
              +----------------------+
              |                      v
              |           +-------> Sub-----------------------------------------------+
              |           |                                                           |
              |           |                                                           v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul --> Add
              |                      ^
              |                      |
              +----------------------+
        """
        # If we are at the end of the graph (e.g., no children) or the current node's output is used multiple places
        # (e.g., more than one child), then return as we don't want to fuse this node
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) == 0 or len(children) > 2:
            return

        root_input = node.input[0]

        # If the current node does not split and reuse its output for children, continue
        if children[0].op_type != 'Sub' or children[0].input[0] != root_input:
            return



        if len(children) == 2:
            if children[1].op_type != 'Sub' or children[1].input[0] != root_input:
                return

        div_node = None
        for child in children:
            div_node = self.model.find_first_child_by_type(child, 'Div', input_name_to_nodes, recursive=False)
            if div_node is not None:
                break
        if div_node is None:
            return

        path_id, parent_nodes, _ = self.model.match_parent_paths(
            div_node, [(['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Sub'], [1, 0, 0, 0, 0]),
                       (['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Cast', 'Sub'], [1, 0, 0, 0, 0, 0])], output_name_to_node)
        if path_id < 0:
            return

        sub_node = parent_nodes[-1]
        if sub_node not in children:
            return

        second_add_node = parent_nodes[1]
        i, add_weight = self.model.get_constant_input(second_add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0E-4:
            print(f"epsilon value is not expeced: {add_weight}")
            return

        pow_node = parent_nodes[3]
        if not self.model.find_constant_input(pow_node, 2.0) == 1:
            return

        mul_node = input_name_to_nodes[div_node.output[0]][0]
        if mul_node.op_type != 'Mul':
            return

        last_add_node = input_name_to_nodes[mul_node.output[0]][0]
        if last_add_node.op_type != 'Add':
            return

        subgraph_nodes = [node]
        subgraph_nodes.extend(children)
        subgraph_nodes.extend(parent_nodes[:-1])

        subgraph_nodes.extend([last_add_node, mul_node, div_node])
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, last_add_node.output, input_name_to_nodes,
                                                output_name_to_node):
            print(f"It is not safe to fuse LayerNormalization node. Skip")
            return

        weight_input = mul_node.input[1 - self.model.input_index(div_node.output[0], mul_node)]
        if not self.model.is_constant_with_specified_dimension(weight_input, 1, "layernorm weight"):
            return

        bias_input = last_add_node.input[1 - self.model.input_index(mul_node.output[0], last_add_node)]
        if not self.model.is_constant_with_specified_dimension(bias_input, 1, "layernorm bias"):
            return


        ## First, insert the transpose which starts the layer normalization
        transpose_output1 = node.input[0] + '_transposed'
        inpt_transpose_node = helper.make_node('Transpose',
                                          inputs=[node.input[0]],
                                          outputs=[transpose_output1],
                                          name=self.model.create_node_name("Transpose",
                                                                           name_prefix="Transpose"))
        inpt_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 2, 1])])
        self.nodes_to_add.append(inpt_transpose_node)
        self.node_name_to_graph_name[inpt_transpose_node.name] = self.this_graph_name
        self.model.replace_input_of_all_nodes(node.input[0], transpose_output1)
        # Update the reduction dimension of the first reduce mean
        for attr in node.attribute:
            if attr.name == "axes":
                assert attr.type == AttributeProto.AttributeType.INTS, f"Invalid type for attribute: {helper.printable_type(attr.type)}"
                attr.ints.pop()
                attr.ints.extend(int(i) for i in [1])

        ## Next, we need to update the axis for any reducemean operations
        last_reduce_mean = None
        for n in subgraph_nodes:
            if n.op_type == "ReduceMean" and n.name != node.name:
                last_reduce_mean = n
                break
        assert last_reduce_mean is not None

        for attr in last_reduce_mean.attribute:
            if attr.name == "axes":
                assert attr.type == AttributeProto.AttributeType.INTS, f"Invalid type for attribute: {helper.printable_type(attr.type)}"
                attr.ints.pop()
                attr.ints.extend(int(i) for i in [1])

        ## Lastly, convert the tensor back to the correct shape with an addition transpose after the
        ## last addition operation
        last_node = div_node
        transpose_output2 = last_node.output[0] + '_transposed'
        out_transpose_node = helper.make_node('Transpose',
                                          inputs=[last_node.output[0]],
                                          outputs=[transpose_output2],
                                          name=self.model.create_node_name("Transpose",
                                                                           name_prefix="Transpose"))
        out_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 2, 1])])
        self.nodes_to_add.append(out_transpose_node)
        self.node_name_to_graph_name[out_transpose_node.name] = self.this_graph_name
        self.model.replace_input_of_all_nodes(last_node.output[0], transpose_output2)


class SoftmaxTranspose(object):
    def __init__(self, model, model_type):
        assert model_type in ["bert", "vit"]
        self.model_type = model_type
        fused_op_type = "LayerNormalization"
        search_op_types = "Softmax"
        description = None
        self.search_op_types: List[str] = [search_op_types] if isinstance(search_op_types, str) else search_op_types
        self.fused_op_type: str = fused_op_type
        self.description: str = f"{fused_op_type}({description})" if description else fused_op_type
        if isinstance(model, ModelProto):
            self.model: OnnxModel = OnnxModel(model)
        elif isinstance(model, OnnxModel):
            self.model = model
        else:
            assert isinstance(model, str)
            model_proto = onnx.load(f"{MODEL_DIR}/{model}.onnx")
            self.model: OnnxModel = OnnxModel(model_proto)
        self.shape_infer = None
        self.shape_infer_done = False
        self.nodes_to_remove: List = []
        self.nodes_to_add: List = []
        self.prune_graph: bool = False
        self.node_name_to_graph_name: dict = {}
        self.this_graph_name: str = None
        # It is optional that subclass updates fused_count since we will also check nodes_to_add to get counter.
        self.fused_count: int = 0

    def apply(self):
        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        # This assumes that two search ops will not be fused at same time!
        for search_op_type in self.search_op_types:
            for node in self.model.get_nodes_by_op_type(search_op_type):
                graph = self.model.get_graph_by_node(node)
                if graph is None:
                    raise Exception("Can not find node in any graphs")
                self.this_graph_name = graph.name
                self.fuse(node, input_name_to_nodes, output_name_to_node)

        op_list = [node.op_type for node in self.nodes_to_add]


        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add, self.node_name_to_graph_name)
        if self.prune_graph:
            self.model.prune_graph()
        elif self.nodes_to_remove or self.nodes_to_add:
            self.model.update_graph()
        self.model.topological_sort()

    def get_dimensions_from_tensor_proto(self, tensor_proto):
        if tensor_proto.type.tensor_type.HasField('shape'):
            return len(tensor_proto.type.tensor_type.shape.dim)
        else:
            return None

    def get_dimensions(self, input_name: str):
        graph_input = self.model.find_graph_input(input_name)
        if graph_input:
            return self.get_dimensions_from_tensor_proto(graph_input)

        if not self.shape_infer_done:
            self.shape_infer = self.model.infer_runtime_shape({}, update=False)
            self.shape_infer_done = True

        if self.shape_infer is not None:
            return self.get_dimensions_from_tensor_proto(self.shape_infer.known_vi_[input_name])

        return None

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse Layer Normalization subgraph into one node LayerNormalization:
              +----------------------+
              |                      |
              |                      v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                     (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0)    ^
                                     |                                               |
                                     +-----------------------------------------------+
         It also handles cases of duplicated sub nodes exported from older version of PyTorch:
              +----------------------+
              |                      v
              |           +-------> Sub-----------------------------------------------+
              |           |                                                           |
              |           |                                                           v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul --> Add
              |                      ^
              |                      |
              +----------------------+
        """

        # input_shape = self.get_dimensions(node.input[0])
        ## First, insert the transpose which starts the layer normalization
        transpose_output1 = node.input[0] + '_transposed'
        inpt_transpose_node = helper.make_node('Transpose',
                                          inputs=[node.input[0]],
                                          outputs=[transpose_output1],
                                          name=self.model.create_node_name("Transpose",
                                                                           name_prefix="Transpose"))
        inpt_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 1, 3, 2])])
        self.nodes_to_add.append(inpt_transpose_node)
        self.node_name_to_graph_name[inpt_transpose_node.name] = self.this_graph_name
        self.model.replace_input_of_all_nodes(node.input[0], transpose_output1)
        # Update the reduction dimension of the first reduce mean
        for attr in node.attribute:
            if attr.name == "axis":
                assert attr.type == AttributeProto.AttributeType.INT, f"Invalid type for attribute"
                assert attr.i == 3, f"Softmax layer {node.name} has axis {attr.i}, expected 3"
                attr.i = 2

        ## Lastly, convert the tensor back to the correct shape with an addition transpose after the
        ## last addition operation

        last_node = node
        transpose_output2 = last_node.output[0] + '_transposed'
        out_transpose_node = helper.make_node('Transpose',
                                          inputs=[last_node.output[0]],
                                          outputs=[transpose_output2],
                                          name=self.model.create_node_name("Transpose",
                                                                           name_prefix="Transpose"))
        out_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 1, 3, 2])])
        self.nodes_to_add.append(out_transpose_node)
        self.node_name_to_graph_name[out_transpose_node.name] = self.this_graph_name
        self.model.replace_input_of_all_nodes(last_node.output[0], transpose_output2)