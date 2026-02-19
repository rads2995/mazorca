#pragma once

#include "mazorca/mazorca.hpp"

#include <unordered_set>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace mazorca {

struct cpu_deletor_t {
    cpu_deletor_t() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

inline constexpr void set_any_layout(
    const std::vector<dnnl::graph::partition> &partitions,
    std::unordered_set<size_t> &id_to_set_any_layout
) {
    // mapping from output tensor id to the all supported flags of
    // supported partitions, we may only need outputs' supported flags
    std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
    for (const auto &p : partitions) {
        for (const auto &out : p.get_output_ports()) {
            std::size_t id = out.get_id();
            if (p.is_supported() && output_to_flag_map.find(id) == output_to_flag_map.end()) {
                output_to_flag_map[id] = {};
            }
        }

        for (const auto &in : p.get_input_ports()) {
            std::size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            if (iter != output_to_flag_map.end()) {
                // collect all of supported flags of this tensor's uses
                // Considering we have such a graph:
                //
                //   partition_A  partition_B
                //        \           |
                //      tensor1    tensor2
                //           \     /     |
                //         partition_C  unsupported partition
                //              |
                //           tensor3
                //              |
                //          framework op
                //
                // so the mapping of partition_A's output will be { true }
                // the mapping of partition_B's output will be { true, false }
                // The mapping of partition_C's output will be { false }
                // Only when all supported flags are true, users can set any
                // layout.
                iter->second.push_back(p.is_supported());
            }
        }
    }

    for (const auto &p : partitions) {
        // no need to set `any` layout if this partition is not supported
        if (!p.is_supported()) {
            continue;
        }
        for (const auto &in : p.get_input_ports()) {
            std::size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            // if this input tensor is not an output of another supported
            // partition, just skip
            if (iter == output_to_flag_map.end()) {
                continue;
            }
            std::vector<bool> flag_vec {iter->second};
            // check if all of uses of this tensor are supported partitions,
            // if not, no need to set ANY layout.
            bool need_set_any = std::ranges::all_of(flag_vec, [](const bool a) {
                return a; 
            });
            if (!need_set_any) {
                continue;
            }
            /// record the id of logical tensor that will be set to ANY layout
            id_to_set_any_layout.insert(id);
        }
    }
}

inline constexpr void allocate_graph_mem(
    std::vector<dnnl::graph::tensor> &tensors,
    const std::vector<dnnl::graph::logical_tensor> &lts,
    std::vector<std::shared_ptr<void>> &data_buffer,
    std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
    const dnnl::engine &eng, bool is_input
) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if (is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if (pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(malloc(mem_size), cpu_deletor_t {});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if (!is_input) global_outputs_ts_map[lt_id] = tensors.back();
    }
}

inline constexpr std::expected<void, mazorca::error_code> inference(const mazorca::grano& grano) {

    if (!grano.sycl_device.is_cpu()) {
        std::println("[{}] [ERROR] Inference is only supported on CPU SYCL devices.", mazorca::current_time());
        return std::unexpected(mazorca::error_code::unsupported);
    }

    constexpr dnnl::graph::logical_tensor::dim N = 8, IC = 3, OC1 = 96, OC2 = 96;
    constexpr dnnl::graph::logical_tensor::dim IH = 227, IW = 227, KH1 = 11, KW1 = 11, KH2 = 1, KW2 = 1;

    // We build a graph containing the pattern 'conv0->relu0->conv1->relu1'
    dnnl::graph::logical_tensor::dims conv0_input_dims {N, IC, IH, IW};
    dnnl::graph::logical_tensor::dims conv0_weight_dims {OC1, IC, KH1, KW1};
    dnnl::graph::logical_tensor::dims conv0_bias_dims {OC1};
    dnnl::graph::logical_tensor::dims conv1_weight_dims {OC1, OC2, KH2, KW2};
    dnnl::graph::logical_tensor::dims conv1_bias_dims {OC2};

    // Create logical tensors for these operations including inputs and outputs
    dnnl::graph::logical_tensor conv0_src_desc {0, dnnl::graph::logical_tensor::data_type::f32};
    dnnl::graph::logical_tensor conv0_weight_desc {1, dnnl::graph::logical_tensor::data_type::f32};
    dnnl::graph::logical_tensor conv0_dst_desc {2, dnnl::graph::logical_tensor::data_type::f32};

    // Create first convolution operation and set attributes to it
    dnnl::graph::op conv0(
        0, 
        dnnl::graph::op::kind::Convolution, 
        {conv0_src_desc, conv0_weight_desc},
        {conv0_dst_desc}, 
        "conv0"
    );

    conv0.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::strides, {4, 4});
    conv0.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::pads_begin, {0, 0});
    conv0.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::pads_end, {0, 0});
    conv0.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::dilations, {1, 1});
    conv0.set_attr<int64_t>(dnnl::graph::op::attr::groups, 1);
    conv0.set_attr<std::string>(dnnl::graph::op::attr::data_format, "NCX");
    conv0.set_attr<std::string>(dnnl::graph::op::attr::weights_format, "OIX");

    // Create input/output logical tensor for first 'BiasAdd' operation and create operation itself
    dnnl::graph::logical_tensor conv0_bias_desc {3, dnnl::graph::logical_tensor::data_type::f32};
    dnnl::graph::logical_tensor conv0_bias_add_dst_desc {4, dnnl::graph::logical_tensor::data_type::f32};

    dnnl::graph::op conv0_bias_add(
        1, 
        dnnl::graph::op::kind::BiasAdd, 
        {conv0_dst_desc, conv0_bias_desc},
        {conv0_bias_add_dst_desc}, 
        "conv0_bias_add"
    );
    
    conv0_bias_add.set_attr<std::string>(dnnl::graph::op::attr::data_format, "NCX");

    // Create output logical tensors for first 'ReLU' operation and create operation itself
    dnnl::graph::logical_tensor relu0_dst_desc {5, dnnl::graph::logical_tensor::data_type::f32};
    
    dnnl::graph::op relu0(
        2, 
        dnnl::graph::op::kind::ReLU, 
        {conv0_bias_add_dst_desc}, 
        {relu0_dst_desc},
        "relu0"
    );

    // Create input/output logical tensors for second convolution operation and create operation itself
    dnnl::graph::logical_tensor conv1_weight_desc {6, dnnl::graph::logical_tensor::data_type::f32};
    dnnl::graph::logical_tensor conv1_dst_desc {7, dnnl::graph::logical_tensor::data_type::f32};
    
    dnnl::graph::op conv1(
        3, 
        dnnl::graph::op::kind::Convolution, 
        {relu0_dst_desc, conv1_weight_desc},
        {conv1_dst_desc}, 
        "conv1"
    );
    
    conv1.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::strides, {1, 1});
    conv1.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::pads_begin, {0, 0});
    conv1.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::pads_end, {0, 0});
    conv1.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::dilations, {1, 1});
    conv1.set_attr<int64_t>(dnnl::graph::op::attr::groups, 1);
    conv1.set_attr<std::string>(dnnl::graph::op::attr::data_format, "NCX");
    conv1.set_attr<std::string>(dnnl::graph::op::attr::weights_format, "OIX");

    // Create input/output logical tensors for second `BiasAdd` operation and create operation itself
    dnnl::graph::logical_tensor conv1_bias_desc {8, dnnl::graph::logical_tensor::data_type::f32};
    dnnl::graph::logical_tensor conv1_bias_add_dst_desc {9, dnnl::graph::logical_tensor::data_type::f32};
    
    dnnl::graph::op conv1_bias_add(
        4, 
        dnnl::graph::op::kind::BiasAdd, 
        {conv1_dst_desc, conv1_bias_desc},
        {conv1_bias_add_dst_desc}, 
        "conv1_bias_add"
    );
    
    conv1_bias_add.set_attr<std::string>(dnnl::graph::op::attr::data_format, "NCX");

    // Create output logical tensors for second `ReLU` operation and create operation itself
    dnnl::graph::logical_tensor relu1_dst_desc {10, dnnl::graph::logical_tensor::data_type::f32};
    
    dnnl::graph::op relu1(
        5, 
        dnnl::graph::op::kind::ReLU, 
        {conv1_bias_add_dst_desc}, 
        {relu1_dst_desc},
        "relu1"
    );

    // Create oneDNN engine and stream objects from SYCL device, context and queue in mazorca object
    dnnl::engine engine {dnnl::sycl_interop::make_engine(grano.sycl_device, grano.sycl_context)};
    dnnl::stream stream {dnnl::sycl_interop::make_stream(engine, grano.sycl_queue)};

    // Create graph object and add operations to it
    dnnl::graph::graph graph {engine.get_kind()};

    graph.add_op(conv0);
    graph.add_op(conv0_bias_add);
    graph.add_op(relu0);

    graph.add_op(conv1);
    graph.add_op(conv1_bias_add);
    graph.add_op(relu1);

    // Finalize the graph before partitioning
    graph.finalize();

    std::vector<dnnl::graph::partition> partitions {graph.get_partitions()};

    assert(partitions.size() == 2);

    // Mapping from logical tensor id to output tensors
    std::unordered_map<size_t, dnnl::graph::tensor> global_outputs_ts_map;

    // Memory buffers bound to the partition input/output tensors
    std::vector<std::shared_ptr<void>> data_buffer;

    // Mapping from id to queried logical tensor from compiled partition
    std::unordered_map<size_t, dnnl::graph::logical_tensor> id_to_queried_logical_tensors;

    std::unordered_set<size_t> ids_with_any_layout;
    set_any_layout(partitions, ids_with_any_layout);

    // Mapping from logical tensor ID to the concrete shapes
    std::unordered_map<size_t, dnnl::graph::logical_tensor::dims> concrete_shapes {
        {0, conv0_input_dims},
        {1, conv0_weight_dims}, 
        {3, conv0_bias_dims},
        {6, conv1_weight_dims}, 
        {8, conv1_bias_dims}
    };

    // Compile and execute the partitions
    for (const auto& partition : partitions) {
        if (!partition.is_supported()) {
            std::println("[{}] [ERROR] Unsupported partition, users need to handle the operators by themselves.", mazorca::current_time());
            continue;
        }

        std::vector<dnnl::graph::logical_tensor> inputs {partition.get_input_ports()};
        std::vector<dnnl::graph::logical_tensor> outputs {partition.get_output_ports()};

        // Update input logical tensors with concrete shape and layout
        for (auto &input : inputs) {
            const auto id = input.get_id();
            // If the tensor is an output of another partition, use the cached logical tensor
            if (id_to_queried_logical_tensors.find(id) != id_to_queried_logical_tensors.end()) {
                input = id_to_queried_logical_tensors[id];
            }
            else {
                // Create logical tensor with strided layout
                input = dnnl::graph::logical_tensor {
                    id, 
                    input.get_data_type(),
                    concrete_shapes[id], 
                    dnnl::graph::logical_tensor::layout_type::strided
                };
            }
        }

        // Update output logical tensors with concrete shape and layout
        for (auto &output : outputs) {
            const auto id = output.get_id();
            output = dnnl::graph::logical_tensor {
                id, 
                output.get_data_type(),
                DNNL_GRAPH_UNKNOWN_NDIMS, // set output dims to unknown
                ids_with_any_layout.count(id) ? 
                    dnnl::graph::logical_tensor::layout_type::any
                    : dnnl::graph::logical_tensor::layout_type::strided
            };
        }

        /// Compile the partition to generate compiled partition with the
        /// input and output logical tensors
        dnnl::graph::compiled_partition cp {partition.compile(inputs, outputs, engine)};

        // Update output logical tensors with queried one
        for (auto &output : outputs) {
            const auto id = output.get_id();
            output = cp.query_logical_tensor(id);
            id_to_queried_logical_tensors[id] = output;
        }

        // Allocate memory for the partition, and bind the data buffers with
        // input and output logical tensors
        std::vector<dnnl::graph::tensor> inputs_ts {}, outputs_ts {};
        allocate_graph_mem(
            inputs_ts, 
            inputs, 
            data_buffer,
            global_outputs_ts_map, 
            engine, 
            /*is partition input=*/true
        );
        allocate_graph_mem(
            outputs_ts, 
            outputs, 
            data_buffer,
            global_outputs_ts_map, 
            engine, 
            /*is partition input=*/false
        );

        /// Execute the compiled partition on the specified stream
        cp.execute(stream, inputs_ts, outputs_ts);
    }

    // Wait for all compiled partition's execution finished
    stream.wait();

    std::cout << "Graph:" << std::endl
              << " [conv0_src] [conv0_wei]" << std::endl
              << "       \\      /" << std::endl
              << "         conv0" << std::endl
              << "          \\    [conv0_bias_src1]" << std::endl
              << "           \\      /" << std::endl
              << "         conv0_bias_add" << std::endl
              << "                |" << std::endl
              << "              relu0" << std::endl
              << "                \\   [conv1_wei]" << std::endl
              << "                 \\    /" << std::endl
              << "                  conv1" << std::endl
              << "                    \\  [conv1_bias_src1]" << std::endl
              << "                     \\      /" << std::endl
              << "                  conv1_bias_add" << std::endl
              << "                          |" << std::endl
              << "                        relu1" << std::endl
              << "                          |" << std::endl
              << "                      [relu_dst]" << std::endl
              << "Note:" << std::endl
              << " '[]' represents a logical tensor, which refers to "
                 "inputs/outputs of the graph. "
              << std::endl;

    return {};
}

} // namespace mazorca
