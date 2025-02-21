const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");

const Stream = @import("stream.zig").Stream;

pub const Graph = struct {
    graph: c.cudaGraph_t = null,

    pub const Node = struct {
        node: c.cudaGraphNode_t = null,

        pub fn findInClone(self: *const Node, cloned_graph: *Graph) !Node {
            var new_node: c.cudaGraphNode_t = null;
            try err.checkCuda(c.cudaGraphNodeFindInClone(&new_node, self.node, cloned_graph.graph));
            return .{ .node = new_node };
        }

        pub fn getDependenciesAlloc(self: *const Node, allocator: std.mem.Allocator) ![]c.cudaGraphNode_t {
            var num: usize = 0;
            try err.checkCuda(c.cudaGraphNodeGetDependencies(self.node, null, &num));

            const dependencies = try allocator.alloc(c.cudaGraphNode_t, num);
            errdefer allocator.free(dependencies);

            try err.checkCuda(c.cudaGraphNodeGetDependencies(self.node, dependencies.ptr, &num));

            return dependencies;
        }

        pub fn getDependentNodesAlloc(self: *const Node, allocator: std.mem.Allocator) ![]c.cudaGraphNode_t {
            var num: usize = 0;
            try err.checkCuda(c.cudaGraphNodeGetDependentNodes(self.node, null, &num));

            const dependent_nodes = try allocator.alloc(c.cudaGraphNode_t, num);
            errdefer allocator.free(dependent_nodes);

            try err.checkCuda(c.cudaGraphNodeGetDependentNodes(self.node, dependent_nodes.ptr, &num));

            return dependent_nodes;
        }

        pub fn getType(self: *const Node) !c.cudaGraphNodeType {
            var ty: c.cudaGraphNodeType = 0;
            try err.checkCuda(c.cudaGraphNodeGetType(self.node, &ty));

            return ty;
        }
    };

    pub fn create(flags: c_uint) !Graph {
        var graph: c.cudaGraph_t = null;
        try err.checkCuda(c.cudaGraphCreate(&graph, flags));
        return .{ .graph = graph };
    }

    pub fn destroy(self: *Graph) void {
        _ = c.cudaGraphDestroy(self.graph);
        self.graph = null;
    }

    pub fn clone(self: *const Graph) !Graph {
        var cloned_graph: c.cudaGraph_t = null;
        try err.checkCuda(c.cudaGraphClone(&cloned_graph, self.graph));
        return .{ .graph = cloned_graph };
    }

    pub fn instantiate(self: *const Graph, flags: u64) !ExecutableGraph {
        var executable_graph: c.cudaGraphExec_t = null;
        try err.checkCuda(c.cudaGraphInstantiate(&executable_graph, self.graph, flags));
        return .{ .executable_graph = executable_graph };
    }

    pub fn addEmptyNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddEmptyNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
        ));
        return .{ .node = node };
    }

    pub fn addEventRecordNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        event: c.cudaEvent_t,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddEventRecordNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            event,
        ));
        return .{ .node = node };
    }

    pub fn addEventWaitNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        event: c.cudaEvent_t,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddEventWaitNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            event,
        ));
        return .{ .node = node };
    }

    // use tuple and @call to prevent making all host functions wrapper
    pub fn addHostNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        params: *const c.cudaHostNodeParams,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddHostNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            params,
        ));
        return .{ .node = node };
    }

    pub fn addKernelNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        params: *const c.cudaKernelNodeParams,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddKernelNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            params,
        ));
        return .{ .node = node };
    }

    pub fn addMemcpyNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        params: *const c.cudaMemcpy3DParms,
    ) !Node {
        std.debug.assert(dependencies.len > 0);
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddMemcpyNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            params,
        ));
        return .{ .node = node };
    }

    pub fn addMemcpyNode1D(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        pdst: ?*anyopaque,
        src: ?*const anyopaque,
        count: usize,
        kind: c.cudaMemcpyKind,
    ) !Node {
        std.debug.assert(dependencies.len > 0);
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddMemcpyNode1D(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            pdst,
            src,
            count,
            kind,
        ));
        return .{ .node = node };
    }

    pub fn addMemsetNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        params: *const c.cudaMemsetParams,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddMemsetNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            params,
        ));
        return .{ .node = node };
    }

    pub fn addMemAllocNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        params: *const c.cudaMemAllocNodeParams,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddMemAllocNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            params,
        ));
        return .{ .node = node };
    }

    pub fn addMemFreeNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        params: *const c.cudaMemFreeNodeParams,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddMemFreeNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            params.dptr,
        ));
        return .{ .node = node };
    }

    pub fn addChildGraphNode(
        self: *Graph,
        dependencies: []const c.cudaGraphNode_t,
        child_graph: *const Graph,
    ) !Node {
        var node: c.cudaGraphNode_t = null;
        try err.checkCuda(c.cudaGraphAddChildGraphNode(
            &node,
            self.graph,
            if (dependencies.len > 0) dependencies.ptr else null,
            dependencies.len,
            child_graph.graph,
        ));
        return .{ .node = node };
    }

    pub fn addDependencies(
        self: *Graph,
        from_nodes: []const c.cudaGraphNode_t,
        to_nodes: []const c.cudaGraphNode_t,
    ) !void {
        std.debug.assert(from_nodes.len == to_nodes.len);
        try err.checkCuda(c.cudaGraphAddDependencies(
            self.graph,
            from_nodes.ptr,
            to_nodes.ptr,
            from_nodes.len,
        ));
    }

    pub fn removeDependencies(
        self: *Graph,
        from_nodes: []const c.cudaGraphNode_t,
        to_nodes: []const c.cudaGraphNode_t,
    ) !void {
        std.debug.assert(from_nodes.len == to_nodes.len);
        try err.checkCuda(c.cudaGraphRemoveDependencies(
            self.graph,
            from_nodes.ptr,
            to_nodes.ptr,
            from_nodes.len,
        ));
    }

    pub fn getNodesAlloc(self: *const Graph, allocator: std.mem.Allocator) ![]c.cudaGraphNode_t {
        var count: usize = 0;
        try err.checkCuda(c.cudaGraphGetNodes(self.graph, null, &count));
        const nodes = try allocator.alloc(c.cudaGraphNode_t, count);
        errdefer allocator.free(nodes);
        try err.checkCuda(c.cudaGraphGetNodes(self.graph, nodes.ptr, &count));
        return nodes;
    }
};

pub const ExecutableGraph = struct {
    executable_graph: c.cudaGraphExec_t = null,

    pub fn destroy(self: *ExecutableGraph) void {
        _ = c.cudaGraphExecDestroy(self.executable_graph);
        self.executable_graph = null;
    }

    pub fn launch(self: *const ExecutableGraph, stream: *const Stream) !void {
        try err.checkCuda(c.cudaGraphLaunch(self.executable_graph, stream.stream));
    }

    pub fn update(self: *ExecutableGraph, graph: *const Graph) !c.cudaGraphExecUpdateResultInfo {
        var res_info: c.cudaGraphExecUpdateResultInfo = .{};
        try err.checkCuda(c.cudaGraphExecUpdate(self.executable_graph, graph.graph, &res_info));
        return res_info;
    }

    pub fn setEventRecordNodeSetEventParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        event: c.cudaEvent_t,
    ) !void {
        try err.checkCuda(c.cudaGraphExecEventRecordNodeSetEvent(
            self.executable_graph,
            node.node,
            event,
        ));
    }

    pub fn setEventWaitNodeSetEventSetEventParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        event: c.cudaEvent_t,
    ) !void {
        try err.checkCuda(c.cudaGraphExecEventWaitNodeSetEvent(
            self.executable_graph,
            node.node,
            event,
        ));
    }

    pub fn setHostNodeParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        params: *const c.cudaHostNodeParams,
    ) !void {
        try err.checkCuda(c.cudaGraphExecHostNodeSetParams(
            self.executable_graph,
            node.node,
            params,
        ));
    }

    pub fn setKernelParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        params: *const c.cudaKernelNodeParams,
    ) !void {
        try err.checkCuda(c.cudaGraphExecKernelNodeSetParams(
            self.executable_graph,
            node.node,
            params,
        ));
    }

    pub fn setMemcpyParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        params: *const c.cudaMemcpy3DParms,
    ) !void {
        try err.checkCuda(c.cudaGraphExecMemcpyNodeSetParams(
            self.executable_graph,
            node.node,
            params,
        ));
    }

    pub fn setMemcpy1DParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        dst: ?*anyopaque,
        src: ?*const anyopaque,
        count: usize,
        kind: c.cudaMemcpyKind,
    ) !void {
        try err.checkCuda(c.cudaGraphExecMemcpyNodeSetParams1D(
            self.executable_graph,
            node.node,
            dst,
            src,
            count,
            kind,
        ));
    }

    pub fn setMemsetParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        params: *const c.cudaMemsetParams,
    ) !void {
        try err.checkCuda(c.cudaGraphExecMemsetNodeSetParams(
            self.executable_graph,
            node.node,
            params,
        ));
    }

    pub fn setNodeParams(
        self: *ExecutableGraph,
        node: Graph.Node,
        params: *const c.cudaGraphNodeParams,
    ) !void {
        try err.checkCuda(c.cudaGraphExecNodeSetParams(
            self.executable_graph,
            node.node,
            params,
        ));
    }
};
