"""CUDA graph wrapper, adapted from the QuIP# CUDA-graph wrapper.

Wraps a model class so that the FIRST forward pass is captured into a CUDA
graph and subsequent calls replay it. Input tensor shapes must be identical
across calls — fine for the single-token forward bench.
"""
import torch


def get_graph_wrapper(cls, device=0):

    class GraphWrapper(cls):

        def __init__(self, *args, **kwargs):
            super(GraphWrapper, self).__init__(*args, **kwargs)
            self.built_graph = False
            self.graph_device = device

        def forward(self, *args, **kwargs):
            with torch.cuda.device(self.graph_device):
                if not self.built_graph:
                    self.static_args = args
                    self.static_kwargs = kwargs

                    s = torch.cuda.Stream(device=self.graph_device)
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        super(GraphWrapper, self).forward(
                            *self.static_args, **self.static_kwargs)
                    torch.cuda.current_stream().wait_stream(s)

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=s):
                        self.static_output = super(GraphWrapper, self).forward(
                            *self.static_args, **self.static_kwargs)

                    self.built_graph = True
                    print("Built CUDA graph of model.")

                for i in range(len(args)):
                    if isinstance(args[i], torch.Tensor):
                        self.static_args[i].copy_(args[i])
                for kw in kwargs:
                    if isinstance(kwargs[kw], torch.Tensor):
                        self.static_kwargs[kw].copy_(kwargs[kw])

                self.graph.replay()
                return self.static_output

    return GraphWrapper
