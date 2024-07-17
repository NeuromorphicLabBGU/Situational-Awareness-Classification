import math

import numpy as np
import torch

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.network.topology import Connection


class Resonators2BandsLayer(Network):

    def __init__(self, channel_name, band_name, clk_freq_resonators: torch.Tensor):
        super().__init__(dt=1)
        self.n_resonators = len(clk_freq_resonators)
        self.band_name = band_name
        self.channel_name = channel_name
        # Layers
        input_layer = Input(
            n=self.n_resonators, shape=(self.n_resonators,), traces=True, tc_trace=20.0
        )

        agg_layer = LIFNodes(
            n=1,
            traces=False,
            rest=0.0,
            reset=3 * self.n_resonators,
            thresh=5 * self.n_resonators,
            tc_decay=60 * self.n_resonators,
            refrac=1,
        )
        w = torch.where(clk_freq_resonators == 153600, 1.0, 11.0)
        input_exc_conn = Connection(
            source=input_layer,
            target=agg_layer,
            w=w,
        )
        self.add_layer(input_layer, name=f"{channel_name}-{band_name}-resonators")
        self.add_layer(agg_layer, name=f"{channel_name}-{band_name}-band")

        self.add_connection(
            input_exc_conn,
            source=f"{channel_name}-{band_name}-resonators",
            target=f"{channel_name}-{band_name}-band",
        )


class Bands2TopographyLayer(Network):

    def __init__(
        self, band_name: str, ch_netowrks: dict, shape: tuple, ch_pos: dict, sigma=1
    ):
        super().__init__(dt=1)

        self.ch_netowrks = ch_netowrks
        self.band_name = band_name
        topography_layer = LIFNodes(
            shape=shape,
            n=math.prod(shape),
            traces=True,
            rest=0.0,
            thresh=4,
        )

        # add layers ch_network to topography network!
        for ch in ch_netowrks.keys():
            ch_network = ch_netowrks[ch]
            self.add_layer(
                ch_network.layers[f"{ch}-{band_name}-resonators"],
                name=f"{ch}-{band_name}-resonators",
            )
        for ch in ch_netowrks.keys():
            ch_network = ch_netowrks[ch]
            self.add_layer(
                ch_network.layers[f"{ch}-{band_name}-band"],
                name=f"{ch}-{band_name}-band",
            )

        self.add_layer(topography_layer, name=f"{band_name}-topography")
        for ch in ch_netowrks.keys():
            ch_network = ch_netowrks[ch]
            for (source, target), conn in ch_network.connections.items():
                self.add_connection(conn, source=source, target=target)

            output_ch_layer = ch_network.layers[f"{ch}-{band_name}-band"]
            w = self.channel_weights(shape, center=ch_pos[ch], sigma=sigma)
            conn = Connection(
                source=output_ch_layer,
                target=topography_layer,
                w=w,
            )
            self.add_connection(
                conn, source=f"{ch}-{band_name}-band", target=f"{band_name}-topography"
            )

    def channel_weights(self, kernel_shape, center, sigma=1):
        # Create an empty kernel
        weights = torch.zeros(kernel_shape)

        # Calculate the Gaussian values for the new kernel
        radius = (kernel_shape[0] - 1) / 2
        dr = 1
        xy_center = ((kernel_shape[0] - 1) / 2, (kernel_shape[1] - 1) / 2)
        for x in range(kernel_shape[0]):
            for y in range(kernel_shape[1]):
                distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
                weights[x, y] = np.exp(-distance_squared / (2 * sigma**2))

                r = np.sqrt((x - xy_center[0]) ** 2 + (y - xy_center[1]) ** 2)
                if (r - dr / 2) > radius:
                    weights[x, y] = 0

        # Normalize the kernel
        weights /= weights.max()
        # return weights
        return weights.view(1, -1)


class Topographies2SNN(Network):

    def __init__(
        self,
        topography_maps: dict[str, Bands2TopographyLayer],
        fc1_neurons: int,
        fc2_neurons: int,
        output_neurons: int,
    ):
        super().__init__()
        self.topography_maps = topography_maps

        fc1_layer = LIFNodes(
            n=fc1_neurons,
            traces=True,
            rest=0.0,
            thresh=10,
        )
        self.add_layer(fc1_layer, name="fc1_layer")

        fc2_layer = LIFNodes(
            n=fc2_neurons,
            traces=True,
            rest=0.0,
            thresh=10,
        )
        self.add_layer(fc2_layer, name="fc2_layer")

        output_layer = LIFNodes(
            n=output_neurons,
            traces=True,
            rest=0.0,
            thresh=10,
        )
        self.add_layer(output_layer, name="output_layer")

        # add full bands module
        for band_name, topography_map in topography_maps.items():

            for layer_name, layer in topography_map.layers.items():
                self.add_layer(layer, name=layer_name)
            for (source, target), conn in topography_map.connections.items():
                self.add_connection(conn, source=source, target=target)

            # Connections from topography to fc1
            topography_layer = topography_map.layers[f"{band_name}-topography"]
            w = 0.3 * torch.rand(topography_layer.n, fc1_neurons)
            fc1_conn = Connection(
                source=topography_layer,
                target=fc1_layer,
                w=w,
                update_rule=PostPre,
            )
            self.add_connection(
                fc1_conn, source=f"{band_name}-topography", target="fc1_layer"
            )

        w = 0.3 * torch.rand(fc1_neurons, fc2_neurons)
        fc2_conn = Connection(
            source=fc1_layer,
            target=fc2_layer,
            w=w,
            update_rule=PostPre,
        )
        self.add_connection(fc2_conn, source=f"fc1_layer", target="fc2_layer")

        w = 0.3 * torch.rand(fc2_neurons, output_neurons)
        output_conn = Connection(
            source=fc2_layer,
            target=output_layer,
            w=w,
            update_rule=PostPre,
        )
        self.add_connection(output_conn, source=f"fc2_layer", target="output_layer")
