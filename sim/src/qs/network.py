import sys
import netsquid as ns
import pandas
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.nodes import Node, Connection, Network
from netsquid.protocols.protocol import Signals
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.util.datacollector import DataCollector
import pydynaa
from netsquid.qubits import ketstates as ks
from netsquid.qubits import qubitapi as qapi
from netsquid.components import instructions as instr
from netsquid.components.instructions import Instruction
from netsquid.qubits.operators import Operator, I, X, Y
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from .hardware import (
    create_processor,
    create_client_processor,
    ExternalEntanglingConnection,
)


class JitteredFibreDelayModel(FibreDelayModel):
    """FibreDelayModel with optional exponential jitter on top of propagation delay."""

    def __init__(self, c=200000, *, cc_jitter_mean_ns: float = 0.0, rng=None, **kwargs):
        super().__init__(c=c, rng=rng, **kwargs)
        self.cc_jitter_mean_ns = float(cc_jitter_mean_ns)

    def generate_delay(self, **kwargs):
        base = super().generate_delay(**kwargs)
        if self.cc_jitter_mean_ns <= 0:
            return base
        jitter = self.rng.exponential(scale=self.cc_jitter_mean_ns)

        return base + jitter


class ClassicalConnection(Connection):
    """A connection that transmits classical messages in one direction, from A to B."""
    def __init__(self, length, name="ClassicalConnection", *, cc_jitter_mean_ns: float = 0.0, rng=None):
        super().__init__(name=name)
        delay_model = (
            FibreDelayModel(c=200000, rng=rng)
            if cc_jitter_mean_ns <= 0
            else JitteredFibreDelayModel(
                c=200000,
                cc_jitter_mean_ns=cc_jitter_mean_ns,
                rng=rng,
            )
        )
        self.add_subcomponent(
            ClassicalChannel("Channel_A2B", length=length,
                             models={"delay_model": delay_model}),
            forward_input=[("A", "send")],
            forward_output=[("B", "recv")]
        )


def example_network_setup(
    dephase_rate,
    node_distance,
    T1,
    T2_ratio,
    client_T1,
    sge,
    dge,
    gate_speed_factor=1.0,
    client_gate_speed_factor=1,
    entanglement_fidelity=1.0,
    client_fidelity=1,
    cc_jitter_mean_ns: float = 0.0,
    rng_ab=None,
    rng_ba=None,
    *,
    timing_mode: str = "factor",
    gate_time_ns: "Optional[float]" = None,
):
    gate_durations_override = None

    if timing_mode == "absolute" and gate_time_ns is not None:

        from netsquid.components import instructions as instr

        t = float(gate_time_ns)
        gate_durations_override = {
            instr.INSTR_INIT: 1000,  
            instr.INSTR_H: t,
            instr.INSTR_X: t,
            instr.INSTR_Z: t,
            instr.INSTR_Y: t,
            instr.INSTR_S: t,
            instr.INSTR_ROT_X: t,
            instr.INSTR_ROT_Y: t,
            instr.INSTR_ROT_Z: t,
            instr.INSTR_CNOT: t,
            instr.INSTR_MEASURE: t,
        }

        gate_speed_factor = 1.0
        client_gate_speed_factor = 1.0

    alice = Node(
        "Alice",
        qmemory=create_processor(
            dephase_rate=dephase_rate,
            T1=T1,
            T2_ratio=T2_ratio,
            sge=sge,
            dge=dge,
            gate_speed_factor=gate_speed_factor,
            gate_durations=gate_durations_override,
        ),
    )
    bob = Node(
        "Bob",
        qmemory=create_client_processor(
            dephase_rate=client_fidelity,
            T1=client_T1,
            T2_ratio=T2_ratio,
            sge=sge,
            dge=dge,
            gate_speed_factor=client_gate_speed_factor,
            gate_durations=gate_durations_override,
        ),
    )
    network = Network("Teleportation_network")
    network.add_nodes([alice, bob])


    c_conn = ClassicalConnection(length=node_distance, cc_jitter_mean_ns=cc_jitter_mean_ns, rng=rng_ab)
    network.add_connection(alice, bob, connection=c_conn, label="classical",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    c_conn_reverse = ClassicalConnection(length=node_distance, cc_jitter_mean_ns=cc_jitter_mean_ns, rng=rng_ba)
    network.add_connection(bob, alice, connection=c_conn_reverse, label="classical_reverse",
                           port_name_node1="cout_alice", port_name_node2="cin_bob")


    q_conn = ExternalEntanglingConnection(length=node_distance,fidelity=entanglement_fidelity)
    port_ac, port_bc = network.add_connection(
        alice, bob,
        connection=q_conn,
        label="quantum",
        port_name_node1="qin_Alice",  
        port_name_node2="qin_Bob"     
    )


    alice.ports[port_ac].forward_input(alice.qmemory.ports['qin3'])

    bob.ports[port_bc].forward_input(bob.qmemory.ports['qin0'])

    return network
