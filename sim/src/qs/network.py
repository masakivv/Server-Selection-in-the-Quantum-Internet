# Cell実験用 測定をでぽられーとにしてみた
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

class ClassicalConnection(Connection):
    """A connection that transmits classical messages in one direction, from A to B."""
    def __init__(self, length, name="ClassicalConnection"):
        super().__init__(name=name)
        self.add_subcomponent(
            ClassicalChannel("Channel_A2B", length=length,
                             models={"delay_model": FibreDelayModel(c=200000)}),
            forward_input=[("A", "send")],
            forward_output=[("B", "recv")]
        )
#
# 9) ネットワークを設定する関数を修正してゲート速度ファクターを受け入れるようにする
#
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
    *,
    timing_mode: str = "factor",
    gate_time_ns: "Optional[float]" = None,
):
    """ネットワークのセットアップを行う関数。
    timing_mode == "absolute" の場合は gate_time_ns で測定/1q/2qすべての時間を統一設定する。
    """
    gate_durations_override = None
    # 絶対時間指定: 測定/1q/2q を同一値で上書き
    if timing_mode == "absolute" and gate_time_ns is not None:
        # デフォルトキー名称に合わせた辞書を作成
        from netsquid.components import instructions as instr

        t = float(gate_time_ns)
        gate_durations_override = {
            instr.INSTR_INIT: 1000,  # 初期化は既定値維持（影響小）
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
        # 絶対時間が指定された場合はスケールファクターを無効化
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

    # 古典接続
    c_conn = ClassicalConnection(length=node_distance)
    network.add_connection(alice, bob, connection=c_conn, label="classical",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    c_conn_reverse = ClassicalConnection(length=node_distance)
    network.add_connection(bob, alice, connection=c_conn_reverse, label="classical_reverse",
                           port_name_node1="cout_alice", port_name_node2="cin_bob")

    # 量子チャネルの接続
    q_conn = ExternalEntanglingConnection(length=node_distance,fidelity=entanglement_fidelity)
    port_ac, port_bc = network.add_connection(
        alice, bob,
        connection=q_conn,
        label="quantum",
        port_name_node1="qin_Alice",  # Alice側ポート名を一意に
        port_name_node2="qin_Bob"     # Bob側ポート名を一意に
    )

    # Aliceの "qin_Alice" → qmemoryの 'qin3'
    alice.ports[port_ac].forward_input(alice.qmemory.ports['qin3'])
    # Bobの "qin_Bob" → qmemoryの 'qin0'
    bob.ports[port_bc].forward_input(bob.qmemory.ports['qin0'])

    return network
