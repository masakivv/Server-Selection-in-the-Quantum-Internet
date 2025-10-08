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

#
# 7) 既存プログラム類 (InitStateProgram, BellMeasurementProgram 等) は変更なし
#
class InitStateProgram(QuantumProgram):
    default_num_qubits = 3
    def program(self):
        q1, q2, q3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_INIT, q2)
        self.apply(instr.INSTR_INIT, q3)
        self.apply(instr.INSTR_ROT_Y, q1, angle=np.pi / 2)
        self.apply(instr.INSTR_X, q2)
        self.apply(instr.INSTR_ROT_X, q2, angle=-np.pi / 2)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_CNOT, [q2, q3])
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_ROT_Y, q1, angle=-np.pi / 2)
        self.apply(instr.INSTR_ROT_X, q2, angle=np.pi / 2)
        yield self.run()


class BellMeasurementProgram(QuantumProgram):
    """2量子ビットに対するベル測定を行うプログラム"""
    default_num_qubits = 4
    def program(self):
        q1, q2, q3, q4 = self.get_qubit_indices(4)
        self.apply(instr.INSTR_CNOT, [q3, q4])
        self.apply(instr.INSTR_H, q3)
        self.apply(instr.INSTR_MEASURE, q3, output_key="M3")
        self.apply(instr.INSTR_MEASURE, q4, output_key="M4")
        yield self.run()


class ClientProgram(QuantumProgram):
    """1量子ビットを測定するだけの簡単プログラム"""
    default_num_qubits = 1
    def program(self):
        # q1 = self.get_qubit_indices(1)[0]
        q1 = self.get_qubit_indices(1)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M5")
        yield self.run()


class ServerProgram(QuantumProgram):
    """2量子ビットを測定するだけの簡単プログラム"""
    default_num_qubits = 2
    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
        yield self.run()


class M5XServerProgram(QuantumProgram):
    default_num_qubits = 2
    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_X, q2)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
        yield self.run()


class Collect1Program(QuantumProgram):
    default_num_qubits = 3
    def program(self):
        q1, q2, q3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_CNOT, [q1, q3])
        self.apply(instr.INSTR_H, q1)
        yield self.run()


class Collect2Program(QuantumProgram):
    default_num_qubits = 3
    def program(self):
        q1, q2, q3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_CNOT, [q2, q3])
        self.apply(instr.INSTR_H, q2)
        yield self.run()

class XmeasureProgram(QuantumProgram):
    default_num_qubits = 2
    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_H, q2)
        yield self.run()