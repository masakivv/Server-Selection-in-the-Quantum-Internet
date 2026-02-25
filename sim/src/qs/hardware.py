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
from typing import Optional


def memory_is_available(mem_position):

    return mem_position.is_empty


class ExternalEntanglingConnection(Connection):
    def __init__(self, length, name="ExternalEntanglingConnection", fidelity=1.0):
        super().__init__(name=name)


        qchannel_c2a = QuantumChannel(
            "qchannel_C2A",
            length=length / 2,
            models={
                "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0),
                "quantum_noise_model": DepolarNoiseModel(depolar_rate=0)
            }
        )
        qchannel_c2b = QuantumChannel(
            "qchannel_C2B",
            length=length / 2,
            models={
                "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0),
                "quantum_noise_model": DepolarNoiseModel(depolar_rate=0)
            }
        )

        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])


        qsource = create_external_qsource(name="AliceQSource", fidelity=fidelity)
        self.add_subcomponent(qsource)


        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])


class ExternalSourceProtocol(NodeProtocol):

    def __init__(
        self,
        node,
        qsource,
        other_node,
        mem_pos_a=3,
        mem_pos_b=0,
        base_delay=1e9 / 200,
        extra_delay=1e6,
        max_retries=3000000,
        *,
        stochastic=False,
        rng=None,
        stochastic_min_delay=1.0,
        run_id: Optional[str] = None,
        seed_label: Optional[str] = None,
        server_id: Optional[str] = None,
        link_id: Optional[str] = None,
        entanglement_speed_factor: Optional[float] = None,
    ):
        super().__init__(node)
        self.qsource = qsource
        self.other_node = other_node
        self.mem_pos_a = mem_pos_a
        self.mem_pos_b = mem_pos_b
        self.base_delay = base_delay
        self.extra_delay = extra_delay
        self.max_retries = max_retries
        self.stochastic = stochastic
        self.rng = rng
        self.stochastic_min_delay = stochastic_min_delay
        self.run_id = run_id or ""
        self.seed_label = seed_label or ""
        self.server_id = server_id or (getattr(node, "name", None) or "")
        self.link_id = link_id or f"{self.server_id}-{getattr(other_node, 'name', None) or ''}"
        self.entanglement_speed_factor = entanglement_speed_factor
        self.pair_seq_id = 0

        self.entanglement_delay_samples_ns = []

    def _debug_delay(self, label, delay):
        return

    def run(self):
        DEBUG_DELAY_LOG_THRESHOLD = 1e12  

        while True:
            current_pair_id = self.pair_seq_id
            if self.stochastic:
                if self.rng is None:
                    self.rng = np.random.default_rng()
                delay = self.rng.exponential(scale=self.base_delay)
                if self.stochastic_min_delay is not None:
                    delay = max(delay, self.stochastic_min_delay)
            else:
                delay = self.base_delay

            self.entanglement_delay_samples_ns.append(delay)
            self._debug_delay("base_delay", delay)
            if delay > DEBUG_DELAY_LOG_THRESHOLD:
                print(
                    f"[ExtSrc] large base delay {delay} ns "
                    f"(esf={self.entanglement_speed_factor}, "
                    f"stochastic={self.stochastic}, "
                    f"run={self.run_id}, seed={self.seed_label}, pair={current_pair_id})"
                )
            yield self.await_timer(delay)

            retries = 0
            while True:

                mem_position_a = self.node.qmemory.mem_positions[self.mem_pos_a]

                mem_position_b = self.other_node.qmemory.mem_positions[self.mem_pos_b]

                if memory_is_available(mem_position_a) and memory_is_available(mem_position_b):


                    self.qsource.trigger()
                    self.pair_seq_id += 1
                    break
                else:
                    if retries >= self.max_retries:
                        print(
                            f"[ExtSrc] max_retries reached after "
                            f"{retries} attempts (extra_delay={self.extra_delay} ns, "
                            f"run={self.run_id}, seed={self.seed_label}, pair={current_pair_id})"
                        )

                        break


                    retries += 1
                    self._debug_delay("extra_delay", self.extra_delay)
                    yield self.await_timer(self.extra_delay)


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


from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components import instructions as instr
from netsquid.components.models.qerrormodels import T1T2NoiseModel

def create_processor(dephase_rate=0.0039,T1=1e10,T2_ratio=0.1,sge=None,dge=None,gate_speed_factor=1.0, gate_durations=None):
    if gate_durations is None:

        gate_durations = {
            instr.INSTR_INIT: 1000,
            instr.INSTR_H: 135000,
            instr.INSTR_X: 135000,
            instr.INSTR_Z: 135000,
            instr.INSTR_Y: 135000,
            instr.INSTR_S: 135000,
            instr.INSTR_ROT_X: 135000,
            instr.INSTR_ROT_Y: 135000,
            instr.INSTR_ROT_Z: 135000,
            instr.INSTR_CNOT: 600000,
            instr.INSTR_MEASURE: 200000
        }


    scaled_gate_durations = {k: v * gate_speed_factor for k, v in gate_durations.items()}


    memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)


    gate_noise_models = {
        instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge,time_independent=True),  
        instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True), 
        instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_CNOT: DepolarNoiseModel(depolar_rate=dge,time_independent=True),

    }


    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=scaled_gate_durations[instr.INSTR_INIT], parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=scaled_gate_durations[instr.INSTR_H], parallel=True, topology=[0, 1, 2, 3],
                           quantum_noise_model=gate_noise_models.get(instr.INSTR_H, None)),
        PhysicalInstruction(instr.INSTR_X, duration=scaled_gate_durations[instr.INSTR_X], parallel=True, topology=[0, 1, 2, 3],
                           quantum_noise_model=gate_noise_models.get(instr.INSTR_X, None)),
        PhysicalInstruction(instr.INSTR_Z, duration=scaled_gate_durations[instr.INSTR_Z], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_Y, duration=scaled_gate_durations[instr.INSTR_Y], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_S, duration=scaled_gate_durations[instr.INSTR_S], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_ROT_X, duration=scaled_gate_durations[instr.INSTR_ROT_X], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_ROT_Y, duration=scaled_gate_durations[instr.INSTR_ROT_Y], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_ROT_Z, duration=scaled_gate_durations[instr.INSTR_ROT_Z], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(0, 1)]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(0, 2)]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(1, 2)]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(2, 3)]),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[0],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[1],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[2],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[3],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False)
    ]


    processor = QuantumProcessor("quantum_processor", num_positions=4,
                                 memory_noise_models=[memory_noise_model] * 4,
                                 phys_instructions=physical_instructions)
    return processor


def create_client_processor(dephase_rate=0.0039, T1=1e10, T2_ratio=0.1, sge=None, dge=None, gate_speed_factor=1.0, gate_durations=None):
    if gate_durations is None:

        gate_durations = {
            instr.INSTR_INIT: 1000,
            instr.INSTR_H: 135000,
            instr.INSTR_X: 135000,
            instr.INSTR_Z: 135000,
            instr.INSTR_Y: 135000,
            instr.INSTR_S: 135000,
            instr.INSTR_ROT_X: 135000,
            instr.INSTR_ROT_Y: 135000,
            instr.INSTR_ROT_Z: 135000,
            instr.INSTR_CNOT: 600000,
            instr.INSTR_MEASURE: 200000
        }


    scaled_gate_durations = {k: v * gate_speed_factor for k, v in gate_durations.items()}


    memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)


    gate_noise_models = {
        instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge, time_independent=True)
    }


    physical_instructions = [

        PhysicalInstruction(instr.INSTR_H, duration=scaled_gate_durations[instr.INSTR_H], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_H)),


        PhysicalInstruction(instr.INSTR_ROT_Z, duration=scaled_gate_durations[instr.INSTR_ROT_Z], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_ROT_Z)),


        PhysicalInstruction(instr.INSTR_X, duration=scaled_gate_durations[instr.INSTR_X], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_X)),

        PhysicalInstruction(instr.INSTR_Z, duration=scaled_gate_durations[instr.INSTR_Z], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_Z)),

        PhysicalInstruction(instr.INSTR_Y, duration=scaled_gate_durations[instr.INSTR_Y], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_Y)),


        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], 
                          parallel=False, topology=[0],
                          quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                          apply_q_noise_after=False)
    ]


    processor = QuantumProcessor("client_quantum_processor", num_positions=1,
                                memory_noise_models=[memory_noise_model],
                                phys_instructions=physical_instructions)
    return processor


def create_external_qsource(name="ExtQSource", fidelity=1.0):
    from netsquid.qubits import ketstates as ks


    p_depol = 4/3 * (1 - fidelity)
    if p_depol < 0:
        p_depol = 0.0  

    qsource = QSource(
        name=name,
        state_sampler=StateSampler([ks.b00], [1.0]),
        num_ports=2,
        status=SourceStatus.EXTERNAL,
        models={

            "emission_noise_model": DepolarNoiseModel(
                depolar_rate=p_depol,
                time_independent=True  
            )
        }
    )
    return qsource
