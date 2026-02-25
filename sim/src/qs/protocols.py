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
from .programs import (
    InitStateProgram,
    BellMeasurementProgram,
    ServerProgram,
    M5XServerProgram,
    Collect1Program,
    Collect2Program,
    XmeasureProgram,
    ClientProgram,
)
from .control import SimulationControl


def memory_status(node):
    for pos in node.qmemory.mem_positions:
        print(f"Position {pos} is_empty: {pos.is_empty}")

class BellMeasurementProtocol(NodeProtocol):


    def __init__(self, node, shot_times_list, band_times_list, server_times_list, serverlist, max_runs, flag,
                 control: SimulationControl, density_store=None):
        super().__init__(node)
        self.shot_times_list = shot_times_list
        self.band_times_list = band_times_list
        self.server_times_list = server_times_list
        self.serverlist = serverlist
        self.max_runs = max_runs  
        self.run_count = 0        
        self.flag = flag
        self.control = control
        self.density_store = density_store
        self._density_taken = False

    def _capture_density_matrix(self):
        """Capture server two-qubit density matrix before measurement if requested."""
        if self.density_store is None or self._density_taken:
            return
        try:
            qubits = list(self.node.qmemory.peek([0, 1]))
        except Exception:
            return
        if not qubits or any(q is None for q in qubits):
            return
        try:
            rho = qapi.reduced_dm(qubits)
        except Exception:
            return
        self.density_store.append(np.asarray(rho).copy())
        self._density_taken = True

    def run(self):

        qubit_initialised = False
        entanglement_ready = False
        qubit_init_program = InitStateProgram()
        measure_program = BellMeasurementProgram()
        server = ServerProgram()
        m5xserver = M5XServerProgram()
        collect1 = Collect1Program()
        collect2 = Collect2Program()
        xmesure = XmeasureProgram()

        init = True
        round_base = self.control.bob_round
        sync_poll_ns = 1e5  
        b1 = []
        b2 = []
        b3 = []
        b4 = []
        b5 = []
        astate=0
        start_time = ns.sim_time()
        self._density_taken = False

        self.node.qmemory.execute_program(qubit_init_program)

        while True:


            if not init:
                yield self.await_timer(10)
                start_time = ns.sim_time()
                round_base = self.control.bob_round

                self.node.qmemory.execute_program(qubit_init_program)
                init = True
                qubit_index_4 = 3
                qubit_4, = self.node.qmemory.pop(qubit_index_4)
                qapi.discard(qubit_4)
                self._density_taken = False
            expr = yield (self.await_program(self.node.qmemory) |
                          self.await_port_input(self.node.ports["qin_Alice"]))  
            if expr.first_term.value:
                qubit_initialised = True

                s1 = ns.sim_time() - start_time
            else:
                entanglement_ready = True

                b1.append(ns.sim_time())
            if qubit_initialised and entanglement_ready:

                qubit_initialised = False
                entanglement_ready = False
                e1 = ns.sim_time()
                yield self.node.qmemory.execute_program(measure_program)
                s2 = ns.sim_time() - e1

                m3, = measure_program.output["M3"]
                m4, = measure_program.output["M4"]

                qubit_index_3 = 2
                qubit_index_4 = 3
                qubit_4, = self.node.qmemory.pop(qubit_index_4)
                qapi.discard(qubit_4)
                self.node.ports["cout_bob"].tx_output((m3, m4))
                expected_round_1 = round_base + 1
                astate = 1
            while astate==1:
                if self.control.bob_round >= expected_round_1:
                    e2 = ns.sim_time()
                    self.node.qmemory.execute_program(collect1)
                    astate = 10
                    continue
                expr = yield (self.await_timer(sync_poll_ns) |
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.second_term.value:
                    entanglement_ready = True
                    b2.append(ns.sim_time())
            while astate==10:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"])) 
                if expr.first_term.value:
                    qubit_initialised = True

                    s3 = ns.sim_time() - e2


                else:
                    entanglement_ready = True

                    b2.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False

                    e3 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s4 = ns.sim_time() - e3

                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_4)
                    qapi.discard(qubit_4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))
                    expected_round_2 = round_base + 2
                    astate = 2
            while astate==2:
                if self.control.bob_round >= expected_round_2:
                    e4 = ns.sim_time()
                    self.node.qmemory.execute_program(collect1)
                    astate = 11
                    continue
                expr = yield (self.await_timer(sync_poll_ns) |
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.second_term.value:
                    entanglement_ready = True
                    b3.append(ns.sim_time())
            while astate==11:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"]))  
                if expr.first_term.value:
                    qubit_initialised = True
                    s5 = ns.sim_time() - e4
                else:
                    entanglement_ready = True

                    b3.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False
                    e5 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s6 = ns.sim_time() - e5

                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_4)
                    qapi.discard(qubit_4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))
                    expected_round_3 = round_base + 3
                    astate = 3


            while astate==3:
                if self.control.bob_round >= expected_round_3:
                    e6 = ns.sim_time()
                    self.node.qmemory.execute_program(collect2)
                    astate = 12
                    continue
                expr = yield (self.await_timer(sync_poll_ns) |
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.second_term.value:
                    entanglement_ready = True
                    b4.append(ns.sim_time())
            while astate==12:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"]))  
                if expr.first_term.value:
                    qubit_initialised = True
                    s7 = ns.sim_time() - e6
                else:
                    entanglement_ready = True

                    b4.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False
                    e7 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s8 = ns.sim_time() - e7

                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_4)
                    qapi.discard(qubit_4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))
                    expected_round_4 = round_base + 4
                    astate = 4
            while astate==4:
                if self.control.bob_round >= expected_round_4:
                    e8 = ns.sim_time()
                    self.node.qmemory.execute_program(collect2)
                    e9 = ns.sim_time()
                    astate = 13
                    continue
                expr = yield (self.await_timer(sync_poll_ns) |
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.second_term.value:
                    entanglement_ready = True
                    b5.append(ns.sim_time())
            while astate==13:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"]))  
                if expr.first_term.value:
                    qubit_initialised = True
                    s9 = ns.sim_time() - e8
                else:
                    entanglement_ready = True

                    b5.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False
                    e9 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s10 = ns.sim_time() - e9
                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_3)  
                    qapi.discard(qubit_4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))

                    expected_round_5 = round_base + 5
                    astate = 5
            while astate==5:
                if self.control.bob_round < expected_round_5:
                    yield self.await_timer(sync_poll_ns)
                    continue
                e10 = ns.sim_time()
                if self.flag == 1:
                    yield self.node.qmemory.execute_program(xmesure)
                self._capture_density_matrix()
                yield self.node.qmemory.execute_program(server)
                m1, = server.output["M1"]
                m2, = server.output["M2"]
                s11 = ns.sim_time() - e10
                self.serverlist.append((m1,m2))
                end_time = ns.sim_time()
                run_time = end_time - start_time


                band_time = (b1[0]-start_time) + (b2[0]-b1[-1]) + (b3[0]-b2[-1]) + (b4[0]-b3[-1]) + (b5[0]-b4[-1])
                self.band_times_list.append(band_time)
                self.shot_times_list.append(run_time)
                self.server_times_list.append(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11)
                b1 = []
                b2 = []
                b3 = []
                b4 = []
                b5 = []
                self.send_signal(Signals.SUCCESS, 0)
                self.run_count += 1
                if self.run_count < self.max_runs:
                    qubit_initialised = False
                    entanglement_ready = False
                    init = False
                    self._density_taken = False
                    astate=0
                    qubit_index_1 = 0
                    qubit_index_2 = 1
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    round_base = self.control.bob_round

                else:


                    self.control.finished_bell = True
                    self.control.stop_if_done()


class CorrectionProtocol(NodeProtocol):

    def __init__(self, node, clientlist, telelist, max_runs, parameter, control: SimulationControl):
        super().__init__(node)
        self.clientlist = clientlist
        self.parameter = parameter
        self.telelist = telelist
        self.max_runs = max_runs  
        self.run_count = 0        
        self.control = control
    def run(self):

        port_alice = self.node.ports["cin_alice"]
        port_bob = self.node.ports["qin_Bob"]        

        entanglement_ready = False
        meas_results = None
        m5_result = None
        program = ClientProgram()
        state=0
        angle=self.parameter
        while True:
            expr = yield (self.await_port_input(port_alice) | self.await_port_input(port_bob))
            if expr.first_term.value:
                meas_results, = port_alice.rx_input().items


            else:
                entanglement_ready = True


            if meas_results is not None and entanglement_ready:
                angle=self.parameter
                if meas_results[0] == 1:
                    angle=angle+np.pi
                if meas_results[1] == 1:
                    angle=-angle

                self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, [0], angle=angle)
                yield self.await_program(self.node.qmemory)

                yield self.node.qmemory.execute_program(program)

                qubit_index_5 = 0
                qubit_5, = self.node.qmemory.pop(qubit_index_5)  
                qapi.discard(qubit_5)
                m5, = program.output["M5"]
                m5_result = m5
                tele1 = meas_results
                self.control.mark_bob_round_done()

                entanglement_ready = False
                meas_results = None
                state=1

            while state==1:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items


                else:

                    entanglement_ready = True

                if meas_results is not None and entanglement_ready:
                    angle=0
                    if m5_result == 1:
                        angle=np.pi
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  
                    qapi.discard(qubit_5)
                    m5, = program.output["M5"]
                    m5_2=m5
                    tele2 = meas_results
                    self.control.mark_bob_round_done()

                    entanglement_ready = False
                    meas_results = None

                    state=2

            while state==2:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items


                else:
                    entanglement_ready = True
                if meas_results is not None and entanglement_ready:
                    angle=0
                    if m5_result == 1:
                        angle=np.pi
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  
                    qapi.discard(qubit_5)
                    m5, = program.output["M5"]
                    m5_3 = m5
                    tele3 = meas_results
                    self.control.mark_bob_round_done()

                    entanglement_ready = False
                    meas_results = None
                    state=3

            while state==3:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items


                else:
                    entanglement_ready = True
                if meas_results is not None and entanglement_ready:
                    angle=0
                    if m5_result == 1:
                        angle=0
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  
                    qapi.discard(qubit_5)
                    m5, = program.output["M5"]
                    m5_4 = m5
                    tele4 = meas_results

                    self.control.mark_bob_round_done()

                    entanglement_ready = False
                    meas_results = None
                    state=4

            while state==4:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items


                else:
                    entanglement_ready = True
                if meas_results is not None:

                    angle=0
                    if m5_result == 1:
                        angle=np.pi
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    m5, = program.output["M5"]
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  
                    qapi.discard(qubit_5)
                    m5_5 = m5
                    tele5 = meas_results
                    client_tuple = (m5_result, m5_2, m5_3, m5_4, m5_5)
                    tele_tuple = (tele1, tele2, tele3, tele4, tele5)
                    self.clientlist.append(client_tuple)
                    self.telelist.append(tele_tuple)
                    self.control.mark_bob_round_done()
                    state=5
                    self.send_signal(Signals.SUCCESS, 0)
                    self.run_count += 1

                    if self.run_count < self.max_runs:
                        qubit_initialised = False
                        entanglement_ready = False
                        init = False
                        meas_results = None

                    else:


                        self.control.finished_correction = True
                        self.control.stop_if_done()
