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



# グローバルフラグの初期値
# def global_set():
#     global global_finished_bell
#     global global_finished_correction
#     global_finished_bell = False
#     global_finished_correction = False
# global_finished_bell = False
# global_finished_correction = False
#
# 1) メモリが空いているかどうかをチェックする関数
#
def memory_is_available(mem_position):
    """メモリポジションに qubit が載っていない (is_empty=True) かどうかだけを確認"""
    # print(f"メモリ位置 {mem_position} の空き状況: {mem_position.is_empty}")
    return mem_position.is_empty

#
# 2) EXTERNALモードで動くQSourceを持つためのConnectionクラス(改変後)
#
class ExternalEntanglingConnection(Connection):
    """
    QSourceを内蔵しないで、量子チャネルだけ持つ Connection。
    ただしサブコンポーネントとして QSource を追加する。
    """
    def __init__(self, length, name="ExternalEntanglingConnection", fidelity=1.0):
        super().__init__(name=name)

        # 量子チャネルの設定
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
        # サブコンポーネントに追加
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])

        # fidelityを指定してQSourceを作成
        qsource = create_external_qsource(name="AliceQSource", fidelity=fidelity)
        self.add_subcomponent(qsource)

        # QSourceのポートをQuantumChannelの送信ポートに接続
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])

#
# 3) QSourceを段階的に待機してトリガーするプロトコル　アリスとボブの両方を考慮
#
class ExternalSourceProtocol(NodeProtocol):
    """
    - 最初に base_delay [ns] 待つ
    - アリスとボブのメモリが空いていなければ extra_delay [ns] を足して再度待機
    - 両方のメモリが空き次第 qsource.trigger() する
    """

    def __init__(self, node, qsource, other_node, mem_pos_a=3, mem_pos_b=0,
                 base_delay=1e9/200, extra_delay=1e6, max_retries=3000000):
        """
        Parameters
        ----------
        node : Node
            プロトコルを動かす側のノード (アリス)
        qsource : QSource
            EXTERNAL モードで生成した QSource
        other_node : Node
            メモリ状態をチェックする他のノード (ボブ)
        mem_pos_a : int
            アリスのメモリポジション番号
        mem_pos_b : int
            ボブのメモリポジション番号
        base_delay : float
            最初に待つ時間 [ns]
        extra_delay : float
            メモリが空いていなかった場合に追加で待機する時間 [ns]
        max_retries : int
            何度まで再待機を繰り返すか
        """
        super().__init__(node)
        self.qsource = qsource
        self.other_node = other_node
        self.mem_pos_a = mem_pos_a
        self.mem_pos_b = mem_pos_b
        self.base_delay = base_delay
        self.extra_delay = extra_delay
        self.max_retries = max_retries

    def _debug_delay(self, label, delay):
        return

    def run(self):
        # まず base_delay 待つ
        while True:
            self._debug_delay("base_delay", self.base_delay)
            yield self.await_timer(self.base_delay)

            retries = 0
            while True:
                # アリスのメモリポジションを確認
                mem_position_a = self.node.qmemory.mem_positions[self.mem_pos_a]
                # ボブのメモリポジションを確認
                mem_position_b = self.other_node.qmemory.mem_positions[self.mem_pos_b]

                if memory_is_available(mem_position_a) and memory_is_available(mem_position_b):
                    # 両方のメモリが空いていれば、QSourceをトリガーしてペア生成
                    # print(f"[{ns.sim_time()} ns] アリスとボブのメモリが空いたので qsource.trigger() します。")
                    self.qsource.trigger()
                    break
                else:
                    if retries >= self.max_retries:
                        # print(f"[{ns.sim_time()} ns] メモリ空き待ちリトライ上限（max_retries={self.max_retries}）に達しました。中断します。")
                        break
                    # まだ空いていない → extra_delay 待って再チェック
                    # print(f"[{ns.sim_time()} ns] メモリが埋まっているので {self.extra_delay} ns 待つ (retries={retries})")
                    retries += 1
                    self._debug_delay("extra_delay", self.extra_delay)
                    yield self.await_timer(self.extra_delay)

            # print(f"[{ns.sim_time()} ns] ExternalSourceProtocol 終了")

#
# 4) ClassicalConnection クラス (変更なし)
#
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
    """ゲート速度を可変にした量子プロセッサを作成するファクトリ関数"""
    if gate_durations is None:
        # デフォルトのゲート実行時間 [ns]
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
    
    # ゲート速度ファクターを適用
    scaled_gate_durations = {k: v / gate_speed_factor for k, v in gate_durations.items()}
    
    # メモリノイズモデル
    memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)
    
    
    # ゲートごとのノイズモデルを設定
    gate_noise_models = {
        instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge,time_independent=True),  # Xゲートのフィデリティ99%
        instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True), # Hゲートのフィデリティ99.5%
        instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_CNOT: DepolarNoiseModel(depolar_rate=dge,time_independent=True),
        # 他のゲートも同様に設定可能
    }
    
    # PhysicalInstructionのリストを作成
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
    
    # 量子プロセッサの作成
    processor = QuantumProcessor("quantum_processor", num_positions=4,
                                 memory_noise_models=[memory_noise_model] * 4,
                                 phys_instructions=physical_instructions)
    return processor

# def create_client_processor(dephase_rate=0.0039,T1=1e10,T2_ratio=0.1,sge=None,dge=None,gate_speed_factor=1.0, gate_durations=None):
#     """ゲート速度を可変にした量子プロセッサを作成するファクトリ関数"""
#     if gate_durations is None:
#         # デフォルトのゲート実行時間 [ns]
#         gate_durations = {
#             instr.INSTR_INIT: 1000,
#             instr.INSTR_H: 135000,
#             instr.INSTR_X: 135000,
#             instr.INSTR_Z: 135000,
#             instr.INSTR_Y: 135000,
#             instr.INSTR_S: 135000,
#             instr.INSTR_ROT_X: 135000,
#             instr.INSTR_ROT_Y: 135000,
#             instr.INSTR_ROT_Z: 135000,
#             instr.INSTR_CNOT: 600000,
#             instr.INSTR_MEASURE: 200000
#         }
    
#     # ゲート速度ファクターを適用
#     scaled_gate_durations = {k: v / gate_speed_factor for k, v in gate_durations.items()}
    
#     # メモリノイズモデル
#     memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)
#     gate_noise_models = {
#         instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge,time_independent=True),  # Xゲートのフィデリティ99%
#         instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True), # Hゲートのフィデリティ99.5%
#         instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_CNOT: DepolarNoiseModel(depolar_rate=dge,time_independent=True),
#         # 他のゲートも同様に設定可能
#     }
    
#     # PhysicalInstructionのリストを作成
#     physical_instructions = [
#         PhysicalInstruction(instr.INSTR_ROT_Z, duration=scaled_gate_durations[instr.INSTR_ROT_Z], parallel=True, topology=[0]),
#         PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[0],
#                            quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
#                            apply_q_noise_after=False)
                           
#     ]
    
#     # 量子プロセッサの作成
#     processor = QuantumProcessor("quantum_processor", num_positions=1,
#                                  memory_noise_models=[memory_noise_model] * 1,
#                                  phys_instructions=physical_instructions)
#     return processor

def create_client_processor(dephase_rate=0.0039, T1=1e10, T2_ratio=0.1, sge=None, dge=None, gate_speed_factor=1.0, gate_durations=None):
    """クライアント用の量子プロセッサを作成するファクトリ関数"""
    if gate_durations is None:
        # デフォルトのゲート実行時間 [ns]
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
    
    # ゲート速度ファクターを適用
    scaled_gate_durations = {k: v / gate_speed_factor for k, v in gate_durations.items()}
    
    # メモリノイズモデル
    memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)
    
    # ゲートノイズモデル
    gate_noise_models = {
        instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge, time_independent=True)
    }
    
    # PhysicalInstructionのリスト - クライアントプログラムが必要とする命令を全て含める
    physical_instructions = [
        # ClientProgramが使用するH命令を追加
        PhysicalInstruction(instr.INSTR_H, duration=scaled_gate_durations[instr.INSTR_H], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_H)),
        
        # ROT_Z命令は必須
        PhysicalInstruction(instr.INSTR_ROT_Z, duration=scaled_gate_durations[instr.INSTR_ROT_Z], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_ROT_Z)),
        
        # 念のため他の単一量子ビット命令も追加
        PhysicalInstruction(instr.INSTR_X, duration=scaled_gate_durations[instr.INSTR_X], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_X)),
        
        PhysicalInstruction(instr.INSTR_Z, duration=scaled_gate_durations[instr.INSTR_Z], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_Z)),
        
        PhysicalInstruction(instr.INSTR_Y, duration=scaled_gate_durations[instr.INSTR_Y], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_Y)),
        
        # 測定命令
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], 
                          parallel=False, topology=[0],
                          quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                          apply_q_noise_after=False)
    ]
    
    # 量子プロセッサの作成 - クライアントは1量子ビットのみ
    processor = QuantumProcessor("client_quantum_processor", num_positions=1,
                                memory_noise_models=[memory_noise_model],
                                phys_instructions=physical_instructions)
    return processor


#
# 6) もつれ生成用の QSource を EXTERNAL で作るファクトリ
#
def create_external_qsource(name="ExtQSource", fidelity=1.0):
    """EXTERNALモードのQSourceを作成するファクトリ関数。
    fidelity < 1.0 の場合、生成されるベル対に Depolarizing Noise を適用し、忠実度を下げる。
    """
    from netsquid.qubits import ketstates as ks
    
    # Fidelity F を depolarizing の確率 p に変換
    p_depol = 4/3 * (1 - fidelity)
    if p_depol < 0:
        p_depol = 0.0  # fidelityが1.0を超えたりした場合の保護（無音で切り捨てる）
    
    qsource = QSource(
        name=name,
        state_sampler=StateSampler([ks.b00], [1.0]),
        num_ports=2,
        status=SourceStatus.EXTERNAL,
        models={
            # 生成した瞬間に Depolarizing チャネルを通過したとみなす
            "emission_noise_model": DepolarNoiseModel(
                depolar_rate=p_depol,
                time_independent=True  # 時間に依存しないノイズ
            )
        }
    )
    return qsource
