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

from .network import example_network_setup
from .protocols import BellMeasurementProtocol, CorrectionProtocol
from .hardware import ExternalSourceProtocol
from .control import SimulationControl

#
# 10) シミュレーションのセットアップ関数を修正してエンタングルメント速度ファクターを受け入れるようにする
#
import netsquid as ns, numpy as np, random, os
def initialize_seeds(base_seed: int, worker_rank: int = 0):
    """すべての RNG を同じ系列で初期化するヘルパ."""
    seed = base_seed + worker_rank     # 並列実行なら rank でずらす
    ns.set_random_state(seed)          # NetSquid 内部 RNG
    np.random.seed(seed)               # NumPy
    random.seed(seed)                  # Python 標準
    return seed
def example_sim_setup(node_A, node_B, shot_times_list_alice, band_times_list_alice, server_times_list_alice,
                     max_runs,gate_speed_factor=1.0, entanglement_speed_factor=1.0,serverlist=None,clientlist=None,
                     telelist=None,parameter=0,flag=0,control=None,density_matrices=None,base_seed=42):
    """シミュレーションのセットアップを行う関数。ゲート速度ファクターとエンタングルメント速度ファクターを受け入れるように修正"""
    def collect_fidelity_data(evexpr):
        protocol = evexpr.triggered_events[-1].source
        mem_pos = protocol.get_signal_result(Signals.SUCCESS)
        q1, = protocol.node.qmemory.pop(mem_pos)
        q2, = protocol.node.qmemory.pop(mem_pos+1)
        fidelity = qapi.fidelity(q2, ns.qubits.ketstates.s1, squared=True)
        qapi.discard(q2)
        return {"fidelity": fidelity}
    
    initialize_seeds(base_seed)
    if control is None:
        control = SimulationControl()

    # BellMeasurementProtocol (Alice) と CorrectionProtocol (Bob)
    # protocol_alice = BellMeasurementProtocol(node_A, shot_times_list_alice, band_times_list_alice, server_times_list_alice,serverlist,max_runs,flag)
    # protocol_bob = CorrectionProtocol(node_B,clientlist,telelist,max_runs,parameter=parameter)
    protocol_alice = BellMeasurementProtocol(
        node_A,
        shot_times_list_alice,
        band_times_list_alice,
        server_times_list_alice,
        serverlist,
        max_runs,
        flag,
        control,
        density_store=density_matrices,
    )
    protocol_bob = CorrectionProtocol(node_B, clientlist, telelist, max_runs, parameter=parameter, control=control)
    # ノード_Aが属するネットワークを取得
    network = node_A.supercomponent  # Teleportation_network

    # 量子接続は label="quantum" で追加
    q_conn = network.get_connection(node_A, node_B, label="quantum")

    # サブコンポーネントから QSource を取得
    qsource = q_conn.subcomponents.get("AliceQSource")

    if qsource is None:
        raise ValueError("AliceQSource が見つかりませんでした。サブコンポーネントの名前を確認してください。")

    # EXTERNALモードのQSourceを段階的にトリガーするプロトコル
    ext_source_protocol = ExternalSourceProtocol(
        node=node_A,
        qsource=qsource,  
        other_node=node_B,
        mem_pos_a=3,
        mem_pos_b=0,
        base_delay=1e9  / entanglement_speed_factor,  # オリジナルの base_delay をスケーリング
        extra_delay=1e6,        # オリジナルの extra_delay をスケーリング
        max_retries=int(1e3)
    )

    # DataCollector
    dc = DataCollector(collect_fidelity_data)
    dc.collect_on(pydynaa.EventExpression(source=protocol_alice, event_type=Signals.SUCCESS.value))

    # プロトコル開始
    ext_source_protocol.start()
    protocol_alice.start()
    protocol_bob.start()

    return protocol_alice, protocol_bob, ext_source_protocol, dc

#
# 11) 実験を実行する関数を修正してゲート速度ファクターとエンタングルメント速度ファクターを変数として扱うようにする
#
def convert_tuple_list_to_counts(tuple_list):
    """
    例: [(0,1), (1,0), (0,1), ...] のようなリストを受け取り、
    {'01': 回数, '10': 回数, ...} のカウント辞書にまとめる。
    """
    counts = {}
    for outcome_tuple in tuple_list:
        # タプル → 文字列 "xy" に変換（0,1 -> "01" など）
        outcome_str = "".join(str(bit) for bit in outcome_tuple)
        # カウントをインクリメント
        counts[outcome_str] = counts.get(outcome_str, 0) + 1
    
    return counts

def evaluate_prob_difference(counts, ideal_state='10'):
    """
    'ideal_state' の測定確率が 1.0 であるはず、としたときの
    「確率がどれだけ理想値(=1)とズレているか」を返す
    """
    total_shots = sum(counts.values())
    prob_ideal = counts.get(ideal_state, 0) / total_shots
    # 真の値1.0からのズレ
    return abs(prob_ideal - 1.0)

import itertools

def run_experiment(
    num_runs,
    dephase_rates,
    client_fidelitys,
    distances,
    T1s,
    client_T1s,
    T2_ratios,
    sges,
    dges,
    gate_speed_factors,
    client_gate_speed_factors,
    entanglement_fidelities,
    entanglement_speed_factors,
    max_runs,
    angle,
    flag,
    collect_density=False,
    base_seed=42,
    *,
    timing_mode: str = "factor",
    gate_time_ns: "Optional[float]" = None,
):
    # global global_finished_bell, global_finished_correction
    """
    多くのパラメータを変化させてシミュレーションを実行し、結果を収集する関数。

    Parameters
    ----------
    num_runs : int
        各パラメータ組み合わせでのシミュレーション回数。
    dephase_rates : list of float
        減衰率のリスト。
    distances : list of float
        距離のリスト（km）。
    T1s : list of float
        T1時間のリスト（ns）。
    T2s : list of float
        T2時間のリスト（ns）。
    sges : list of float
        単量子ゲートの誤り率のリスト。
    dges : list of float
        2量子ゲートの誤り率のリスト。
    gate_speed_factors : list of float
        ゲート速度ファクターのリスト。
    entanglement_fidelities : list of float
        エンタングルメントの忠実度のリスト。
    entanglement_speed_factors : list of float
        エンタングルメント速度ファクターのリスト。

    Returns
    -------
    pandas.DataFrame
        収集されたすべてのデータを含むデータフレーム。
    """
    # 結果を保存するリスト
    results = []

    # パラメータのすべての組み合わせを生成
    parameter_combinations = list(itertools.product(
        dephase_rates, distances, T1s, T2_ratios,client_T1s,client_fidelitys,client_gate_speed_factors, 
        sges, dges, gate_speed_factors, 
        entanglement_fidelities, entanglement_speed_factors
    ))

    # total_combinations = len(parameter_combinations)
    # print(f"総パラメータ組み合わせ数: {total_combinations}")

    for idx, (dephase_rate, distance, T1, T2_ratio, client_T1,client_fidelity,client_gate_speed_factor,
              sge, dge, gate_factor, 
              entanglement_fidelity, entanglement_factor) in enumerate(parameter_combinations, 1):
        # print(f"\nシミュレーション {idx}/{total_combinations}:")
        # print(f"dephase_rate={dephase_rate}, distance={distance} km, T1={T1} ns, T2={T2} ns, "
        #       f"sge={sge}, dge={dge}, gate_factor={gate_factor}, "
        #       f"entanglement_fidelity={entanglement_fidelity}, entanglement_factor={entanglement_factor}")

        # 古典通信時間を距離に基づいて計算
        cc_time = (1000 * distance / 200000 * 1e6)  # 光ファイバー中の光速を200,000 km/sと仮定

        # シミュレーションをリセット
        ns.sim_reset()
        network = example_network_setup(
            dephase_rate,
            distance,
            T1,
            T2_ratio,
            client_T1,
            sge,
            dge,
            gate_speed_factor=gate_factor,
            client_gate_speed_factor=client_gate_speed_factor,
            entanglement_fidelity=entanglement_fidelity,
            client_fidelity=client_fidelity,
            timing_mode=timing_mode,
            gate_time_ns=gate_time_ns,
        )

        node_a = network.get_node("Alice")
        node_b = network.get_node("Bob")
        shot_times_list_alice = []
        band_times_list_alice = []
        server_times_list_alice = []
        serverlist = []
        clientlist = []
        telelist =[]
        control = SimulationControl()

        density_records = [] if collect_density else None

        protocol_alice, protocol_bob, ext_source_protocol, dc = \
            example_sim_setup(node_a, node_b,
                              shot_times_list_alice, band_times_list_alice, 
                              server_times_list_alice,
                              gate_speed_factor=gate_factor, 
                              entanglement_speed_factor=entanglement_factor,
                              serverlist=serverlist,
                              clientlist=clientlist,
                              telelist=telelist,
                              max_runs=max_runs,
                              parameter=angle,
                              flag=flag,
                              control=control,
                              density_matrices=density_records,
                              base_seed=base_seed,
                              )

        # 指定された回数分シミュレーションを実行
        ns.sim_run(1e12)
        # global_finished_bell = False
        # global_finished_correction = False

        # 測定結果の処理
        if flag==0:
            for i in range(len(clientlist)):
                c = clientlist[i]
                s0, s1 = serverlist[i]
                # print(s0,s1,c)
                # clientの2番目(インデックス1)が1の時、serverの1番目(s0)を反転
                if c[2] == 1:
                    s0 = 1 - s0  # 0 → 1、 1 → 0 にトグルします

                # clientの4番目(インデックス3)が1の時、serverの2番目(s1)を反転
                if c[4] == 1:
                    s1 = 1 - s1

                # 反転した結果をserverlistに更新
                
                serverlist[i] = (s0, s1)
            # print(serverlist[i])
        if flag==1:
            for i in range(len(clientlist)):
                c = clientlist[i]
                s0, s1 = serverlist[i]
                t=telelist[i]
                # clientの2番目(インデックス1)が1の時、serverの1番目(s0)を反転
                if c[1] == 1:
                    s0 = 1 - s0

                # clientの4番目(インデックス3)が1の時、serverの2番目(s1)を反転
                if c[3] == 1:
                    s1 = 1 - s1

                # clientの5番目(インデックス4)が1の時、s0 と s1 の両方を反転
                if c[0] == 1:
                    s0 = 1 - s0
                    s1 = 1 - s1
                #t[1]=1katuc[1]==1の時
                # if c[0]==1 and t[0][1]==1:
                #     s0 = 1 - s0
                #     s1 = 1 - s1
                # if c[1]==1 and t[1][1]==1:
                #     s0 = 1 - s0
                # if c[2]==1 and t[2][1]==1:
                #     s0 = 1 - s0
                # if c[3]==1 and t[3][1]==1:
                #     s1 = 1 - s1
                # if c[4]==1 and t[4][1]==1:
                #     s1 = 1 - s1

                serverlist[i] = (s0, s1)

        # Fidelityデータを収集
        df_fidelity = dc.dataframe
        df_fidelity['distance'] = distance
        df_fidelity['gate_speed_factor'] = gate_factor
        df_fidelity['entanglement_speed_factor'] = entanglement_factor

        #     # --- シミュレーション終了直前のデバッグ出力例 ---
        # print("---- Debug: 各データリストの長さ ----")
        # print("shot_times_list_alice の長さ:", len(shot_times_list_alice))
        # print("band_times_list_alice の長さ:", len(band_times_list_alice))
        # print("server_times_list_alice の長さ:", len(server_times_list_alice))
        # print("serverlist の長さ:", len(serverlist))
        # print("clientlist の長さ:", len(clientlist))
        # print("telelist の長さ:", len(telelist))

        # 時間データを収集
        shot_times_df = pandas.DataFrame({
            "distance": [distance] * len(shot_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(shot_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(shot_times_list_alice),
            "execution_time": shot_times_list_alice
        })
        band_times_df = pandas.DataFrame({
            "distance": [distance] * len(band_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(band_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(band_times_list_alice),
            "band_time": band_times_list_alice
        })
        server_times_df = pandas.DataFrame({
            "distance": [distance] * len(server_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(server_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(server_times_list_alice),
            "server_time": server_times_list_alice
        })
        meas_df = pandas.DataFrame({
            "distance": [distance] * len(server_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(server_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(server_times_list_alice),
            "server_result": serverlist,
            "client_result": clientlist
        })

        # counts_resultの作成
        counts_result = convert_tuple_list_to_counts(serverlist)
        diff_prob = evaluate_prob_difference(counts_result, ideal_state='10')

        # 結果の保存
        for i in range(len(serverlist)):
            m = serverlist[i]
            c = clientlist[i]
            t = telelist[i]
            result = {
                "dephase_rate": dephase_rate,
                "client_fidelity": client_fidelity,
                "distance": distance,
                "T1": T1,
                "T2_ratio": T2_ratio,
                "client_T1": client_T1,
                "sge": sge,
                "dge": dge,
                "gate_speed_factor": gate_factor,
                "client_gate_speed_factor": client_gate_speed_factor,
                "entanglement_fidelity": entanglement_fidelity,
                "entanglement_speed_factor": entanglement_factor,
                "shot_time": shot_times_list_alice[i],
                "band_time": band_times_list_alice[i],
                "server_time": server_times_list_alice[i],
                "cc_time": cc_time,
                "server_result": m,
                "client_result": c,
                "teleport_result": t,
                "diff_prob": diff_prob,
                "angle" : angle
            }
            if collect_density:
                dm_value = density_records[i] if density_records and i < len(density_records) else None
                result["server_dm"] = dm_value
            results.append(result)

        # print(f"シミュレーション完了: diff_prob={diff_prob}")

    # データフレームに変換
    results_df = pandas.DataFrame(results)
    

    return results_df



def count_tuple_frequencies(data):
    freq_dict = {}
    for pair in data:
        key = "".join(map(str, pair[::-1]))  # 前後を反転
        if key in freq_dict:
            freq_dict[key] += 1
        else:
            freq_dict[key] = 1
    return freq_dict

def count_tuple_frequencies(data):
    freq_dict = {}
    for pair in data:
        key = "".join(map(str, pair[::-1]))  # 前後を反転
        if key in freq_dict:
            freq_dict[key] += 1
        else:
            freq_dict[key] = 1
    return freq_dict

def calculate_Z_cost(ans,shots):
    cost = 0
    #1
    cost += shots*(-0.4804)
    #Z0
    cost += (-ans.get("10",0)+ans.get("01",0)-ans.get("11",0)+ans.get("00",0))*(0.3435)
    #Z1
    cost += (ans.get("10",0)-ans.get("01",0)-ans.get("11",0)+ans.get("00",0))*(-0.4347)
    #Z0Z1
    cost += (-ans.get("10",0)-ans.get("01",0)+ans.get("11",0)+ans.get("00",0))*(0.5716)
    cost=cost/shots

    return cost

def calculate_X_cost(ans,shots):
    cost = 0
    cost += (-ans.get("10",0)-ans.get("01",0)+ans.get("11",0)+ans.get("00",0))*(0.0910)
    cost=cost/shots

    return cost


# Pre-computed Pauli operators for density-matrix based evaluation
_PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_Z_I = np.kron(_PAULI_Z, _PAULI_I)
_I_Z = np.kron(_PAULI_I, _PAULI_Z)
_Z_Z = np.kron(_PAULI_Z, _PAULI_Z)
_X_X = np.kron(_PAULI_X, _PAULI_X)
_X_I = np.kron(_PAULI_X, _PAULI_I)
_I_X = np.kron(_PAULI_I, _PAULI_X)


def _expectation_from_dm(dm, operator):
    if dm is None:
        raise ValueError("Density matrix is required to compute expectation values.")
    return float(np.real(np.trace(dm @ operator)))


def calculate_Z_cost_from_dm(dm, flip0=False, flip1=False, return_expectations=False):
    exp_z0 = _expectation_from_dm(dm, _Z_I)
    exp_z1 = _expectation_from_dm(dm, _I_Z)
    exp_z0z1 = _expectation_from_dm(dm, _Z_Z)
    if flip0:
        exp_z0 *= -1
        exp_z0z1 *= -1
    if flip1:
        exp_z1 *= -1
        exp_z0z1 *= -1

    exp_z0_cost = exp_z1  # swap only for Hamiltonian coefficients
    exp_z1_cost = exp_z0

    term_z0 = 0.3435 * exp_z0_cost
    term_z1 = -0.4347 * exp_z1_cost
    term_z0z1 = 0.5716 * exp_z0z1
    cost = -0.4804 + term_z0 + term_z1 + term_z0z1
    if return_expectations:
        return cost, {"exp_z0": exp_z0, "exp_z1": exp_z1, "exp_z0z1": exp_z0z1}
    return cost


def calculate_X_cost_from_dm(dm, flip0=False, flip1=False, return_expectations=False):
    exp_z0 = _expectation_from_dm(dm, _Z_I)
    exp_z1 = _expectation_from_dm(dm, _I_Z)
    exp_z0z1 = _expectation_from_dm(dm, _Z_Z)
    if flip0:
        exp_z0 *= -1
        exp_z0z1 *= -1
    if flip1:
        exp_z1 *= -1
        exp_z0z1 *= -1

    exp_xx = exp_z0z1
    term_xx = 0.0910 * exp_xx
    cost = term_xx
    if return_expectations:
        return cost, {"exp_xx": exp_xx}
    return cost


def _extract_density_records(results_df):
    if "server_dm" not in results_df.columns:
        raise ValueError("Density matrices were not collected in the simulation results.")
    records = []
    for _, row in results_df.iterrows():
        dm = row.get("server_dm")
        if dm is None:
            continue
        client = row.get("client_result")
        records.append((dm, client))
    if not records:
        raise ValueError("Density matrix data is empty.")
    return records


def ZZ_cost_density(
    num_runs,
    dephase_rates,
    client_fidelitys,
    distances,
    T1s,
    client_T1s,
    T2_ratios,
    sges,
    dges,
    gate_speed_factors,
    client_gate_speed_factors,
    entanglement_fidelities,
    entanglement_speed_factors,
    shots,
    angle,
    flag=0,
    base_seed=42,
    *,
    timing_mode: str = "factor",
    gate_time_ns: "Optional[float]" = None,
):
    results_df = run_experiment(
        num_runs=num_runs,
        dephase_rates=dephase_rates,
        client_fidelitys=client_fidelitys,
        distances=distances,
        T1s=T1s,
        T2_ratios=T2_ratios,
        client_T1s=client_T1s,
        sges=sges,
        dges=dges,
        gate_speed_factors=gate_speed_factors,
        client_gate_speed_factors=client_gate_speed_factors,
        entanglement_fidelities=entanglement_fidelities,
        entanglement_speed_factors=entanglement_speed_factors,
        max_runs=shots,
        angle=angle,
        flag=flag,
        collect_density=True,
        base_seed=base_seed,
        timing_mode=timing_mode,
        gate_time_ns=gate_time_ns,
    )
    density_records = _extract_density_records(results_df)
    total_cost = 0.0
    rho_ref = None
    exp_z0_total = 0.0
    exp_z1_total = 0.0
    exp_z0z1_total = 0.0
    for _, (dm, client_bits) in enumerate(density_records, start=1):
        flip0 = bool(client_bits[2]) if client_bits is not None else False
        flip1 = bool(client_bits[4]) if client_bits is not None else False
        cost_val, exps = calculate_Z_cost_from_dm(
            dm,
            flip0=flip0,
            flip1=flip1,
            return_expectations=True,
        )
        total_cost += cost_val
        exp_z0_total += exps['exp_z0']
        exp_z1_total += exps['exp_z1']
        exp_z0z1_total += exps['exp_z0z1']
        if rho_ref is None:
            rho_ref = dm
    count = len(density_records)
    cost = total_cost / count
    zz_expectations = {
        "exp_z0": exp_z0_total / count,
        "exp_z1": exp_z1_total / count,
        "exp_z0z1": exp_z0z1_total / count,
    }
    total_shot_time = results_df['shot_time'].sum()
    return cost, total_shot_time, rho_ref, zz_expectations


def XX_cost_density(
    num_runs,
    dephase_rates,
    client_fidelitys,
    distances,
    T1s,
    client_T1s,
    T2_ratios,
    sges,
    dges,
    gate_speed_factors,
    client_gate_speed_factors,
    entanglement_fidelities,
    entanglement_speed_factors,
    shots,
    angle,
    flag=1,
    base_seed=42,
    *,
    timing_mode: str = "factor",
    gate_time_ns: "Optional[float]" = None,
):
    results_df = run_experiment(
        num_runs=num_runs,
        dephase_rates=dephase_rates,
        client_fidelitys=client_fidelitys,
        distances=distances,
        T1s=T1s,
        T2_ratios=T2_ratios,
        client_T1s=client_T1s,
        sges=sges,
        dges=dges,
        gate_speed_factors=gate_speed_factors,
        client_gate_speed_factors=client_gate_speed_factors,
        entanglement_fidelities=entanglement_fidelities,
        entanglement_speed_factors=entanglement_speed_factors,
        max_runs=shots,
        angle=angle,
        flag=flag,
        collect_density=True,
        base_seed=base_seed,
        timing_mode=timing_mode,
        gate_time_ns=gate_time_ns,
    )
    density_records = _extract_density_records(results_df)
    total_cost = 0.0
    rho_ref = None
    exp_xx_total = 0.0
    for _, (dm, client_bits) in enumerate(density_records, start=1):
        if client_bits is not None:
            flip0 = (client_bits[1] + client_bits[0]) % 2 == 1
            flip1 = (client_bits[3] + client_bits[0]) % 2 == 1
        else:
            flip0 = flip1 = False
        cost_val, exps = calculate_X_cost_from_dm(
            dm,
            flip0=flip0,
            flip1=flip1,
            return_expectations=True,
        )
        total_cost += cost_val
        exp_xx_total += exps['exp_xx']
        if rho_ref is None:
            rho_ref = dm
    count = len(density_records)
    cost = total_cost / count
    xx_expectations = {"exp_xx": exp_xx_total / count}
    total_shot_time = results_df['shot_time'].sum()
    return cost, total_shot_time, rho_ref, xx_expectations


def ZZ_cost(num_runs, 
                   dephase_rates,client_fidelitys, distances, T1s, client_T1s, T2_ratios, 
                   sges, dges, gate_speed_factors, client_gate_speed_factors,
                   entanglement_fidelities, entanglement_speed_factors,shots,angle,flag=0, base_seed=42):
    # print(angle)
    # print("zz hajimari")
    results_df = run_experiment(
    num_runs=num_runs,
    dephase_rates=dephase_rates,
    client_fidelitys=client_fidelitys,
    distances=distances,
    T1s=T1s,
    T2_ratios=T2_ratios,
    client_T1s=client_T1s,
    sges=sges,
    dges=dges,
    gate_speed_factors=gate_speed_factors,
    client_gate_speed_factors=client_gate_speed_factors,
    entanglement_fidelities=entanglement_fidelities,
    entanglement_speed_factors=entanglement_speed_factors,
    max_runs=shots,
    angle=angle,
    flag=flag,
    base_seed=base_seed)
    # print("zz owari")
    serverlist = results_df['server_result'].tolist()
    result = count_tuple_frequencies(serverlist)
    x=calculate_Z_cost(result,shots)
    total_shot_time = results_df['shot_time'].sum()
    return x,total_shot_time

def XX_cost(num_runs, 
                   dephase_rates,client_fidelitys, distances, T1s, client_T1s, T2_ratios, 
                   sges, dges, gate_speed_factors, client_gate_speed_factors,
                   entanglement_fidelities, entanglement_speed_factors,shots,angle,flag=1, base_seed=42):
    results_df = run_experiment(
    num_runs=num_runs,
    dephase_rates=dephase_rates,
    client_fidelitys=client_fidelitys,
    distances=distances,
    T1s=T1s,
    T2_ratios=T2_ratios,
    client_T1s=client_T1s,
    sges=sges,
    dges=dges,
    gate_speed_factors=gate_speed_factors,
    client_gate_speed_factors=client_gate_speed_factors,
    entanglement_fidelities=entanglement_fidelities,
    entanglement_speed_factors=entanglement_speed_factors,
    max_runs=shots,
    angle=angle,
    flag=flag,
    base_seed=base_seed)
    serverlist = results_df['server_result'].tolist()
    result = count_tuple_frequencies(serverlist)
    x=calculate_X_cost(result,shots)
    total_shot_time = results_df['shot_time'].sum()
    return x,total_shot_time

# def test_cost(angle):
#     shots=100
#     num_runs = 10  # テストのため少ない回数に設定。実際の実験では増やしてください。
#     dephase_rates = [0.00]
#     # distances = list(range(100, 1001, 100))  # km
#     distances = [1000]
#     T1s = [1e160]  # ns
#     T2s = [1e150]   # ns
#     sges = [0.000]  # 単量子ゲートの誤り率
#     dges = [0.00]    # 2量子ゲートの誤り率
#     gate_speed_factors = [1.0]
#     entanglement_fidelities = [1]
#     entanglement_speed_factors = [300]
#     z,zts = ZZ_cost(
#     num_runs=num_runs,
#     dephase_rates=dephase_rates,
#     distances=distances,
#     T1s=T1s,
#     T2s=T2s,
#     sges=sges,
#     dges=dges,
#     gate_speed_factors=gate_speed_factors,
#     entanglement_fidelities=entanglement_fidelities,
#     entanglement_speed_factors=entanglement_speed_factors,
#     shots=shots,
#     angle=angle
#     )   
#     x,xts=XX_cost(num_runs=num_runs,
#     dephase_rates=dephase_rates,
#     distances=distances,
#     T1s=T1s,
#     T2s=T2s,
#     sges=sges,
#     dges=dges,
#     gate_speed_factors=gate_speed_factors,
#     entanglement_fidelities=entanglement_fidelities,
#     entanglement_speed_factors=entanglement_speed_factors,
#     shots=shots,
#     angle=angle
#     )
    # cost=z+2*x
    # angle_history.append(angle)
    # cost_history.append(cost)
    # return cost

# from scipy.optimize import minimize_scalar

# angle_history=[]
# cost_history=[]
# result = minimize_scalar(test_cost, method='bounded', bounds=(-np.pi,np.pi),tol=0.0015)

# if result.success:
#     print("エネルギー:", result.fun + 1/1.4172975)
#     print("θ:", result.x)
#     print("評価回数:", result.nfev)
#     print("反復回数:", result.nit)
# else:
#     print("最適化に失敗しました。メッセージ:", result.message)

# print(result.x)

import time
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar


def run_vqe_optimization_experiment(
    fidelity_factors,
    num_runs,
    dephase_rates,
    client_fidelitys,
    distances,
    T1s,
    T2_ratios,
    client_T1s,
    sges,
    dges,
    gate_speed_factors,
    client_gate_speed_factors,
    entanglement_fidelities,
    entanglement_speed_factors,
    shots,
    flag,
    tol=0.0015,
    bounds=(-np.pi, np.pi),
    base_seed=42,
):
    results = []
    # Apply fidelity_factors to sges and dges to create lists
    # sges = [sges[0] * factor for factor in fidelity_factors]
    # dges = [dges[0] * factor for factor in fidelity_factors]
    # パラメータの全組み合わせを生成
    
    parameter_combinations = list(itertools.product(
        dephase_rates, distances, T1s, T2_ratios, client_T1s,client_fidelitys,client_gate_speed_factors, 
        fidelity_factors, gate_speed_factors, 
        entanglement_fidelities, entanglement_speed_factors
    ))
    total_combinations = len(parameter_combinations)
    for idx, (dephase_rate, distance, T1, T2_ratio, client_T1,client_fidelity,client_gate_speed_factor,fidelity_factor, gate_factor, entanglement_fidelity, entanglement_factor) in enumerate(parameter_combinations, 1):
        sge = sges[0] * fidelity_factor
        dge = dges[0] * fidelity_factor
        print(f"\nシミュレーション {idx}/{total_combinations}:")
        print(f"dephase_rate={dephase_rate}, distance={distance} km, T1={T1} ns, T2={T1*T2_ratio} ns, "
              f"sge={sge}, dge={dge}, gate_speed_factor={gate_factor}, "
              f"entanglement_fidelity={entanglement_fidelity}, entanglement_speed_factor={entanglement_factor}")

        total_time_records = []

        def cost_func(angle):
            cost_zz, total_time_zz = ZZ_cost(
                num_runs=num_runs,
                dephase_rates=[dephase_rate],
                client_fidelitys=[client_fidelity],
                distances=[distance],
                T1s=[T1],
                T2_ratios=[T2_ratio],
                client_T1s=[client_T1],
                sges=[sge],
                dges=[dge],
                gate_speed_factors=[gate_factor],
                entanglement_fidelities=[entanglement_fidelity],
                entanglement_speed_factors=[entanglement_factor],
                client_gate_speed_factors=[client_gate_speed_factor],
                shots=shots,
                angle=angle,
                flag=0,
                base_seed=base_seed,
            )
            cost_xx, total_time_xx = XX_cost(
                num_runs=num_runs,
                dephase_rates=[dephase_rate],
                client_fidelitys=[client_fidelity],
                distances=[distance],
                T1s=[T1],
                T2_ratios=[T2_ratio],
                client_T1s=[client_T1],
                sges=[sge],
                dges=[dge],
                gate_speed_factors=[gate_factor],
                entanglement_fidelities=[entanglement_fidelity],
                entanglement_speed_factors=[entanglement_factor],
                client_gate_speed_factors=[client_gate_speed_factor],
                shots=shots,
                angle=angle,
                flag=1,
                base_seed=base_seed,
            )
            total_time = total_time_zz + total_time_xx
            total_time_records.append({total_time})
            return cost_zz + 2 * cost_xx

        start_time = time.perf_counter()
        result = minimize_scalar(cost_func, method='bounded', bounds=bounds, tol=tol)
        end_time = time.perf_counter()
        optimization_time = end_time - start_time

        final_energy = result.fun + 1 / 1.4172975
        final_angle = result.x
        nfev = result.nfev
        nit = result.nit
        total_time_sum = sum(next(iter(time_set)) for time_set in total_time_records)

        df = pd.DataFrame({
            'final_energy': [final_energy],
            'final_angle': [final_angle],
            'nfev': [nfev],
            'nit': [nit],
            'total_time': [total_time_sum],
            'fidelity_factors': [fidelity_factor],
            'num_runs': [num_runs],
            'dephase_rate': [dephase_rate],
            'client_fidelity': [client_fidelity],
            'distance': [distance],
            'T1': [T1],
            'T2_ratio': [T2_ratio],
            'client_T1': [client_T1],
            'sge': [sge],
            'dge': [dge],
            'gate_speed_factor': [gate_factor],
            'client_gate_speed_factor': [client_gate_speed_factor],
            'entanglement_fidelity': [entanglement_fidelity],
            'entanglement_speed_factor': [entanglement_factor],
            'shots': [shots],
            'flag': [flag]
        })
        results.append(df)

    results_df = pd.concat(results, ignore_index=True)
    return results_df
