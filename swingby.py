#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import *
import numpy as np

G = 6.672 * pow(10, -11)  # 万有引力定数
MS = 1.989 * pow(10, 30)  # 太陽の質量[kg]
MJ = 1.898 * pow(10, 27)  # 木星の質量[kg]
ME = 5.972 * pow(10, 24)  # 地球の質量[kg]
MU = 10  # 探査機の質量[kg]

# RS = 695700 * pow(10, 3)  # 太陽の半径[m]
RE = 1.496 * pow(10, 11)  # 地球と太陽の距離[m]
RJ = RE * 5.20260  # 木星と太陽の距離[m]
T = 365 * 24 * 60 * 60  # 1年間を秒に変換した時間 [s]
AS = G * (T ** 2) * MS / (RE ** 3)  # 無次元化パラメータ(太陽)
AJ = G * (T ** 2) * MJ / (RE ** 3)  # 無次元化パラメータ(木星)
AE = G * (T ** 2) * ME / (RE ** 3)  # 無次元化パラメータ(地球)
AU = G * (T ** 2) * MU / (RE ** 3)  # 無次元化パラメータ(探査機)

VE = 29.78 * pow(10, 3)  # 地球の公転速度[m/s]
VJ = 47016 / 60 / 60 * 1000  # 木星の公転速度[m/s]
VU_INIT = 39.9 * 1000  # 探査機の地球から脱出する時の速度 [m/s]

ECL_LONG_1 = -18.64 / 360 * (2 * pi)  # 地球から探査機が脱出するときの地球の黄経[rad]
ECL_LONG_2 = 127.13 / 360 * (2 * pi)  # 木星に探査機が到着するときの木星の黄経[rad]

ECL_J_OFFSET = 43 / 360 * (2 * pi)  # 地球から木星に到着するまでに木星が移動する黄経のずれ[rad]


def get_due_dt(all_list):
    return AS * (0 - all_list[0]) / \
           ((0 - all_list[0]) ** 2 + (0 - all_list[1]) ** 2 + (0 - all_list[2]) ** 2) ** (3 / 2) \
           + AJ * (all_list[6] - all_list[0]) / ((all_list[6] - all_list[0]) ** 2 +
                                                 (all_list[7] - all_list[1]) ** 2 + (
                                                     all_list[8] - all_list[2]) ** 2) ** (3 / 2)


def get_dve_dt(all_list):
    return AS * (0 - all_list[1]) / \
           ((0 - all_list[0]) ** 2 + (0 - all_list[1]) ** 2 + (0 - all_list[2]) ** 2) ** (3 / 2) \
           + AJ * (all_list[7] - all_list[1]) / ((all_list[6] - all_list[0]) ** 2 + (all_list[7] - all_list[1])
                                                 ** 2 + (all_list[8] - all_list[2]) ** 2) ** (3 / 2)


def get_dwe_dt(all_list):
    return AS * (0 - all_list[2]) / \
           ((0 - all_list[0]) ** 2
            + (0 - all_list[1]) ** 2 + (0 - all_list[2]) ** 2) ** (3 / 2) \
           + AJ * (all_list[8] - all_list[2]) / ((all_list[6] - all_list[0]) ** 2 + (all_list[7] - all_list[1])
                                                 ** 2 + (all_list[8] - all_list[2]) ** 2) ** (3 / 2)


def get_duj_dt(all_list):
    return AS * (0 - all_list[6]) / ((0 - all_list[6]) ** 2 + (0 - all_list[7]) ** 2 + (0 - all_list[8]) ** 2) ** (
        3 / 2) \
           + AE * (all_list[0] - all_list[6]) / ((all_list[0] - all_list[6]) ** 2 + (all_list[1] - all_list[7])
                                                 ** 2 + (all_list[2] - all_list[8]) ** 2) ** (3 / 2)


def get_dvj_dt(all_list):
    return AS * (0 - all_list[7]) / ((0 - all_list[6]) ** 2 + (0 - all_list[7]) ** 2 + (0 - all_list[8]) ** 2) ** (
        3 / 2) \
           + AE * (all_list[1] - all_list[7]) / ((all_list[0] - all_list[6]) ** 2 + (all_list[1] - all_list[7])
                                                 ** 2 + (all_list[2] - all_list[8]) ** 2) ** (3 / 2)


def get_dwj_dt(all_list):
    return AS * (0 - all_list[8]) / ((0 - all_list[6]) ** 2 + (0 - all_list[7]) ** 2 + (
        0 - all_list[8]) ** 2) ** (3 / 2) \
           + AE * (all_list[2] - all_list[8]) / ((all_list[0] - all_list[6]) ** 2 + (all_list[1] - all_list[7])
                                                 ** 2 + (all_list[2] - all_list[8]) ** 2) ** (3 / 2)


def get_duu_dt(all_list):
    return AS * (0 - all_list[12]) / ((0 - all_list[12]) ** 2 + (0 - all_list[13]) ** 2 + (0 - all_list[14]) ** 2) ** (
        3 / 2) \
           + AE * (all_list[0] - all_list[12]) / ((all_list[0] - all_list[12]) ** 2 + (all_list[1] - all_list[13])
                                                  ** 2 + (all_list[2] - all_list[14]) ** 2) ** (3 / 2) + AJ * (
        all_list[6] - all_list[12]) / ((all_list[6] - all_list[12]) ** 2 + (all_list[7] - all_list[13])
                                       ** 2 + (all_list[8] - all_list[14]) ** 2) ** (3 / 2)


def get_dvu_dt(all_list):
    return AS * (0 - all_list[13]) / ((0 - all_list[12]) ** 2 + (0 - all_list[13]) ** 2 + (0 - all_list[14]) ** 2) ** (
        3 / 2) \
           + AE * (all_list[1] - all_list[13]) / ((all_list[0] - all_list[12]) ** 2 + (all_list[1] - all_list[13])
                                                  ** 2 + (all_list[2] - all_list[14]) ** 2) ** (3 / 2) + AJ * (
        all_list[7] - all_list[13]) / ((all_list[6] - all_list[12]) ** 2 + (all_list[7] - all_list[13])
                                       ** 2 + (all_list[8] - all_list[14]) ** 2) ** (3 / 2)


def get_dwu_dt(all_list):
    return AS * (0 - all_list[14]) / ((0 - all_list[12]) ** 2 + (0 - all_list[13]) ** 2 + (0 - all_list[14]) ** 2) ** (
        3 / 2) \
           + AE * (all_list[2] - all_list[14]) / ((all_list[0] - all_list[12]) ** 2 + (all_list[1] - all_list[13])
                                                  ** 2 + (all_list[2] - all_list[14]) ** 2) ** (3 / 2) + AJ * (
        all_list[2] - all_list[14]) / ((all_list[6] - all_list[12]) ** 2 + (all_list[7] - all_list[13])
                                       ** 2 + (all_list[8] - all_list[14]) ** 2) ** (3 / 2)


def make_derivative(all_list):
    return np.array([all_list[3], all_list[4], all_list[5],
                     get_due_dt(all_list), get_dve_dt(all_list), get_dwe_dt(all_list),
                     all_list[9], all_list[10], all_list[11],
                     get_duj_dt(all_list), get_dvj_dt(all_list), get_dwj_dt(all_list),
                     all_list[15], all_list[16], all_list[17],
                     get_duu_dt(all_list), get_dvu_dt(all_list), get_dwu_dt(all_list)])


def cal_and_plot(tend):
    dlt = 0.001
    time_list = np.arange(0.0, tend, dlt)
    lis_num = time_list.shape[0]
    # [xe, ye, ze, ue, ve, we, xj, yj, zj, uj, vj, wj, xu, yu, zu, uu, vu, wu]のn*18配列を用意
    all_list = np.zeros((lis_num, 18))
    # 初期条件を代入
    all_list[0, :] = [RE / RE * cos(ECL_LONG_1), RE / RE * sin(ECL_LONG_1), 0,
                      -VE / RE * T * sin(ECL_LONG_1), VE / RE * T * cos(ECL_LONG_1), 0,
                      RJ / RE * cos(ECL_LONG_2 - ECL_J_OFFSET), RJ / RE * sin(ECL_LONG_2 - ECL_J_OFFSET), 0,
                      -VJ / RE * T * sin(ECL_LONG_2 - ECL_J_OFFSET),
                      VE / RJ * T * cos(ECL_LONG_2 - ECL_J_OFFSET), 0,
                      RE * 1.0007 / RE * cos(ECL_LONG_1), RE * 1.0007 / RE * sin(ECL_LONG_1), 0,
                      -VU_INIT / RE * T * sin(ECL_LONG_1), VU_INIT / RE * T * cos(ECL_LONG_1), 0]

    """RK4による計算"""
    for n in range(lis_num - 1):
        k1 = make_derivative(all_list[n])
        k2 = make_derivative(all_list[n] + dlt / 2 * k1)
        k3 = make_derivative(all_list[n] + dlt / 2 * k2)
        k4 = make_derivative(all_list[n] + dlt * k3)
        all_list[n + 1, :] = all_list[n, :] + dlt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        v_u = sqrt(all_list[n][12] ** 2 + all_list[n][13] ** 2 + all_list[n][14] ** 2) * RE / T
        print("division: {0}/{1}, V_U:{2}[km/s]".format(n, lis_num, v_u))

    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect=1)  # 軌道のxy平面投影グラフの作成
    sun = patches.Circle(xy=(0, 0), radius=0.3, fc='orange', label="sun")
    ax1.add_patch(sun)
    ax1.plot(all_list[:, 0], all_list[:, 1], label="earth", color="green")
    ax1.plot(all_list[:, 6], all_list[:, 7], label="venus", color="black")
    ax1.plot(all_list[:, 12], all_list[:, 13], label="satellite", color="blue")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    #ax1.set_xlim([-5.5, 5.5])
    #ax1.set_ylim([-5.5, 5.5])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    cal_and_plot(546 / 365*10)
