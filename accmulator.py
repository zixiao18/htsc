#!/usr/bin/env python
# coding: utf-8

import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

# 参数
S0=100  
r=0.05 # interest rate
alpha=0.07 # drift
sigma=0.50 # volatility
# 本模型暂未考虑dividend
T=0.5  #时间（年）
I=30000  #模拟的次数
M=126 #每月21个交易日, 半年共126交易日
# 暂不考虑节假日等
MT = 252 #每年252交易日
rate = 2 # 累计倍数
terminated = np.zeros(I) # 0-未发生障碍事件；1-发生障碍事件

def stock_simulator(S0, r, alpha, sigma, T, I, M):
	# 本函数用于生成布朗运动模型下的股票数据
	# 返回值S为矩阵，记录各次模拟下各观测时间点股票的价格
	# 每一列分别对应一次模拟. 每一列的各数字分别对应在0，dt, 2dt, ..., T 时间下的价格
	S=np.zeros((M+1,I)) 
	S[0]=S0 # 初始化S，并设置初始金额S0
	dt = T/M 
	# 此模拟中，将每一交易日 计为 1/252 年，暂不考虑节假日等； 
	npr.seed(0)
	for t in range(1,M+1):
		S[t] = S[t-1]*np.exp((alpha-sigma**2/2)*dt+sigma*np.sqrt(dt)*npr.standard_normal(I))
		# 通过公式得出
	return S 


def observe_windows(S):
	#该函数输入生成的股票价格S，输出各次模拟的payoff
	up_barrier = S[0] * 1.05 # 设置障碍，为初始价格的105%
	strike = S[0] * 0.90 # 设置实行价格，为初始价格的90%
	stop_time = np.array([M-1 for _ in range(I)]) #初始化达到障碍的时间，默认为 M-1
	shares_total = np.zeros(I) #
	below_barrier_count = np.zeros(I)
	npr.seed(0)
	for t in range(M): 
		# 本循环统计各次试验：股票价格在行使价格以上/以下的天数；及达到障碍的时间
		S_new = S[t]
		below_barrier_count += (S_new < strike) 
		for j in range(I):
			if terminated[j] == 0 and S_new[j] > up_barrier[j]:
				stop_time[j], terminated[j] = t, 1
	shares_total = (below_barrier_count * (rate - 1) + stop_time) * 100
	#计算总共购买的股数；行使价格以上时购买100股，行使价格以下时购买200股
	final_price = S[stop_time,np.arange(0,I)] # 最终价格为停止时间时的价格
	discount = np.exp(-stop_time/MT*r) # 根据停止时间计算折扣比例
	payoff = np.round(discount * shares_total * (final_price - strike),2) # 折扣后收益（DPO）
	pay_term = np.sum((terminated * payoff)) / np.sum(terminated) # 达到障碍条件的折扣收益
	pay_no_term = np.sum(((1-terminated)*payoff)) / (I - np.sum(terminated)) # 未达到障碍条件的折扣收益
	return payoff, terminated, pay_no_term, pay_term, stop_time

def report(payoff, terminated, pay_no_term, pay_term, tictoc, stop_time):
	# 通过生成的数据返回可读的结果
	fair = np.mean(payoff)
	val_MC = f"经过{I}次蒙特卡洛试验，由各次试验的平均收益推算出该accumulator的合理价格为{np.mean(fair):0.2f}元 \n"
	std = np.std(payoff)/np.sqrt(I)
	lower_bound = fair - 1.96 * std
	upper_bound = fair + 1.96 * std
	error = f"根据中心极限定律，该合理价格的95%置信区间为({lower_bound:.2f},{upper_bound:.2f})元 \n"
	terminated_rate = f"在此{I}次试验中, 有{np.sum(terminated):0.0f}次({np.mean(terminated)*100:0.2f})%\
因达到障碍条件而提前终止\n\
当发生障碍条件时每次平均收益为{pay_no_term:.2f}元; \n\
未发生障碍条件，每次平均收益为{pay_term:.2f}元\n"
	stop_avg = f"发生障碍条件时，平均{np.mean(stop_time):0.0f}个交易日达到障碍条件\n"
	time_elapsed = f"计算总用时:{tictoc:.3f}秒\n"
	to_report = val_MC + error + terminated_rate + stop_avg + time_elapsed
	return to_report


if __name__ == "__main__":
	tic = time.perf_counter() #计时开始
	S = stock_simulator(S0, r, alpha, sigma, T, I, M) #模拟股票价格
	payoff, terminated, pay_no_term, pay_term, stop_time = observe_windows(S) #统计
	toc = time.perf_counter() #计时结束
	tictoc = toc - tic
	to_report = report(payoff, terminated, pay_no_term, pay_term, tictoc, stop_time) #生成报告
	print(to_report)

	





