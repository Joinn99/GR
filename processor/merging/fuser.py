import torch
import torch.nn as nn

from .task_vector import TaskVector


class TemporalModelFuser:
    def __init__(self, base_model):
        """
        初始化Fuser
        :param base_model: 预训练的Base LLM，用于计算任务向量。
        """
        self.base_model = base_model

    def _get_task_vector(self, model):
        """辅助函数，计算单个模型的任务向量"""
        return TaskVector(pretrained_model=self.base_model, finetuned_model=model)

    def _get_diff_vector(self, model1, model2):
        """辅助函数，计算两个模型之间的差分向量 (M2 - M1)"""
        return TaskVector(pretrained_model=model1, finetuned_model=model2)

    # 策略 0: 您当前的实现（为了对比）
    def extrapolate_linear_velocity(self, checkpoints, scaling_coefficient=1.0):
        """
        使用最后两个checkpoint进行线性外推（一阶速度）。
        M_pred = M_t + alpha * (M_t - M_{t-1})
        """
        if len(checkpoints) < 2:
            raise ValueError("Linear extrapolation requires at least 2 checkpoints.")
        
        phase_t_minus_1_model = checkpoints[-2]
        phase_t_model = checkpoints[-1]
        
        diff_vector = self._get_diff_vector(phase_t_minus_1_model, phase_t_model)
        merged_model = diff_vector.combine_with_pretrained_model(
            pretrained_model=phase_t_model, 
            scaling_coefficient=scaling_coefficient
        )
        return merged_model

    # 策略 1: 基于移动平均速度的外推 (更平滑的速度)
    def extrapolate_moving_average_velocity(self, checkpoints, scaling_coefficient=1.0):
        """
        计算历史迁移向量的平均值，并将其应用到最新的checkpoint上，以获得更平滑的预测。
        V_avg = mean(M_i - M_{i-1} for i=1..t)
        M_pred = M_t + alpha * V_avg
        """
        if len(checkpoints) < 2:
            raise ValueError("Moving average velocity requires at least 2 checkpoints.")
        
        diff_vectors = []
        for i in range(len(checkpoints) - 1):
            vec = self._get_diff_vector(checkpoints[i], checkpoints[i+1])
            diff_vectors.append(vec)
            
        # 计算平均迁移向量
        avg_diff_vector = diff_vectors[0]
        for i in range(1, len(diff_vectors)):
            avg_diff_vector += diff_vectors[i]
        avg_diff_vector *= (1.0 / len(diff_vectors))
        
        # 应用到最新的模型上
        latest_model = checkpoints[-1]
        merged_model = avg_diff_vector.combine_with_pretrained_model(
            pretrained_model=latest_model,
            scaling_coefficient=scaling_coefficient
        )
        return merged_model

    # 策略 2: 基于加速度的外推 (捕捉趋势变化)
    def extrapolate_with_acceleration(self, checkpoints, velocity_scale=1.0, acceleration_scale=0.5):
        """
        引入二阶差分（加速度）来预测更复杂的非线性趋势。
        V_t = M_t - M_{t-1}
        A_t = V_t - V_{t-1} = (M_t - M_{t-1}) - (M_{t-1} - M_{t-2})
        M_pred = M_t + alpha * V_t + beta * A_t
        """
        if len(checkpoints) < 3:
            raise ValueError("Acceleration extrapolation requires at least 3 checkpoints.")
            
        m_t_minus_2, m_t_minus_1, m_t = checkpoints[-3:]
        
        # 计算最近的速度 V_t
        velocity_vector = self._get_diff_vector(m_t_minus_1, m_t)
        
        # 计算加速度 A_t
        prev_velocity_vector = self._get_diff_vector(m_t_minus_2, m_t_minus_1)
        acceleration_vector = velocity_vector + (prev_velocity_vector * -1.0)

        # 组合应用
        # M_pred = M_t + alpha * V_t
        intermediate_model = velocity_vector.combine_with_pretrained_model(
            pretrained_model=m_t,
            scaling_coefficient=velocity_scale
        )
        # M_pred = (M_t + alpha*V_t) + beta * A_t
        final_model = acceleration_vector.combine_with_pretrained_model(
            pretrained_model=intermediate_model,
            scaling_coefficient=acceleration_scale
        )
        return final_model
    
    # 策略 3: 指数衰减加权融合 (平滑历史信息)
    def fuse_exponential_decay(self, checkpoints, decay_rate=0.8):
        """
        对所有历史模型的任务向量进行加权平均融合，权重随时间指数衰减。
        这是一种融合方法，而非外推预测。它更保守，但通常非常稳健。
        V_merged = sum(w_i * V_i) / sum(w_i)  where w_i = decay_rate^(t-i)
        M_merged = M_base + V_merged
        """
        if not checkpoints:
            raise ValueError("Checkpoints list cannot be empty.")
        
        num_checkpoints = len(checkpoints)
        task_vectors = [self._get_task_vector(c) for c in checkpoints]
        
        # 计算权重
        weights = [decay_rate ** (num_checkpoints - 1 - i) for i in range(num_checkpoints)]
        
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 加权求和
        weighted_sum_vector = task_vectors[0] * normalized_weights[0]
        for i in range(1, num_checkpoints):
            weighted_sum_vector += (task_vectors[i] * normalized_weights[i])
            
        # 应用于Base Model
        merged_model = weighted_sum_vector.combine_with_pretrained_model(self.base_model)
        return merged_model