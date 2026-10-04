from xml.etree import ElementTree

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

NAVY = "#163B65"
BLUE = "#4C78A8"
STEEL = "#71879C"
SLATE = "#475569"
MIST = "#F4F7FA"
BORDER = "#D7E0E8"
WHITE = "#FFFFFF"
ANNUAL_RISK_BENEFIT_BASE = 40.0
ROI_REFERENCE = 0.20

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.titleweight': 'bold',
    'axes.edgecolor': BORDER,
    'axes.labelcolor': SLATE,
    'grid.color': BORDER,
})

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="罐区安全-绿色协同改造决策支持系统", layout="wide")

MEASURE_LIBRARY = (
    {'name': '密封改造', 'risk': 0.20, 'recovery': 0.10, 'emission': 0.10, 'energy': 0.05, 'investment': 50},
    {'name': '泄漏监测', 'risk': 0.30, 'recovery': 0.20, 'emission': 0.05, 'energy': 0.05, 'investment': 80},
    {'name': '应急切断', 'risk': 0.25, 'recovery': 0.30, 'emission': 0.05, 'energy': 0.05, 'investment': 60},
    {'name': '油气回收', 'risk': 0.15, 'recovery': 0.05, 'emission': 0.40, 'energy': 0.10, 'investment': 100},
    {'name': '泵组节能', 'risk': 0.10, 'recovery': 0.05, 'emission': 0.10, 'energy': 0.30, 'investment': 120},
)
MAX_PLAN_INVESTMENT = sum(measure['investment'] for measure in MEASURE_LIBRARY)
MAX_RISK_REDUCTION = 1 - np.prod([1 - measure['risk'] for measure in MEASURE_LIBRARY])
MAX_RECOVERY_REDUCTION = 1 - np.prod([1 - measure['recovery'] for measure in MEASURE_LIBRARY])
MAX_EMISSION_REDUCTION = 1 - np.prod([1 - measure['emission'] for measure in MEASURE_LIBRARY])
MAX_ENERGY_REDUCTION = 1 - np.prod([1 - measure['energy'] for measure in MEASURE_LIBRARY])
MAX_ANNUAL_RISK_BENEFIT = MAX_RISK_REDUCTION * ANNUAL_RISK_BENEFIT_BASE
PAYBACK_REFERENCE_YEARS = 20.0
SCORING_SCHEMA_VERSION = "safety-economy-environment-v1"

MEASURE_SPATIAL_INFO = {
    '密封改造': {'location': '事故储罐顶部密封部位', 'mechanism': '降低挥发与泄漏源强', 'color': '#B8860B', 'symbol': 'circle'},
    '泄漏监测': {'location': '事故储罐周边监测点', 'mechanism': '提升早期识别与报警能力', 'color': '#2E6F69', 'symbol': 'diamond'},
    '应急切断': {'location': '事故储罐进出料切断点', 'mechanism': '缩短隔离时间并限制事故升级', 'color': '#E67E22', 'symbol': 'square'},
    '油气回收': {'location': '装车台油气回收接口', 'mechanism': '降低装卸挥发排放', 'color': '#367C6A', 'symbol': 'circle-open'},
    '泵组节能': {'location': '动力辅房泵组区域', 'mechanism': '降低能耗并提升设备运行稳定性', 'color': '#7A5195', 'symbol': 'x'},
}

SCENARIO_LIBRARY = {
    '小泄漏': {'before': (5.8, 4.8, 5.0), 'impact_factor': 1.0, 'recovery_days': 3.0, 'loss': 0.22, 'category': '常规情景', 'impact_scale': 0.35, 'incident_tank': 'V-01', 'dispersion_length': 30},
    '中泄漏': {'before': (4.3, 3.7, 3.8), 'impact_factor': 1.0, 'recovery_days': 9.0, 'loss': 0.46, 'category': '常规情景', 'impact_scale': 0.65, 'incident_tank': 'V-03', 'dispersion_length': 80},
    '大泄漏': {'before': (2.8, 2.6, 2.4), 'impact_factor': 1.0, 'recovery_days': 18.0, 'loss': 0.68, 'category': '常规情景', 'impact_scale': 1.00, 'incident_tank': 'V-04', 'dispersion_length': 150},
    '蓄意冲击': {'before': (2.8, 2.6, 2.4), 'impact_factor': 1.5, 'recovery_days': 23.4, 'loss': 0.78, 'category': '非常规情景', 'impact_scale': 1.20, 'incident_tank': 'V-04', 'dispersion_length': 200},
    '随机冲击': {'before': (4.3, 3.7, 3.8), 'impact_factor': 1.2, 'recovery_days': 13.5, 'loss': 0.58, 'category': '非常规情景', 'impact_scale': 0.85, 'incident_tank': 'V-03', 'dispersion_length': 120, 'visual_mode': 'random_impact'},
}

MODEL_TANKS = (
    {'code': 'V-01', 'medium': '柴油', 'capacity': '约1-2万m³', 'diameter': '约30m', 'height_label': '约20m', 'radius': 16, 'height': 21, 'x': 30, 'y': 60, 'color': '#2C3E50', 'edge_color': '#4C78A8'},
    {'code': 'V-02', 'medium': '柴油', 'capacity': '约2-3万m³', 'diameter': '约40m', 'height_label': '约20m', 'radius': 23, 'height': 21, 'x': 80, 'y': 60, 'color': '#2C3E50', 'edge_color': '#4C78A8'},
    {'code': 'V-03', 'medium': '汽油', 'capacity': '约2-3万m³', 'diameter': '约40m', 'height_label': '约20m', 'radius': 23, 'height': 21, 'x': 130, 'y': 60, 'color': '#2C3E50', 'edge_color': '#B64545'},
    {'code': 'V-04', 'medium': '汽油', 'capacity': '约2-3万m³', 'diameter': '约40m', 'height_label': '约20m', 'radius': 23, 'height': 21, 'x': 180, 'y': 60, 'color': '#2C3E50', 'edge_color': '#B64545'},
    {'code': 'V-05', 'medium': '含油污水', 'capacity': '约2-3万m³', 'diameter': '约40m', 'height_label': '约20m', 'radius': 23, 'height': 21, 'x': 80, 'y': 120, 'color': '#2C3E50', 'edge_color': '#3E8B71'},
    {'code': 'V-06', 'medium': '含油污水', 'capacity': '约2-3万m³', 'diameter': '约40m', 'height_label': '约20m', 'radius': 23, 'height': 21, 'x': 130, 'y': 120, 'color': '#2C3E50', 'edge_color': '#3E8B71'},
)

WEATHER_DIRECTIONS = ('N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW')


@st.cache_data(show_spinner=False, max_entries=64)
def calculate_qra(selected_tank_storage_wan_tons, alpha, beta, critical_quantity_tons,
                  consequence_factor):
    q_tons = selected_tank_storage_wan_tons * 10000
    risk_value = alpha * beta * (q_tons / critical_quantity_tons)
    storage_range = np.linspace(max(0.1, selected_tank_storage_wan_tons * 0.25),
                                max(10.0, selected_tank_storage_wan_tons * 2.0), 120)
    risk_range = alpha * beta * ((storage_range * 10000) / critical_quantity_tons)
    scale = np.sqrt(selected_tank_storage_wan_tons / 4.3) * consequence_factor
    base_radii = np.array([
        [200, 120, 30, 60, 30, 10],
        [240, 140, 40, 70, 40, 10],
        [330, 200, 50, 100, 50, 10],
    ])
    if risk_value >= 100:
        level = '一级重大危险源'
    elif risk_value >= 50:
        level = '二级重大危险源'
    elif risk_value >= 10:
        level = '三级重大危险源'
    else:
        level = '一般风险源'
    return {
        'risk_value': risk_value,
        'risk_level': level,
        'storage_range': storage_range,
        'risk_range': risk_range,
        'modes': ('管道完全破裂', '管道大孔', '管道中孔', '阀门大孔', '阀门中孔', '阀门小孔'),
        'radii': base_radii * scale,
    }


@st.cache_data(show_spinner=False)
def calculate_weather_conditions():
    low_speed = np.array([2, 2, 2, 2, 3, 5, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2])
    moderate_speed = np.array([3, 5, 6, 4, 4, 7, 8, 5, 3, 2, 2, 2, 2, 2, 3, 4])
    strong_speed = np.array([4, 9, 8, 4, 2, 3, 4, 2, 1, 1, 1, 1, 1, 1, 2, 4])
    return {
        'directions': WEATHER_DIRECTIONS,
        'speed_groups': (low_speed, moderate_speed, strong_speed),
        'stability': {'C 类': 20, 'D 类': 50, 'E 类': 20, 'F 类': 10},
        'landform': '某港口配套区周边以滨水平坦地貌为主，北侧为开阔水域。',
        'mean_wind_speed': '约3.4 m/s',
        'mean_temperature': '约15.8℃',
    }


def calculate_entropy_details(values):
    matrix = np.asarray(values, dtype=float)
    proportions = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1e-12)
    indicator_count = matrix.shape[0]
    entropy_constant = 1 / np.log(indicator_count)
    entropy = -entropy_constant * np.sum(proportions * np.log(np.maximum(proportions, 1e-12)), axis=0)
    divergence = 1 - entropy
    if divergence.sum() <= 1e-12:
        weights = np.full(matrix.shape[1], 1 / matrix.shape[1])
    else:
        weights = divergence / divergence.sum()
    return {
        'sample_count': indicator_count,
        'proportions': proportions,
        'entropy': entropy,
        'divergence': divergence,
        'weights': weights,
    }


@st.cache_data(show_spinner=False, max_entries=128)
def calculate_entropy_weights(values):
    return calculate_entropy_details(values)['weights']


def calculate_plan_effect(measure_names):
    selected = [measure for measure in MEASURE_LIBRARY if measure['name'] in measure_names]
    reductions = {}
    for key in ('risk', 'recovery', 'emission', 'energy'):
        residual = np.prod([1 - measure[key] for measure in selected]) if selected else 1.0
        reductions[key] = 1 - residual
    investment = sum(measure['investment'] for measure in selected)
    plan_name = ' + '.join(measure_names) if measure_names else '不实施改造'
    return {
        'measures': tuple(measure_names),
        'plan_name': plan_name,
        'risk_reduction': reductions['risk'],
        'recovery_reduction': reductions['recovery'],
        'emission_reduction': reductions['emission'],
        'energy_reduction': reductions['energy'],
        'investment': investment,
    }


def calculate_economic_metrics(effect):
    annual_risk_benefit = effect['risk_reduction'] * ANNUAL_RISK_BENEFIT_BASE
    investment = float(effect['investment'])
    roi = annual_risk_benefit / investment if investment > 0 else 0.0
    payback_years = investment / annual_risk_benefit if annual_risk_benefit > 0 else None
    return {
        'investment': investment,
        'annual_risk_benefit': annual_risk_benefit,
        'roi': roi,
        'payback_years': payback_years,
    }


def calculate_plan_scores(effect):
    economics = calculate_economic_metrics(effect)
    risk_score = 100 * effect['risk_reduction'] / MAX_RISK_REDUCTION
    recovery_score = 100 * effect['recovery_reduction'] / MAX_RECOVERY_REDUCTION
    investment_score = 100 * max(0.0, 1 - economics['investment'] / MAX_PLAN_INVESTMENT)
    benefit_score = 100 * min(economics['annual_risk_benefit'] / MAX_ANNUAL_RISK_BENEFIT, 1.0)
    payback_score = (
        100 * max(0.0, 1 - economics['payback_years'] / PAYBACK_REFERENCE_YEARS)
        if economics['payback_years'] is not None else 0.0
    )
    energy_index = 1 - effect['energy_reduction']
    emission_index = 1 - effect['emission_reduction']
    energy_score = 100 * (1 - energy_index) / MAX_ENERGY_REDUCTION
    emission_score = 100 * (1 - emission_index) / MAX_EMISSION_REDUCTION
    safety_score = (0.20 * risk_score + 0.20 * recovery_score) / 0.40
    economic_score = (0.10 * investment_score + 0.10 * benefit_score + 0.10 * payback_score) / 0.30
    environmental_score = (0.15 * energy_score + 0.15 * emission_score) / 0.30
    return {
        'safety_score': safety_score,
        'economic_score': economic_score,
        'environmental_score': environmental_score,
        'score': 0.40 * safety_score + 0.30 * economic_score + 0.30 * environmental_score,
    }


def select_greedy_plan(budget_wan, strategy):
    if strategy == '按风险高低依次改造':
        ordered_measures = sorted(MEASURE_LIBRARY, key=lambda item: (-item['risk'], item['investment']))
    else:
        ordered_measures = sorted(MEASURE_LIBRARY, key=lambda item: (item['investment'], -item['risk']))
    selected_measures = []
    invested = 0
    for measure in ordered_measures:
        if invested + measure['investment'] <= budget_wan:
            selected_measures.append(measure['name'])
            invested += measure['investment']
    effect = calculate_plan_effect(selected_measures)
    effect.update(calculate_plan_scores(effect))
    return effect


@st.cache_data(show_spinner=False, max_entries=64)
def enumerate_plan_options(budget_wan, score_schema_version):
    combinations_result = []
    measure_names = [measure['name'] for measure in MEASURE_LIBRARY]
    for mask in range(1 << len(measure_names)):
        measure_combo = tuple(
            name for index, name in enumerate(measure_names) if mask & (1 << index)
        )
        effect = calculate_plan_effect(measure_combo)
        effect['within_budget'] = effect['investment'] <= budget_wan
        effect.update(calculate_plan_scores(effect))
        combinations_result.append(effect)
    combinations_result.sort(key=lambda item: item['score'], reverse=True)
    return combinations_result


@st.cache_data(show_spinner=False, max_entries=64)
def enumerate_feasible_plans(budget_wan, score_schema_version):
    return [
        plan for plan in enumerate_plan_options(budget_wan, score_schema_version)
        if plan['within_budget']
    ]


@st.cache_data(show_spinner=False)
def calculate_strategy_budget_sensitivity(score_schema_version):
    rows = []
    for budget_wan in range(50, 501, 10):
        budget_plans = enumerate_feasible_plans(float(budget_wan), score_schema_version)
        comprehensive_plan = max(budget_plans, key=lambda plan: plan['score'])
        strategy_candidates = (
            ('按风险高低依次改造', select_greedy_plan(budget_wan, '按风险高低依次改造')),
            ('优先低成本', select_greedy_plan(budget_wan, '优先低成本')),
            ('综合优选', comprehensive_plan),
        )
        for strategy_name, strategy_plan in strategy_candidates:
            rows.append({
                '预算（万元）': budget_wan,
                '策略': strategy_name,
                '三维综合得分': strategy_plan['score'],
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=64)
def calculate_resilience(recommended_measures):
    effect = calculate_plan_effect(recommended_measures)
    before_matrix = np.array([
        np.asarray(scenario['before'], dtype=float) / scenario.get('impact_factor', 1.0)
        for scenario in SCENARIO_LIBRARY.values()
    ])
    improvement_basis = (0.45 * effect['risk_reduction'] + 0.35 * effect['recovery_reduction']
                         + 0.12 * effect['emission_reduction'] + 0.08 * effect['energy_reduction'])
    gain_vector = np.array([1.15, 0.95, 1.30]) * (4.8 * improvement_basis)
    after_matrix = np.minimum(9.5, before_matrix + gain_vector)
    combined = np.vstack([before_matrix, after_matrix])
    weights = calculate_entropy_weights(tuple(map(tuple, combined)))
    before_scores = before_matrix @ weights
    after_scores = after_matrix @ weights
    recovery_before = np.array([scenario['recovery_days'] for scenario in SCENARIO_LIBRARY.values()])
    recovery_after = recovery_before * (1 - effect['recovery_reduction'])
    return {
        'dimensions': ('吸收能力', '适应能力', '恢复能力'),
        'scenarios': tuple(SCENARIO_LIBRARY.keys()),
        'before_matrix': before_matrix,
        'after_matrix': after_matrix,
        'weights': weights,
        'before_scores': before_scores,
        'after_scores': after_scores,
        'recovery_before': recovery_before,
        'recovery_after': recovery_after,
        'effect': effect,
    }


@st.cache_data(show_spinner=False, max_entries=128)
def calculate_recovery_curve(initial_before, initial_after, days_before, days_after):
    before_duration = max(days_before, 0.1)
    after_duration = max(days_after, 0.1)
    horizon = max(before_duration, after_duration) * 1.35 + 1
    sample_days = np.unique(np.concatenate((
        np.linspace(0, horizon, 160),
        [0.2 * before_duration, 0.2 * after_duration, before_duration, after_duration],
    )))
    days = np.concatenate(([0.0], sample_days))

    def build_curve(loss, duration):
        recovery_start = 0.2 * duration
        progress = np.clip((days - recovery_start) / (duration - recovery_start), 0, 1)
        recovery_fraction = progress ** 2 * (3 - 2 * progress)
        ability = 100 * (1 - loss + loss * recovery_fraction)
        ability[days >= duration] = 100.0
        ability[0] = 100.0
        return ability

    before_curve = build_curve(initial_before, before_duration)
    after_curve = build_curve(initial_after, after_duration)
    return days, before_curve, after_curve


def add_box(fig, center_x, center_y, width, depth, height, color, label, opacity=1.0, base_z=0.0):
    x_min, x_max = center_x - width / 2, center_x + width / 2
    y_min, y_max = center_y - depth / 2, center_y + depth / 2
    vertices = np.array([
        [x_min, y_min, base_z], [x_max, y_min, base_z], [x_max, y_max, base_z], [x_min, y_max, base_z],
        [x_min, y_min, base_z + height], [x_max, y_min, base_z + height],
        [x_max, y_max, base_z + height], [x_min, y_max, base_z + height],
    ])
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
        j=[1, 2, 4, 2, 5, 3, 6, 0, 5, 7, 6, 7],
        k=[2, 3, 5, 6, 6, 0, 1, 4, 6, 4, 7, 4],
        color=color, opacity=opacity, flatshading=True, name=label,
        lighting=dict(ambient=0.65, diffuse=0.8, specular=0.25, roughness=0.75, fresnel=0.1),
        hovertemplate=f'{label}<extra></extra>', showlegend=False,
    ))


def add_ground_surface(fig):
    x_values = np.linspace(-130, 380, 18)
    y_values = np.linspace(-90, 430, 18)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    gradient = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())
    fig.add_trace(go.Surface(
        x=x_grid,
        y=y_grid,
        z=np.full_like(x_grid, -0.08),
        surfacecolor=gradient,
        colorscale=[[0.0, '#F0F2F5'], [1.0, '#E0E4E8']],
        cmin=0,
        cmax=1,
        opacity=1.0,
        showscale=False,
        hoverinfo='skip',
        showlegend=False,
    ))


def add_ground_patch(fig, center_x, center_y, width, depth, color, label, opacity=1.0):
    x_min, x_max = center_x - width / 2, center_x + width / 2
    y_min, y_max = center_y - depth / 2, center_y + depth / 2
    fig.add_trace(go.Mesh3d(
        x=[x_min, x_max, x_max, x_min],
        y=[y_min, y_min, y_max, y_max],
        z=[0.02, 0.02, 0.02, 0.02],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color,
        opacity=opacity,
        hovertemplate=f'{label}<extra></extra>',
        showlegend=False,
    ))


def add_ring(fig, center_x, center_y, radius, z_value, color, width=3, opacity=1.0):
    theta = np.linspace(0, 2 * np.pi, 72)
    fig.add_trace(go.Scatter3d(
        x=center_x + radius * np.cos(theta),
        y=center_y + radius * np.sin(theta),
        z=np.full_like(theta, z_value),
        mode='lines',
        line=dict(color=color, width=width),
        opacity=opacity,
        hoverinfo='skip',
        showlegend=False,
    ))


def add_tank_ladder(fig, tank):
    ladder_x = tank['x'] + tank['radius'] * 0.92
    rail_offset = 1.15
    x_values = [ladder_x, ladder_x, None, ladder_x, ladder_x, None]
    y_values = [tank['y'] - rail_offset, tank['y'] - rail_offset, None,
                tank['y'] + rail_offset, tank['y'] + rail_offset, None]
    z_values = [0.7, tank['height'] * 0.9, None, 0.7, tank['height'] * 0.9, None]
    for rung_z in np.linspace(2.0, tank['height'] * 0.86, 7):
        x_values.extend((ladder_x, ladder_x, None))
        y_values.extend((tank['y'] - rail_offset, tank['y'] + rail_offset, None))
        z_values.extend((rung_z, rung_z, None))
    fig.add_trace(go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode='lines',
        line=dict(color='#BDC3C7', width=2),
        hoverinfo='skip',
        showlegend=False,
    ))


def add_dashed_line(fig, start, end, dash_count, z_value, color, width=2):
    start_point = np.asarray(start, dtype=float)
    end_point = np.asarray(end, dtype=float)
    direction = end_point - start_point
    x_values, y_values, z_values = [], [], []
    for index in range(dash_count):
        segment_start = start_point + direction * (index / dash_count)
        segment_end = start_point + direction * ((index + 0.55) / dash_count)
        x_values.extend((segment_start[0], segment_end[0], None))
        y_values.extend((segment_start[1], segment_end[1], None))
        z_values.extend((z_value, z_value, None))
    fig.add_trace(go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode='lines',
        line=dict(color=color, width=width),
        hoverinfo='skip',
        showlegend=False,
    ))


def add_tank(fig, tank, is_selected):
    segments = 48
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring_x = tank['x'] + tank['radius'] * np.cos(theta)
    ring_y = tank['y'] + tank['radius'] * np.sin(theta)
    height = float(tank['height'])
    vx = np.concatenate([ring_x, ring_x, [float(tank['x']), float(tank['x'])]])
    vy = np.concatenate([ring_y, ring_y, [float(tank['y']), float(tank['y'])]])
    vz = np.concatenate([np.zeros(segments), np.full(segments, height), [0.0, height]])
    i_idx, j_idx, k_idx = [], [], []
    bottom_center, top_center = 2 * segments, 2 * segments + 1
    for a in range(segments):
        b = (a + 1) % segments
        a_top, b_top = a + segments, b + segments
        i_idx += [a, a]                     # 侧壁（两个三角形）
        j_idx += [b, b_top]
        k_idx += [b_top, a_top]
        i_idx.append(top_center)            # 顶盖
        j_idx.append(a_top)
        k_idx.append(b_top)
        i_idx.append(bottom_center)         # 底盖
        j_idx.append(b)
        k_idx.append(a)
    hover_text = (
        f"<b>{tank['code']}</b><br>介质：{tank['medium']}<br>容量：{tank['capacity']}"
        f"<br>直径：{tank['diameter']}<br>高度：{tank['height_label']}<extra></extra>"
    )
    fig.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz, i=i_idx, j=j_idx, k=k_idx,
        color=tank['color'], opacity=1.0, flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.4, roughness=0.5, fresnel=0.2),
        name=tank['code'], hovertemplate=hover_text,
        customdata=np.full(vx.shape, tank['code'], dtype=object), showlegend=False,
    ))
    edge_color = tank.get('edge_color', '#BDC3C7')
    add_ring(fig, tank['x'], tank['y'], tank['radius'] + 1.0, 0.35, '#E8ECEF', width=4)
    add_ring(fig, tank['x'], tank['y'], tank['radius'] + 0.4, height + 0.25, edge_color, width=4)
    add_ring(fig, tank['x'], tank['y'], tank['radius'] + 0.85, height + 0.75, '#BDC3C7', width=2)
    add_tank_ladder(fig, tank)
    if is_selected:
        add_ring(fig, tank['x'], tank['y'], tank['radius'] + 2.3, height + 0.45, '#F2C14E', width=6)


def add_impact_shell(fig, center_x, center_y, radius, color):
    # 半球壳也用 Mesh3d，避免与实体网格混用不同渲染模块导致遮挡错乱
    theta = np.linspace(0, 2 * np.pi, 42)
    phi = np.linspace(0, np.pi / 2, 20)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x_grid = center_x + radius * np.sin(phi_grid) * np.cos(theta_grid)
    y_grid = center_y + radius * np.sin(phi_grid) * np.sin(theta_grid)
    z_grid = radius * np.cos(phi_grid)
    rows, cols = x_grid.shape
    i_idx, j_idx, k_idx = [], [], []
    for r in range(rows - 1):
        for c in range(cols - 1):
            p00, p01 = r * cols + c, r * cols + c + 1
            p10, p11 = (r + 1) * cols + c, (r + 1) * cols + c + 1
            i_idx += [p00, p00]
            j_idx += [p01, p11]
            k_idx += [p11, p10]
    fig.add_trace(go.Mesh3d(
        x=x_grid.ravel(), y=y_grid.ravel(), z=z_grid.ravel(),
        i=i_idx, j=j_idx, k=k_idx, color=color, opacity=0.055,
        lighting=dict(ambient=0.8, diffuse=0.6, specular=0.1, roughness=0.9, fresnel=0.05),
        hoverinfo='skip', showlegend=False,
    ))


def calculate_downwind_vector(wind_direction_deg):
    downwind_angle = (float(wind_direction_deg) + 180) % 360
    angle_radians = np.deg2rad(downwind_angle)
    direction = np.array([np.sin(angle_radians), np.cos(angle_radians)])
    return downwind_angle, direction


def add_downwind_plume(fig, center_x, center_y, radius, wind_direction_deg):
    _, direction = calculate_downwind_vector(wind_direction_deg)
    perpendicular = np.array([-direction[1], direction[0]])
    start = np.array([center_x, center_y], dtype=float)
    end = start + direction * radius
    start_width = max(2.0, radius * 0.06)
    end_width = max(4.0, radius * 0.28)
    vertices = np.array([
        start - perpendicular * start_width,
        start + perpendicular * start_width,
        end + perpendicular * end_width,
        end - perpendicular * end_width,
    ])
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=[0.3, 0.3, 0.3, 0.3],
        i=[0, 0], j=[1, 2], k=[2, 3], color='#D39C52', opacity=0.20,
        name='下风向影响范围', hovertemplate='下风向影响范围（示意）<extra></extra>',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[center_x, end[0]], y=[center_y, end[1]], z=[0.5, 0.5], mode='lines',
        line=dict(color='#B36B36', width=4), hoverinfo='skip', showlegend=False,
    ))


def add_gas_dispersion_streamlines(
    fig,
    tank,
    model_scenario,
    wind_direction_deg,
    wind_speed,
    length_factor=1.0,
    animation_phase=0.0,
):
    scenario = SCENARIO_LIBRARY[model_scenario]
    length = scenario['dispersion_length'] * np.clip(float(length_factor), 0.15, 1.0)
    _, downwind = calculate_downwind_vector(wind_direction_deg)
    crosswind = np.array([-downwind[1], downwind[0]])
    distances = np.linspace(0, length, 14)
    speed_factor = np.clip(float(wind_speed) / 4.7, 0.5, 2.5)
    trace_indices = []

    for spread in (-0.05, 0.0, 0.05):
        offsets = spread * tank['radius'] * 2.0 + distances * spread * speed_factor
        x_values = tank['x'] + crosswind[0] * offsets + downwind[0] * distances
        y_values = tank['y'] + crosswind[1] * offsets + downwind[1] * distances
        progress = distances / max(length, 1)
        z_values = tank['height'] + 1.5 + (3.0 + 0.15 * float(wind_speed)) * np.sin(
            np.pi * progress + 2 * np.pi * float(animation_phase)
        )
        is_main_line = abs(spread) < 1e-12
        fig.add_trace(go.Scatter3d(
            x=x_values, y=y_values, z=z_values,
            mode='lines',
            line=dict(color='#B64545' if is_main_line else '#D97A7A', width=3),
            customdata=distances,
            name='泄漏扩散流线',
            hovertemplate=f'{model_scenario}扩散流线 · 下风向约%{{customdata:.0f}} m<extra></extra>',
            opacity=0.95 if is_main_line else 0.55, showlegend=False,
        ))
        trace_indices.append(len(fig.data) - 1)
    return trace_indices


def add_risk_heatmap(fig, center_x, center_y, radius):
    grid_radius = max(float(radius) * 0.72, 18.0)
    x_values = np.linspace(center_x - grid_radius, center_x + grid_radius, 36)
    y_values = np.linspace(center_y - grid_radius, center_y + grid_radius, 36)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    distance_grid = np.hypot(x_grid - center_x, y_grid - center_y)
    risk_values = np.where(
        distance_grid <= grid_radius,
        np.clip(1 - distance_grid / grid_radius, 0, 1),
        np.nan,
    )
    fig.add_trace(go.Surface(
        x=x_grid,
        y=y_grid,
        z=np.full_like(x_grid, 0.28),
        surfacecolor=risk_values,
        colorscale=[[0.0, '#F6E7A7'], [0.55, '#F2C14E'], [1.0, '#B64545']],
        cmin=0,
        cmax=1,
        opacity=0.42,
        showscale=False,
        hovertemplate='风险热力图（脱敏示意）<extra></extra>',
        showlegend=False,
    ))


def add_random_impact_marker(fig, tank):
    fig.add_trace(go.Scatter3d(
        x=[tank['x']], y=[tank['y']], z=[tank['height'] + 7.0],
        mode='text', text=['随机受影响区（代表点）'],
        textposition='top center',
        textfont=dict(size=11, color='#7A5195', family='Microsoft YaHei'),
        hovertemplate='随机冲击代表点（脱敏情景）<extra></extra>', showlegend=False,
    ))


def add_measure_spatial_markers(fig, measure_names, incident_tank):
    marker_positions = {
        '密封改造': (incident_tank['x'], incident_tank['y'], incident_tank['height'] + 7.5),
        '泄漏监测': (incident_tank['x'] - incident_tank['radius'] - 5, incident_tank['y'], 7.0),
        '应急切断': (incident_tank['x'] + incident_tank['radius'] + 5, incident_tank['y'], 7.0),
        '油气回收': (160.0, 180.0, 14.5),
        '泵组节能': (50.0, -18.0, 18.5),
    }
    for measure_name in measure_names:
        info = MEASURE_SPATIAL_INFO[measure_name]
        x_value, y_value, z_value = marker_positions[measure_name]
        fig.add_trace(go.Scatter3d(
            x=[x_value], y=[y_value], z=[z_value],
            mode='text', text=[measure_name], textposition='top center',
            textfont=dict(size=10, color=info['color'], family='Microsoft YaHei'),
            hovertemplate=(
                f"<b>{measure_name}</b><br>实施位置：{info['location']}"
                f"<br>作用：{info['mechanism']}<extra></extra>"
            ),
            showlegend=False,
        ))


def add_transport_route(fig, points, color, route_label, line_width=4, arrow_scale=8, show_arrow=False):
    coordinates = np.asarray(points, dtype=float)
    fig.add_trace(go.Scatter3d(
        x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
        mode='lines',
        line=dict(color=color, width=line_width),
        name=route_label,
        hovertemplate=f'{route_label}<extra></extra>',
        showlegend=False,
    ))

    if not show_arrow:
        return

    final_segment = coordinates[-1] - coordinates[-2]
    segment_length = np.linalg.norm(final_segment)
    if segment_length <= 1e-9:
        return
    unit_direction = final_segment / segment_length
    arrow_position = coordinates[-2] + final_segment * 0.72
    fig.add_trace(go.Cone(
        x=[arrow_position[0]], y=[arrow_position[1]], z=[arrow_position[2]],
        u=[unit_direction[0] * arrow_scale], v=[unit_direction[1] * arrow_scale], w=[unit_direction[2] * arrow_scale],
        anchor='tip', sizemode='absolute', sizeref=7,
        colorscale=[[0, color], [1, color]], showscale=False,
        hovertemplate=f'{route_label}流向<extra></extra>', showlegend=False,
    ))


def add_transport_branches(fig, branches, color, route_label):
    x_values, y_values, z_values = [], [], []
    for start, end in branches:
        x_values.extend((start[0], end[0], None))
        y_values.extend((start[1], end[1], None))
        z_values.extend((start[2], end[2], None))
    fig.add_trace(go.Scatter3d(
        x=x_values, y=y_values, z=z_values,
        mode='lines', line=dict(color=color, width=3),
        opacity=0.9, name=route_label,
        hovertemplate=f'{route_label}<extra></extra>', showlegend=False,
    ))


def add_transport_streamlines(fig):
    trunk_color = '#BDC3C7'
    tank_farm_header = np.array([110.0, 165.0, 7.0])
    add_transport_route(
        fig,
        [np.array([120.0, 340.0, 7.0]), np.array([120.0, 235.0, 7.0]), tank_farm_header],
        trunk_color, '码头至罐区主干管', line_width=4,
    )
    add_transport_route(
        fig,
        [tank_farm_header, np.array([160.0, 165.0, 7.0]), np.array([160.0, 180.0, 9.0])],
        trunk_color, '罐区至装车台主干管', line_width=4,
    )
    add_transport_route(
        fig,
        [tank_farm_header, np.array([110.0, 45.0, 7.0]), np.array([145.0, -18.0, 3.0])],
        trunk_color, '罐区至油污水收集池主干管', line_width=4,
    )


def add_site_label(fig, x, y, z, text, color='#475569', size=11):
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z], mode='text', text=[text],
        textfont=dict(size=size, color=color, family='Microsoft YaHei'),
        hoverinfo='skip', showlegend=False,
    ))


@st.cache_data(show_spinner=False, max_entries=16)
def build_tank_model(
    selected_tank,
    model_scenario,
    incident_tank_code,
    impact_radii,
    view_mode,
    wind_direction_deg,
    wind_speed,
    show_dispersion_lines,
    show_impact_shells,
    show_risk_heatmap,
    animation_phase,
    dispersion_factor,
):
    fig = go.Figure()
    add_ground_surface(fig)
    add_ground_patch(fig, 110, 91, 232, 158, '#E9EEF2', '罐区作业区', 0.96)
    grid_x, grid_y = [], []
    for x_value in range(-100, 351, 50):
        grid_x.extend((x_value, x_value, None))
        grid_y.extend((-80, 420, None))
    for y_value in range(-50, 401, 50):
        grid_x.extend((-100, 350, None))
        grid_y.extend((y_value, y_value, None))
    fig.add_trace(go.Scatter3d(
        x=grid_x, y=grid_y, z=[0.08 if value is not None else None for value in grid_x],
        mode='lines', line=dict(color='rgba(113, 135, 156, 0.32)', width=1),
        hoverinfo='skip', showlegend=False,
    ))

    add_ground_patch(fig, 115, 385, 500, 90, '#A8D0E6', '北侧水域', 0.40)
    add_box(fig, 115, 354, 250, 18, 3.2, '#4E5965', '北侧码头', 1.0, base_z=0.10)
    for facility_x in (25, 85, 145, 205):
        add_box(fig, facility_x, 365, 12, 9, 7, '#4E5965', '码头装卸设施', 1.0, base_z=0.10)
    add_site_label(fig, 115, 405, 4.5, '北侧水域及码头', size=12)

    add_box(fig, 115, -58, 480, 12, 0.40, '#B7BEC7', '南侧道路', 1.0, base_z=0.08)
    add_dashed_line(fig, (-115, -58), (345, -58), 18, 0.54, '#FFFFFF', width=2)
    add_site_label(fig, 245, -73, 2.2, '南侧道路')

    add_box(fig, 238, 128, 6, 292, 0.36, '#C8CED4', '东侧规划道路', 1.0, base_z=0.08)
    add_site_label(fig, 251, 250, 2.0, '东侧规划道路')

    add_box(fig, 110, 15, 228, 6, 0.36, '#C8CED4', '环形消防道路', 1.0, base_z=0.08)
    add_box(fig, 110, 167, 228, 6, 0.36, '#C8CED4', '环形消防道路', 1.0, base_z=0.08)
    add_box(fig, -4, 91, 6, 158, 0.36, '#C8CED4', '环形消防道路', 1.0, base_z=0.08)
    add_box(fig, 224, 91, 6, 158, 0.36, '#C8CED4', '环形消防道路', 1.0, base_z=0.08)

    west_tanks = []
    for index, center_y in enumerate((12, 52, 92, 132), start=1):
        west_tanks.append({
            'code': f'西侧-{index}', 'medium': '储罐', 'capacity': '约2万m³',
            'diameter': '约40m', 'height_label': '约15m', 'radius': 20, 'height': 15,
            'x': -70, 'y': center_y, 'color': '#A9A9A9', 'edge_color': '#C8CED4',
        })
    for tank in west_tanks:
        add_tank(fig, tank, False)
    add_site_label(fig, -72, 166, 18, '西侧原有罐区')

    add_box(fig, 318, 100, 80, 100, 15, '#D3D3D3', '东侧企业', 1.0, base_z=0.10)
    add_site_label(fig, 318, 160, 19, '东侧企业')

    dike_color = '#A89F91'
    add_box(fig, 110, 31, 202, 2.2, 2.2, dike_color, '防火堤', 1.0, base_z=0.10)
    add_box(fig, 110, 149, 202, 2.2, 2.2, dike_color, '防火堤', 1.0, base_z=0.10)
    add_box(fig, 11, 90, 2.2, 122, 2.2, dike_color, '防火堤', 1.0, base_z=0.10)
    add_box(fig, 209, 90, 2.2, 122, 2.2, dike_color, '防火堤', 1.0, base_z=0.10)
    add_box(fig, 105, 90, 1.9, 116, 1.9, '#B8B0A4', '隔堤', 1.0, base_z=0.10)

    for tank in MODEL_TANKS:
        add_tank(fig, tank, tank['code'] == selected_tank)

    add_box(fig, 160, 188, 60, 20, 8, '#7F8C8D', '装车台', 1.0, base_z=0.10)
    add_box(fig, 160, 176, 64, 2.5, 2.5, dike_color, '装车台内围墙', 1.0, base_z=0.10)
    add_box(fig, 160, 200, 64, 2.5, 2.5, dike_color, '装车台内围墙', 1.0, base_z=0.10)
    add_box(fig, 127, 188, 2.5, 27, 2.5, dike_color, '装车台内围墙', 1.0, base_z=0.10)
    add_box(fig, 193, 188, 2.5, 27, 2.5, dike_color, '装车台内围墙', 1.0, base_z=0.10)
    add_site_label(fig, 160, 205, 11.5, '装车台', size=10)

    add_box(fig, 48, -18, 42, 22, 12, '#7F8C8D', '动力辅房', 1.0, base_z=0.10)
    add_box(fig, 145, -18, 32, 22, 2.2, '#7F8C8D', '油污水收集池', 1.0, base_z=0.10)
    add_site_label(fig, 48, -34, 15, '动力辅房', size=10)
    add_site_label(fig, 145, -34, 4.5, '油污水收集池', size=10)

    for index, center_x in enumerate((4, 88), start=1):
        water_tank = {
            'code': f'消防水罐-{index}', 'medium': '消防水', 'capacity': '约1万m³',
            'diameter': '约25m', 'height_label': '约20m', 'radius': 12.5, 'height': 20,
            'x': center_x, 'y': -18, 'color': '#7F8C8D', 'edge_color': '#4C78A8',
        }
        add_tank(fig, water_tank, False)
    add_site_label(fig, 46, -42, 3.0, '消防水罐', size=10)

    fig.add_trace(go.Scatter3d(
        x=[30, 80, 130, 180, 180, 160], y=[60, 60, 60, 60, 120, 188],
        z=[22.7, 22.7, 22.7, 22.7, 22.7, 8.8], mode='lines',
        line=dict(color='#BDC3C7', width=5), name='工艺管道',
        hovertemplate='工艺管道（示意）<extra></extra>', showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[15, 205, 205, 15, 15], y=[25, 25, 155, 155, 25],
        z=[1.7, 1.7, 1.7, 1.7, 1.7], mode='lines',
        line=dict(color='#BDC3C7', width=4), name='消防管道',
        hovertemplate='消防管道（示意）<extra></extra>', showlegend=False,
    ))

    _, wind_vector = calculate_downwind_vector(wind_direction_deg)
    wind_start = np.array([330.0, 265.0])
    wind_end = wind_start + wind_vector * 32
    fig.add_trace(go.Scatter3d(
        x=[wind_start[0], wind_end[0]], y=[wind_start[1], wind_end[1]], z=[4.0, 4.0], mode='lines',
        line=dict(color='#163B65', width=6), hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Cone(
        x=[wind_end[0]], y=[wind_end[1]], z=[4.0],
        u=[wind_vector[0] * 8], v=[wind_vector[1] * 8], w=[0], anchor='tip',
        sizemode='absolute', sizeref=7, colorscale=[[0, '#163B65'], [1, '#163B65']],
        showscale=False, hoverinfo='skip', showlegend=False,
    ))
    add_transport_streamlines(fig)

    incident_tank = next(tank for tank in MODEL_TANKS if tank['code'] == incident_tank_code)
    if show_risk_heatmap:
        add_risk_heatmap(fig, incident_tank['x'], incident_tank['y'], impact_radii[-1])
    if show_impact_shells:
        for radius, color in zip(impact_radii, ('#B64545', '#D39C52', '#4C78A8')):
            add_impact_shell(fig, incident_tank['x'], incident_tank['y'], radius, color)
    if show_dispersion_lines:
        if SCENARIO_LIBRARY[model_scenario].get('visual_mode') == 'random_impact':
            add_random_impact_marker(fig, incident_tank)
        add_gas_dispersion_streamlines(
            fig,
            incident_tank,
            model_scenario,
            wind_direction_deg,
            wind_speed,
            dispersion_factor,
            animation_phase,
        )

    camera_reset = dict(eye=dict(x=1.65, y=-1.75, z=1.18), center=dict(x=0, y=0, z=0))
    camera_top = dict(eye=dict(x=0.01, y=0.01, z=2.65), center=dict(x=0, y=0, z=0))
    camera_side = dict(eye=dict(x=2.45, y=0.01, z=0.62), center=dict(x=0, y=0, z=0))
    cameras = {'重置视角': camera_reset, '俯视视角': camera_top, '侧视视角': camera_side}
    fig.update_layout(
        height=840, margin=dict(l=0, r=0, t=54, b=0), paper_bgcolor=WHITE,
        title=dict(text='港口配套区完整沙盘（脱敏示意）', x=0.02, y=0.97, font=dict(color=NAVY, size=17)),
        clickmode='event',
        scene=dict(
            xaxis=dict(title='', range=[-140, 390], showbackground=True, backgroundcolor='#F7FAFC', gridcolor='#D7E0E8', showticklabels=False),
            yaxis=dict(title='', range=[-100, 440], showbackground=True, backgroundcolor='#F7FAFC', gridcolor='#D7E0E8', showticklabels=False),
            zaxis=dict(title='', range=[-2, 60], showbackground=True, backgroundcolor='#F7FAFC', gridcolor='#D7E0E8', showticklabels=False),
            camera=cameras.get(view_mode, camera_reset), aspectmode='manual', aspectratio=dict(x=1.18, y=1.20, z=0.45),
        ),
    )
    return fig

project_details_container = None


def render_module_description(content):
    global project_details_container
    project_details_container = st.expander("项目详情", expanded=False)
    project_details_container.markdown(f"**模块说明**\n\n{content}")


def render_method(method, basis):
    if project_details_container is not None:
        project_details_container.markdown(f"**方法与依据**\n\n**方法：** {method}\n\n**依据：** {basis}")


def render_sources(sources):
    if project_details_container is not None:
        source_lines = "\n".join(f"- {source}" for source in sources)
        project_details_container.markdown(f"**数据来源**\n\n{source_lines}")


def render_process_flow(steps):
    with st.container(key=f"process_flow_{steps[0][1]}"):
        flow_columns = st.columns(
            [2, 0.3, 2, 0.3, 2, 0.3, 2],
            gap="small",
            vertical_alignment="center",
            wrap=False,
        )
        for index, (title, description) in enumerate(steps):
            with flow_columns[index * 2]:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(description)
            if index < len(steps) - 1:
                with flow_columns[index * 2 + 1]:
                    st.markdown("### →")


def label_bar_values(axis, bars, fmt='%.0f', fontsize=8):
    axis.bar_label(bars, fmt=fmt, padding=3, fontsize=fontsize, color=SLATE)


def render_resilience_logic(result):
    matrix = np.vstack([result['before_matrix'], result['after_matrix']])
    details = calculate_entropy_details(matrix)
    indicator_rows = (
        ('吸收能力', '设备完整率', '正向', '完好关键设备数 / 关键设备总数', '设备台账、检查记录'),
        ('吸收能力', '初始功能保持率', '正向', '冲击后初始作业能力 / 事故前作业能力', '监测记录、事故或演练记录'),
        ('吸收能力', '隔离切断响应时间', '逆向', '从事故识别到有效隔离切断的时间', '报警与切断日志'),
        ('适应能力', '监测预警及时率', '正向', '规定时间内有效预警次数 / 应预警事件数', '监测报警记录'),
        ('适应能力', '关键作业维持率', '正向', '冲击期间可维持的关键作业量 / 正常关键作业量', '作业记录、演练记录'),
        ('适应能力', '应急资源调配时间', '逆向', '从资源需求确认到资源到位的时间', '应急调度与演练记录'),
        ('恢复能力', '恢复至基线时间', '逆向', '从事故发生到恢复正常作业水平的时间', '处置与恢复记录'),
        ('恢复能力', '修复任务完成率', '正向', '评价时点已完成的必要修复任务数 / 必要修复任务总数', '维修计划、任务验收记录'),
        ('恢复能力', '恢复验证合格率', '正向', '恢复后验证合格项目数 / 应验证项目数', '复工检查、验证记录'),
    )
    indicator_frame = pd.DataFrame(indicator_rows, columns=('能力维度', '候选底层指标', '指标方向', '示例测量口径', '所需资料'))
    indicator_frame.insert(0, '指标编号', [f'C{index}' for index in range(1, len(indicator_rows) + 1)])
    indicator_frame.insert(1, '指标类别', ['设施防护'] * 3 + ['监测预警'] + ['应急组织'] * 2 + ['修复重启'] * 3)
    with st.container(key="resilience_logic"):
        st.subheader("三维能力的指标体系与评价链路")
        st.markdown("**当前实际计算链路：** 预设情景三维得分与措施系数 → 改造前后样本矩阵 → 列占比 → 信息熵 → 差异系数 → 熵权 → 综合韧性得分。")
        st.caption("雷达图展示三维得分；综合指标展示熵权加权得分；恢复曲线由情景损失率和恢复天数生成。")
        st.markdown("**客观赋权的依据：** 某一维度在评价样本中的分布越不均匀，信息熵越低、差异系数越大，获得的权重越高。权重来自数据的区分度，不由用户手动指定，也不等于该维度在所有项目中都更重要。")
        st.caption(f"当前采用 {len(result['scenarios'])} 种情景的改造前、改造后得分，共 {details['sample_count']} 个样本；三个维度均使用 0–10 分，且分数越高表示能力越强。")
        st.caption("以下为候选底层指标体系（未来扩展方向），当前版本尚未接入评分。")
        with st.expander("底层指标口径（候选体系）", expanded=False):
            st.dataframe(indicator_frame, hide_index=True)
            st.caption("正向指标越高越好，逆向指标越低越好。上述划分为讨论用示例，并非已核验的文献结论；正式采用前需确认文献出处、阈值、评价时点、缺失值处理与维度内汇总规则。")
        with st.expander("熵权计算依据与样本矩阵", expanded=False):
            st.markdown(
                r'<span class="notranslate" translate="no">**列占比：** $p_{ij}=\frac{x_{ij}}{\sum_{k=1}^{n}x_{kj}}$</span>' "\n\n"
                r'<span class="notranslate" translate="no">**信息熵：** $e_j=-\frac{1}{\ln(n)}\sum_{i=1}^{n}p_{ij}\ln(p_{ij})$</span>' "\n\n"
                r'<span class="notranslate" translate="no">**差异系数与权重：** $d_j=1-e_j,\quad w_j=\frac{d_j}{\sum_{\ell=1}^{m}d_{\ell}}$</span>' "\n\n"
                r'<span class="notranslate" translate="no">**综合得分：** $S_i=\sum_{j=1}^{m}w_jx_{ij}$</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"i、k 表示评价样本，j、ℓ 表示能力维度；n 为样本数（当前 {details['sample_count']}），m 为维度数（当前 {matrix.shape[1]}）。ln 表示自然对数，0·ln(0) 按 0 处理。各维度差异系数均接近 0 时采用等权。")
            sample_frame = pd.DataFrame(matrix, columns=result['dimensions'])
            sample_frame.insert(0, '评价样本', [
                f'{scenario} · {phase}'
                for phase in ('改造前', '改造后')
                for scenario in result['scenarios']
            ])
            st.dataframe(sample_frame.round(4), hide_index=True)
            st.markdown("**当前得分来源：** 改造前使用预设的情景三维得分并按冲击系数修正；改造后用固定措施效果系数推演，不是由上表九项底层指标直接计算。当前只对三维得分按列计算占比，尚未对候选原始指标做方向统一与标准化。")
            st.markdown("**客观性的边界：** 熵权由当前样本自动计算，但指标选择、情景初始得分、措施效果系数仍含模型设定。样本或措施改变时权重也会改变；数据区分度不能替代数据质量和工程有效性验证。")
            st.caption("恢复曲线依据情景损失率与恢复时间生成，关键点用于解释三种能力，不是从综合韧性得分反推的实测轨迹。")
        weight_frame = pd.DataFrame({
            '韧性维度': result['dimensions'],
            '信息熵 e_j': np.round(details['entropy'], 6),
            '差异系数 d_j': np.round(details['divergence'], 6),
            '熵权': np.round(result['weights'], 4),
            '权重占比': [f"{weight:.1%}" for weight in result['weights']],
        })
        st.subheader("熵权结果")
        st.dataframe(weight_frame, hide_index=True)
    return details


def build_report(qra_result, resilience_result, scenario_name, scenario_index, selected_plan, recommended_plan, budget):
    improvement = ((resilience_result['after_scores'][scenario_index]
                    - resilience_result['before_scores'][scenario_index])
                   / resilience_result['before_scores'][scenario_index] * 100)
    recommended_economics = calculate_economic_metrics(recommended_plan)
    return f"""港口危化品罐区风险可视化与安全绿色改造辅助决策系统

一、系统概述
- 本系统以脱敏情景参数联动 QRA 风险识别、熵权情景韧性、措施组合优化、投入产出分析和三维空间化展示。
- 输出用于课堂研究和方案比选，实际工程应以现场调查、专业 QRA、设计审查及投资测算结果为准。

二、关键指标
- 当前 R 值：{qra_result['risk_value']:.1f}
- 风险等级：{qra_result['risk_level']}
- 当前事故情景：{scenario_name}
- 改造后综合韧性：{resilience_result['after_scores'][scenario_index]:.2f} / 10
- 恢复时间：{resilience_result['recovery_before'][scenario_index]:.1f} 天 -> {resilience_result['recovery_after'][scenario_index]:.1f} 天
- 推荐投资：{recommended_plan['investment']:.1f} 万元
- 年风险收益：约 {recommended_economics['annual_risk_benefit']:.1f} 万元
- 投资回收期：约 {recommended_economics['payback_years']:.1f} 年

三、推荐方案
- 预算上限：{budget:.1f} 万元
- 推荐措施：{recommended_plan['plan_name']}
- 风险降低：{recommended_plan['risk_reduction']:.1%}
- 恢复时间缩短：{recommended_plan['recovery_reduction']:.1%}
- 污染物减排：{recommended_plan['emission_reduction']:.1%}
- 能耗降低：{recommended_plan['energy_reduction']:.1%}
- 安全维度得分：{recommended_plan['safety_score']:.1f} / 100
- 经济维度得分：{recommended_plan['economic_score']:.1f} / 100
- 环境维度得分：{recommended_plan['environmental_score']:.1f} / 100
- 三维综合得分：{recommended_plan['score']:.1f} / 100

四、当前手动组合
- 当前措施：{selected_plan['plan_name']}
- 当前投资：{selected_plan['investment']:.1f} 万元
- 当前安全维度得分：{selected_plan['safety_score']:.1f} / 100
- 当前经济维度得分：{selected_plan['economic_score']:.1f} / 100
- 当前环境维度得分：{selected_plan['environmental_score']:.1f} / 100
- 当前三维综合得分：{selected_plan['score']:.1f} / 100

五、决策链总结
- 风险识别确定重点泄漏后果与影响范围；韧性评价量化当前情景的吸收、适应和恢复能力；措施组合以安全、绿色与经济指标共同排序；三维模型将事故影响和措施落点进行空间化展示。
- 当前情景下，所选组合使综合韧性提升 {improvement:.1f}%，为后续工程复核提供参数化比选依据。
"""


st.markdown(
    """
    <style>
    .block-container {max-width: 1520px; padding-top: 3rem; padding-bottom: 2.2rem;}
    html, body, [class*="css"] {font-family: "Microsoft YaHei", "SimHei", sans-serif;}
    [data-testid="stSidebar"] {background: #F4F7FA; border-right: 1px solid #D7E0E8;}
    .system-header {box-sizing: border-box; max-width: 100%; border-left: 5px solid #163B65; border-bottom: 1px solid #D7E0E8; padding: 0.9rem 1.2rem; margin-bottom: 0.85rem; overflow-wrap: anywhere;}
    .system-kicker {color: #4C78A8; font-size: 0.78rem; font-weight: 700;}
    .system-title {color: #163B65; font-size: 1.7rem; font-weight: 700; margin: 0.18rem 0; line-height: 1.3; overflow-wrap: anywhere;}
    .system-subtitle {color: #475569; font-size: 0.95rem; margin: 0; line-height: 1.5;}
    [data-testid="stHorizontalBlock"] {gap: 0.75rem; align-items: stretch;}
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {min-width: 0;}
    [class*="st-key-process_flow_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(even) {min-width: 2rem !important;}
    [class*="st-key-process_flow_"] [data-testid="stColumn"]:nth-child(even) h3 {text-align: center; padding: 0;}
    [data-testid="stMetric"] {border: 1px solid #D7E0E8; border-top: 3px solid #163B65; border-radius: 3px; padding: 0.7rem 0.8rem; background: #FFFFFF; min-width: 0; min-height: 5.3rem; height: auto; overflow: visible;}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {min-width: 0; white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere !important; word-break: break-word;}
    [data-testid="stMetricValue"] {line-height: 1.25; font-size: 1.2rem;}
    [data-testid="stExpander"] {border: 1px solid #D7E0E8; border-radius: 2px; background: #FFFFFF;}
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {gap: 0.15rem;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding: 0.42rem 0.6rem; border-radius: 2px;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background: #E3EBF3; color: #163B65; font-weight: 700;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    .element-container:has(.page-visibility-style) {display: none !important;}
    .conclusion-note {border-left: 3px solid #4C78A8; background: #F4F7FA; padding: 0.7rem 0.85rem; color: #1E293B; line-height: 1.65; overflow-wrap: anywhere; word-break: break-word;}
    [data-testid="stCaptionContainer"] {line-height: 1.55; overflow-wrap: anywhere;}
    @media (max-width: 900px) {
        .block-container {padding-top: 2.5rem;}
        .system-header {padding: 0.75rem 0.9rem;}
        .system-title {font-size: 1.35rem;}
    }
    </style>
    <div class="system-header">
        <div class="system-kicker">PORT HAZARDOUS CHEMICALS TANK FARM</div>
        <div class="system-title">港口危化品罐区风险可视化与安全绿色改造辅助决策系统</div>
        <p class="system-subtitle">QRA 风险识别 · 熵权情景韧性 · 措施组合优化</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("本系统仅展示脱敏、取整与情景化参数，不对应任何真实企业、地点、道路、人员或精确工程数据。")

NAVIGATION_GROUPS = (
    ("① 风险预测", ("QRA风险可视化", "气象条件")),
    ("② 可视化", ("三维罐区模型",)),
    ("③ 措施落地", ("熵权情景韧性", "措施组合推荐")),
    ("④ 综合结论", ("综合结论报告",)),
    ("⑤ 项目详情", ("数据来源说明", "系统架构图", "版本与适用范围")),
)


def select_navigation_page(widget_key):
    st.session_state["active_page"] = st.session_state[widget_key]


if "active_page" not in st.session_state:
    st.session_state["active_page"] = "QRA风险可视化"

with st.sidebar:
    st.subheader("系统导航")
    st.caption("展开分组后选择右侧展示模块。")
    for group_index, (group_title, pages) in enumerate(NAVIGATION_GROUPS):
        navigation_key = f"navigation_group_{group_index}"
        selected_for_group = (
            st.session_state["active_page"]
            if st.session_state["active_page"] in pages else None
        )
        if st.session_state.get(navigation_key) != selected_for_group:
            st.session_state[navigation_key] = selected_for_group
        with st.expander(
            group_title,
            expanded=st.session_state["active_page"] in pages,
        ):
            st.radio(
                "选择功能模块",
                pages,
                index=None,
                key=navigation_key,
                label_visibility="collapsed",
                on_change=select_navigation_page,
                args=(navigation_key,),
            )
    st.divider()
    st.header("决策参数")
    st.caption("修改参数后，风险、韧性和推荐措施组合会同步更新。")
    with st.expander("QRA 风险参数", expanded=False):
        st.caption("储量 0.5-20 万吨；校正系数 0.1-5；临界量 1-2,000 吨。")
        q_storage = st.number_input("介质储量（万吨）", min_value=0.5, max_value=20.0, value=4.0, step=0.5)
        st.caption("不同介质对应不同校正系数β，请根据实际介质手动调整β值。")
        alpha = st.number_input("厂外暴露人员校正系数 α", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
        beta = st.number_input("危险化学品校正系数 β", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        critical_quantity = st.number_input(
            "临界量 Q（吨）", min_value=1.0, max_value=2000.0, value=200.0, step=10.0,
            help="Q 为该介质的临界量（吨），请根据实际介质及适用标准填写。",
        )
        consequence_factor = st.number_input("后果情景修正系数（倍）", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
    with st.expander("组合优化参数", expanded=False):
        st.caption("枚举全部 32 种措施组合；超出预算的组合不参与推荐排序。")
        budget_wan = st.slider("改造预算上限（万元）", min_value=50, max_value=500, value=220, step=10)

all_plan_options = enumerate_plan_options(float(budget_wan), SCORING_SCHEMA_VERSION)
feasible_plans = enumerate_feasible_plans(float(budget_wan), SCORING_SCHEMA_VERSION)
system_recommended_plan = feasible_plans[0]
system_recommended_economics = calculate_economic_metrics(system_recommended_plan)

with st.sidebar:
    with st.expander("手动选择改造措施", expanded=False):
        st.caption("默认选中系统推荐组合，可按需调整。")
        current_measures = [
            measure['name']
            for index, measure in enumerate(MEASURE_LIBRARY)
            if st.toggle(
                measure['name'],
                value=measure['name'] in system_recommended_plan['measures'],
                key=f"measure_toggle_{index}",
            )
        ]

if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "小泄漏"
selected_scenario = st.session_state["selected_scenario"]

selected_plan = calculate_plan_effect(current_measures)
selected_plan.update(calculate_plan_scores(selected_plan))
selected_economics = calculate_economic_metrics(selected_plan)
selected_plan_rank = next(
    (index for index, plan in enumerate(feasible_plans, start=1)
     if plan['measures'] == selected_plan['measures']),
    None,
)
qra_result = calculate_qra(q_storage, alpha, beta, critical_quantity, consequence_factor)
resilience_result = calculate_resilience(selected_plan['measures'])
scenario_index = list(resilience_result['scenarios']).index(selected_scenario)
scenario_profile = SCENARIO_LIBRARY[selected_scenario]
recovery_days, before_ability, after_ability = calculate_recovery_curve(
    scenario_profile['loss'],
    scenario_profile['loss'] * (1 - selected_plan['risk_reduction']),
    resilience_result['recovery_before'][scenario_index],
    resilience_result['recovery_after'][scenario_index],
)
resilience_improvement = ((resilience_result['after_scores'][scenario_index]
                           - resilience_result['before_scores'][scenario_index])
                          / resilience_result['before_scores'][scenario_index] * 100)

comprehensive_strategy = max(feasible_plans, key=lambda plan: plan['score'])
risk_first_strategy = select_greedy_plan(budget_wan, '按风险高低依次改造')
low_cost_strategy = select_greedy_plan(budget_wan, '优先低成本')
strategy_plans = [
    ('按风险高低依次改造', risk_first_strategy),
    ('优先低成本', low_cost_strategy),
    ('综合优选', comprehensive_strategy),
]
strategy_scores = [plan['score'] for _, plan in strategy_plans]

page_section_keys = {
    "QRA风险可视化": ("page_qra",),
    "气象条件": ("page_weather",),
    "三维罐区模型": ("page_model",),
    "熵权情景韧性": ("page_resilience",),
    "措施组合推荐": ("page_plans", "page_strategy", "page_economy"),
    "综合结论报告": ("page_summary",),
    "数据来源说明": ("page_sources",),
    "系统架构图": ("page_architecture",),
    "版本与适用范围": ("page_version",),
}
visible_page_keys = set(page_section_keys[st.session_state["active_page"]])
hidden_page_rules = "".join(
    f".stVerticalBlock > div:has(> .st-key-{page_key}){{display:none!important;}}"
    for page_group in page_section_keys.values()
    for page_key in page_group
    if page_key not in visible_page_keys
)
st.markdown(
    f'<span class="page-visibility-style"></span><style>{hidden_page_rules}</style>',
    unsafe_allow_html=True,
)

with st.container(key="page_summary"):
    st.header("综合结论报告")
    render_process_flow((
        ("输入", "各模块关键指标"),
        ("计算", "汇总与决策链整合"),
        ("输出", "完整决策报告"),
        ("结论", "方案、投资、韧性、风险降低"),
    ))
    render_module_description("汇总当前 QRA 风险、事故情景韧性、系统推荐措施和投入产出结果，形成面向方案比选的完整决策报告。")
    render_method("多模块指标联动汇总", "风险识别、韧性评价、措施组合优化和投入产出分析共享同一组脱敏参数，综合结论随输入与手动组合实时更新。")
    st.subheader("系统概述")
    st.write("系统以脱敏情景参数联动 QRA 风险识别、熵权情景韧性、措施组合优化、投入产出分析和三维空间化展示，为港口危化品罐区安全绿色改造提供研究性辅助决策。")
    st.subheader("关键指标")
    summary_columns = st.columns(3)
    summary_columns[0].metric("R 值", f"{qra_result['risk_value']:.1f}")
    summary_columns[1].metric("改造后韧性", f"{resilience_result['after_scores'][scenario_index]:.2f} / 10")
    summary_columns[2].metric("恢复时间", f"{resilience_result['recovery_after'][scenario_index]:.1f} 天")
    summary_columns = st.columns(3)
    summary_columns[0].metric("推荐投资", f"{system_recommended_plan['investment']:.0f} 万元")
    summary_columns[1].metric("年风险收益", f"约 {system_recommended_economics['annual_risk_benefit']:.1f} 万元")
    summary_columns[2].metric(
        "投资回收期",
        f"约 {system_recommended_economics['payback_years']:.1f} 年"
        if system_recommended_economics['payback_years'] is not None else "不可回收",
    )
    st.subheader("推荐方案")
    st.markdown(f"**系统推荐：** {system_recommended_plan['plan_name']}")
    recommendation_columns = st.columns(4)
    recommendation_columns[0].metric("三维综合得分", f"{system_recommended_plan['score']:.1f} / 100")
    recommendation_columns[1].metric("安全维度得分", f"{system_recommended_plan['safety_score']:.1f} / 100")
    recommendation_columns[2].metric("经济维度得分", f"{system_recommended_plan['economic_score']:.1f} / 100")
    recommendation_columns[3].metric("环境维度得分", f"{system_recommended_plan['environmental_score']:.1f} / 100")
    recommendation_columns = st.columns(3)
    recommendation_columns[0].metric("风险降低", f"{system_recommended_plan['risk_reduction']:.1%}")
    recommendation_columns[1].metric("恢复改善", f"{system_recommended_plan['recovery_reduction']:.1%}")
    recommendation_columns[2].metric("投入产出比", f"{system_recommended_economics['roi']:.1%}")
    st.subheader("决策链总结")
    st.markdown(
        f"<div class='conclusion-note'><strong>风险识别 → 韧性评估 → 措施优选 → 经济核验 → 空间化展示：</strong>"
        f"在{selected_scenario}情景下，当前组合“{selected_plan['plan_name']}”使综合韧性由 "
        f"{resilience_result['before_scores'][scenario_index]:.2f} 提升至 "
        f"{resilience_result['after_scores'][scenario_index]:.2f}，提升 {resilience_improvement:.1f}%；"
        f"风险降低 {selected_plan['risk_reduction']:.1%}，恢复时间缩短 {selected_plan['recovery_reduction']:.1%}。"
        "建议以系统推荐方案为工程复核起点，并结合现场资料、设计审查和投资测算确认实施路径。</div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "下载决策报告（txt）",
        data=build_report(
            qra_result,
            resilience_result,
            selected_scenario,
            scenario_index,
            selected_plan,
            system_recommended_plan,
            budget_wan,
        ),
        file_name="港口危化品罐区改造辅助决策报告.txt",
        mime="text/plain",
    )
    render_sources((
        "储量、罐容、物料属性和作业条件取自储罐台账、设计文件与现场调查。",
        "风险阈值和事故后果参数应按适用危险化学品标准、QRA 报告与情景分析资料复核。",
        "措施投资、减排与节能参数应以供应商报价、运营台账和工艺设计资料为准。",
    ))

with st.container(key="page_qra"):
    st.header("QRA 风险可视化")
    render_process_flow((
        ("输入", "储量、校正系数、临界量"),
        ("计算", "R 值与事故后果模拟"),
        ("输出", "R 值敏感性、后果柱状图"),
        ("结论", "风险等级、重点泄漏模式"),
    ))
    render_module_description("根据选定储罐的介质储量、厂外人员暴露和危险化学品校正系数计算风险指标 R，并展示储量敏感性及典型泄漏后果半径。")
    render_method("简化 QRA 风险指标", "采用 R = αβ(q/Q) 识别风险等级；后果半径以典型泄漏数据为基准，并按储量与情景修正系数校正。")
    with project_details_container:
        st.markdown(r"**风险公式：** $R = \alpha\beta(q/Q)$")
        st.caption("系统以选定储罐的泄漏事故为分析情景，其中 q 为该储罐的实际储量，Q 为对应介质的临界量。")
        st.markdown(
            "**参数定义**\n\n"
            "- R：风险指标\n"
            "- α：厂外暴露人员校正系数\n"
            "- β：危险化学品校正系数\n"
            "- q：选定储罐的实际储量（吨）\n"
            "- Q：该介质的临界量（吨）"
        )
        st.caption("介质储量输入以万吨计，计算前换算为吨；q 与 Q 使用相同单位。")
    risk_columns = st.columns(2)
    risk_columns[0].metric("当前 R 值", f"{qra_result['risk_value']:.1f}")
    risk_columns[1].metric("风险等级", qra_result['risk_level'])

    st.subheader("R 值敏感性分析")
    fig1, ax1 = plt.subplots(figsize=(8.8, 4.7))
    ax1.plot(qra_result['storage_range'], qra_result['risk_range'], color=NAVY, linewidth=2.4, label='R 值曲线')
    ax1.axhline(100, color=SLATE, linestyle='--', linewidth=1.2, label='一级阈值（R=100）')
    ax1.axhline(50, color=STEEL, linestyle='--', linewidth=1.1, label='二级阈值（R=50）')
    ax1.axhline(10, color=BLUE, linestyle='--', linewidth=1.1, label='三级阈值（R=10）')
    ax1.scatter(q_storage, qra_result['risk_value'], s=85, color=BLUE, edgecolor=WHITE, linewidth=1.2, zorder=5)
    ax1.annotate(f"R={qra_result['risk_value']:.0f}", xy=(q_storage, qra_result['risk_value']),
                 xytext=(0, 10), textcoords='offset points', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=NAVY)
    ax1.set_xlabel('介质储量（万吨）')
    ax1.set_ylabel('R 值')
    ax1.set_title('储量变化下的 R 值敏感性')
    ax1.grid(True, alpha=0.28)
    ax1.margins(y=0.12)
    fig1.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig1.subplots_adjust(top=0.90, bottom=0.24, left=0.10, right=0.97)
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("事故后果模拟")
    positions = np.arange(len(qra_result['modes']))
    width = 0.25
    fig2, ax2 = plt.subplots(figsize=(10.8, 4.8))
    death_bars = ax2.bar(positions - width, qra_result['radii'][0], width, label='死亡半径', color=NAVY)
    severe_bars = ax2.bar(positions, qra_result['radii'][1], width, label='重伤半径', color=STEEL)
    minor_bars = ax2.bar(positions + width, qra_result['radii'][2], width, label='轻伤半径', color=BLUE)
    label_bar_values(ax2, death_bars)
    label_bar_values(ax2, severe_bars)
    label_bar_values(ax2, minor_bars)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(qra_result['modes'], rotation=12, ha='right')
    ax2.set_ylabel('后果半径（m）')
    ax2.set_title('典型泄漏模式的事故后果半径')
    ax2.grid(True, axis='y', alpha=0.28)
    ax2.margins(y=0.18)
    fig2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig2.subplots_adjust(top=0.90, bottom=0.31, left=0.08, right=0.98)
    st.pyplot(fig2)
    plt.close(fig2)
    st.markdown(f"<div class='conclusion-note'><strong>QRA 结论：</strong>当前 R 值为 {qra_result['risk_value']:.1f}，判定为{qra_result['risk_level']}。应优先控制高后果泄漏模式对应的库存、隔离距离与应急资源配置。</div>", unsafe_allow_html=True)
    render_sources((
        "储量、物料类别和临界量来自储罐台账、设计资料及适用标准。",
        "事故后果基准数据来自典型泄漏情景；现场应用时应由专业 QRA 软件和气象条件校核。",
    ))

with st.container(key="page_resilience"):
    st.header("熵权情景韧性评价")
    render_process_flow((
        ("输入", "事故情景、措施、韧性维度"),
        ("计算", "熵权法与恢复曲线"),
        ("输出", "韧性得分、雷达、恢复过程"),
        ("结论", "恢复短板、改造重点"),
    ))
    render_module_description("围绕韧性三角的吸收、适应、恢复能力，在三类常规泄漏和两类非常规冲击下比较改造前后的综合韧性和安全作业能力恢复过程。")
    render_method("熵权法 + 韧性三角理论", "当前按五种情景的改造前后预设三维得分计算熵权，权重由列占比分布的信息熵与差异系数决定；底层指标体系为待接入的候选示例，恢复曲线依据情景损失率与恢复时间生成。")
    selected_scenario = st.selectbox(
        "事故情景选择",
        tuple(SCENARIO_LIBRARY.keys()),
        key="selected_scenario",
    )
    with st.container(border=True):
        st.subheader("非常规冲击说明")
        st.markdown(
            "**蓄意冲击**：指具有明确破坏意图、冲击强度高且可能针对关键设施的非常规事件；"
            "本系统按 1.5 倍冲击强度、约为大泄漏 1.3 倍的恢复时间进行脱敏情景化处理。\n\n"
            "**随机冲击**：指发生位置、时机或影响范围具有较大不确定性的突发事件；"
            "本系统按 1.2 倍冲击强度、约为中泄漏 1.5 倍的恢复时间进行情景化处理。\n\n"
            "与常规泄漏相比，非常规情景更强调意外性、针对性或不确定性，要求系统具备更强的监测预警、隔离切断、资源调度和恢复冗余能力。"
        )
    resilience_columns = st.columns(4)
    resilience_columns[0].metric("当前情景", selected_scenario)
    resilience_columns[1].metric("改造前韧性", f"{resilience_result['before_scores'][scenario_index]:.2f} / 10")
    resilience_columns[2].metric("改造后韧性", f"{resilience_result['after_scores'][scenario_index]:.2f} / 10")
    resilience_columns[3].metric("提升幅度", f"{resilience_improvement:.1f}%")

    render_resilience_logic(resilience_result)

    radar_column, recovery_column = st.columns((1, 1.1))
    with radar_column:
        angles = np.linspace(0, 2 * np.pi, len(resilience_result['dimensions']), endpoint=False).tolist()
        angles += angles[:1]
        fig3, ax3 = plt.subplots(figsize=(6.5, 5.7), subplot_kw=dict(polar=True))
        before_values = resilience_result['before_matrix'][scenario_index].tolist()
        after_values = resilience_result['after_matrix'][scenario_index].tolist()
        ax3.plot(
            angles,
            before_values + [before_values[0]],
            color=SLATE,
            linewidth=1.8,
            linestyle=':',
            label=f'{selected_scenario}（改造前）',
        )
        ax3.fill(angles, before_values + [before_values[0]], color=STEEL, alpha=0.10)
        ax3.plot(
            angles,
            after_values + [after_values[0]],
            color=NAVY,
            linewidth=2.6,
            label=f'{selected_scenario}（改造后）',
        )
        ax3.fill(angles, after_values + [after_values[0]], color=BLUE, alpha=0.16)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(resilience_result['dimensions'])
        ax3.set_ylim(0, 10)
        ax3.set_title(f'{selected_scenario}情景韧性三角', pad=18)
        fig3.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=7)
        fig3.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.92)
        st.pyplot(fig3)
        plt.close(fig3)
    with recovery_column:
        st.subheader(f'{selected_scenario}情景恢复曲线')
        st.caption("关键点标注对应改造后曲线；C 点按恢复时间的 20% 设置为示意恢复起点，并非实测拐点。")
        fig4, ax4 = plt.subplots(figsize=(8.8, 6.8))
        ax4.plot(recovery_days, before_ability, color=STEEL, linewidth=2.0, linestyle=':', label='改造前')
        ax4.plot(recovery_days, after_ability, color=NAVY, linewidth=2.7, label='改造后')
        ax4.axhline(100, color=SLATE, linestyle='--', linewidth=1.1, label='正常作业基线（100%）')
        recovery_duration = max(float(resilience_result['recovery_after'][scenario_index]), 0.1)
        recovery_start = 0.2 * recovery_duration
        minimum_ability = float(after_ability[1])
        key_points = (
            ('A', (0.0, 100.0), (0.02, 1.42), '事故瞬间', '吸收能力：系统承受冲击的起点', '#2E6F69'),
            ('B', (0.0, minimum_ability), (0.02, -0.43), '最低点', '适应能力：系统承压极限', '#B8860B'),
            ('C', (recovery_start, minimum_ability), (0.60, -0.43), '恢复拐点', '适应能力：系统开始恢复', BLUE),
            ('D', (recovery_duration, 100.0), (0.60, 1.42), '恢复至基线', '恢复能力：恢复至正常作业水平', NAVY),
        )
        for point_code, coordinates, label_position, point_name, description, point_color in key_points:
            ax4.scatter(*coordinates, s=65, color=point_color, edgecolor=WHITE, linewidth=1.1, zorder=6)
            ax4.annotate(
                f'{point_code}点：{point_name}\n{description}',
                xy=coordinates,
                xytext=label_position,
                textcoords='axes fraction',
                ha='left',
                va='top' if point_code in ('A', 'D') else 'bottom',
                fontsize=12,
                color=point_color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=WHITE, edgecolor=point_color, alpha=0.96),
                arrowprops=dict(arrowstyle='->', color=point_color, linewidth=1.1),
                annotation_clip=False,
            )
        ax4.set_xlabel('事故后时间（天）')
        ax4.set_ylabel('安全作业能力（%）')
        ax4.set_ylim(0, 108)
        ax4.set_xlim(-0.04 * recovery_days[-1], recovery_days[-1])
        ax4.grid(True, alpha=0.28)
        fig4.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=9)
        fig4.subplots_adjust(top=0.73, bottom=0.35, left=0.12, right=0.97)
        st.pyplot(fig4)
        plt.close(fig4)
        st.caption("恢复曲线标注了四个关键点：A点体现吸收能力（冲击起点），B点和C点体现适应能力（承压极限与恢复拐点），D点体现恢复能力（恢复至基线）。三个维度在一条曲线上完整呈现。")
    st.markdown(
        f"<div class='conclusion-note'><strong>韧性结论：</strong>在{selected_scenario}情景下，所选组合使事故后的安全作业能力损失减少，"
        f"恢复时间由 {resilience_result['recovery_before'][scenario_index]:.1f} 天缩短至 "
        f"{resilience_result['recovery_after'][scenario_index]:.1f} 天。</div>",
        unsafe_allow_html=True,
    )
    render_sources((
        "当前三维得分来自预设脱敏情景参数与固定措施效果系数，并非现场底层指标实测值。",
        "候选底层指标为讨论用示例；接入正式评价前需依据核验文献、现场台账、监测与演练记录确认指标体系和评分规则。",
        "事故情景、恢复时间和安全作业能力曲线应结合演练记录、应急预案和历史处置数据校核。",
        "蓄意冲击与随机冲击的强度系数、恢复时间为非常规事件研究用脱敏情景参数，不对应真实事件或企业。",
    ))

with st.container(key="page_plans"):
    st.header("措施组合推荐")
    render_process_flow((
        ("输入", "预算、措施开关、事故情景"),
        ("计算", "32 种组合、三维综合评分"),
        ("输出", "系统推荐、当前组合、可行排序"),
        ("结论", "推荐组合、投入产出比"),
    ))
    render_module_description("在预算约束下全量枚举五项措施的 32 种组合，展示系统推荐以及由侧栏开关实时控制的当前组合。")
    render_method("全量枚举 + 安全经济环境三维评分", "安全维度占 40%，风险降低和恢复改善各占 20%；经济维度占 30%，投入、风险收益和回收期各占 10%；环境维度占 30%，能耗和污染排放指数各占 15%。")

    st.subheader("系统推荐组合")
    st.markdown(f"**推荐措施：** {system_recommended_plan['plan_name']}")
    recommendation_metrics = st.columns(5)
    recommendation_metrics[0].metric("投资", f"{system_recommended_plan['investment']:.0f} 万元")
    recommendation_metrics[1].metric("三维综合得分", f"{system_recommended_plan['score']:.1f} / 100")
    recommendation_metrics[2].metric("安全维度得分", f"{system_recommended_plan['safety_score']:.1f} / 100")
    recommendation_metrics[3].metric("经济维度得分", f"{system_recommended_plan['economic_score']:.1f} / 100")
    recommendation_metrics[4].metric("环境维度得分", f"{system_recommended_plan['environmental_score']:.1f} / 100")
    recommendation_metrics = st.columns(4)
    recommendation_metrics[0].metric("风险降低", f"{system_recommended_plan['risk_reduction']:.1%}")
    recommendation_metrics[1].metric("恢复改善", f"{system_recommended_plan['recovery_reduction']:.1%}")
    recommendation_metrics[2].metric("减排", f"{system_recommended_plan['emission_reduction']:.1%}")
    recommendation_metrics[3].metric("节能", f"{system_recommended_plan['energy_reduction']:.1%}")

    st.subheader("当前手动组合")
    st.markdown(f"**当前措施：** {selected_plan['plan_name']}")
    if selected_plan['investment'] > budget_wan:
        st.error(
            f"当前组合投资 {selected_plan['investment']:.0f} 万元，超出预算 "
            f"{budget_wan:.0f} 万元；该组合不纳入可行组合排序。"
        )
    elif selected_plan['measures'] == system_recommended_plan['measures']:
        st.success("当前组合即系统推荐")
    if selected_plan_rank is not None:
        st.caption(f"当前组合在可行组合中排名第 {selected_plan_rank}。")
    elif selected_plan['investment'] > budget_wan:
        st.caption("当前组合暂无可行排名，请降低投资或提高预算。")

    current_metrics = st.columns(5)
    current_metrics[0].metric("投资", f"{selected_plan['investment']:.0f} 万元")
    current_metrics[1].metric("三维综合得分", f"{selected_plan['score']:.1f} / 100")
    current_metrics[2].metric("安全维度得分", f"{selected_plan['safety_score']:.1f} / 100")
    current_metrics[3].metric("经济维度得分", f"{selected_plan['economic_score']:.1f} / 100")
    current_metrics[4].metric("环境维度得分", f"{selected_plan['environmental_score']:.1f} / 100")
    current_metrics = st.columns(5)
    current_metrics[0].metric("风险降低", f"{selected_plan['risk_reduction']:.1%}")
    current_metrics[1].metric("恢复改善", f"{selected_plan['recovery_reduction']:.1%}")
    current_metrics[2].metric("减排", f"{selected_plan['emission_reduction']:.1%}")
    current_metrics[3].metric("节能", f"{selected_plan['energy_reduction']:.1%}")
    current_metrics[4].metric("投入产出比", f"{selected_economics['roi']:.1%}")

    measure_frame = pd.DataFrame([
        {
            '措施': measure['name'],
            '风险降低': f"{measure['risk']:.0%}",
            '恢复改善': f"{measure['recovery']:.0%}",
            '减排': f"{measure['emission']:.0%}",
            '节能': f"{measure['energy']:.0%}",
            '投资（万元）': measure['investment'],
        }
        for measure in MEASURE_LIBRARY
    ])
    st.subheader("单项措施库")
    st.dataframe(measure_frame, hide_index=True)

    measures = [measure['name'] for measure in MEASURE_LIBRARY]
    measure_scores = [
        calculate_plan_scores(calculate_plan_effect((measure['name'],)))
        for measure in MEASURE_LIBRARY
    ]
    measure_score_positions = np.arange(len(measures))
    measure_score_figure, measure_score_axis = plt.subplots(figsize=(10.8, 4.8))
    measure_score_width = 0.24
    safety_bars = measure_score_axis.bar(
        measure_score_positions - measure_score_width,
        [score['safety_score'] for score in measure_scores],
        measure_score_width,
        label='安全维度得分',
        color=NAVY,
    )
    economic_bars = measure_score_axis.bar(
        measure_score_positions,
        [score['economic_score'] for score in measure_scores],
        measure_score_width,
        label='经济维度得分',
        color=BLUE,
    )
    environmental_bars = measure_score_axis.bar(
        measure_score_positions + measure_score_width,
        [score['environmental_score'] for score in measure_scores],
        measure_score_width,
        label='环境维度得分',
        color=STEEL,
    )
    for bars in (safety_bars, economic_bars, environmental_bars):
        label_bar_values(measure_score_axis, bars, fmt='%.1f', fontsize=8)
    measure_score_axis.set_xticks(measure_score_positions)
    measure_score_axis.set_xticklabels(measures)
    measure_score_axis.set_ylabel('得分（0-100）')
    measure_score_axis.set_title('单项措施的安全、经济与环境维度得分对比')
    measure_score_axis.set_ylim(0, 105)
    measure_score_axis.grid(True, axis='y', alpha=0.28)
    measure_score_figure.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    measure_score_figure.subplots_adjust(top=0.89, bottom=0.23, left=0.08, right=0.98)
    st.pyplot(measure_score_figure)
    plt.close(measure_score_figure)

    positions = np.arange(len(measures))
    fig6, ax6 = plt.subplots(figsize=(11.2, 5.0))
    width = 0.18
    risk_bars = ax6.bar(positions - 1.5 * width, [measure['risk'] * 100 for measure in MEASURE_LIBRARY], width, label='风险降低', color=NAVY)
    recovery_bars = ax6.bar(positions - 0.5 * width, [measure['recovery'] * 100 for measure in MEASURE_LIBRARY], width, label='恢复改善', color=BLUE)
    emission_bars = ax6.bar(positions + 0.5 * width, [measure['emission'] * 100 for measure in MEASURE_LIBRARY], width, label='污染物减排', color=STEEL)
    energy_bars = ax6.bar(positions + 1.5 * width, [measure['energy'] * 100 for measure in MEASURE_LIBRARY], width, label='能耗降低', color=SLATE)
    for bars in (risk_bars, recovery_bars, emission_bars, energy_bars):
        label_bar_values(ax6, bars, fmt='%.0f', fontsize=7)
    ax6.set_xticks(positions)
    ax6.set_xticklabels(measures)
    ax6.set_ylabel('改善幅度（%）')
    ax6.set_title('单项措施的风险、恢复、减排与节能效益')
    ax6.grid(True, axis='y', alpha=0.28)
    ax6.margins(y=0.22)
    fig6.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, columnspacing=1.4)
    fig6.subplots_adjust(top=0.89, bottom=0.30, left=0.08, right=0.98)
    st.pyplot(fig6)
    plt.close(fig6)

    fig7, ax7 = plt.subplots(figsize=(10.0, 4.5))
    investment_bars = ax7.bar(measures, [measure['investment'] for measure in MEASURE_LIBRARY], color=NAVY)
    label_bar_values(ax7, investment_bars)
    ax7.set_ylabel('投资（万元）')
    ax7.set_title('单项措施投资对比')
    ax7.grid(True, axis='y', alpha=0.28)
    ax7.margins(y=0.16)
    fig7.subplots_adjust(top=0.89, bottom=0.16, left=0.09, right=0.98)
    st.pyplot(fig7)
    plt.close(fig7)

    st.subheader("可行组合排序")
    st.caption("仅展示预算内组合，并按三维综合得分从高到低排序；第一行即当前预算下的最优方案。星标表示系统推荐，圆点表示当前组合。")

    def build_ranking_rows(plans, include_rank):
        rows = []
        for rank, plan in enumerate(plans, start=1):
            markers = []
            if plan['measures'] == system_recommended_plan['measures']:
                markers.append('★ 系统推荐')
            if plan['measures'] == selected_plan['measures']:
                markers.append('● 当前组合')
            plan_economics = calculate_economic_metrics(plan)
            row = {
                '标记': ' / '.join(markers),
                '措施组合': plan['plan_name'],
                '投资（万元）': round(plan['investment'], 1),
                '风险降低': f"{plan['risk_reduction']:.1%}",
                '恢复改善': f"{plan['recovery_reduction']:.1%}",
                '减排': f"{plan['emission_reduction']:.1%}",
                '节能': f"{plan['energy_reduction']:.1%}",
                '投入产出比': f"{plan_economics['roi']:.1%}",
                '安全维度得分': round(plan['safety_score'], 1),
                '经济维度得分': round(plan['economic_score'], 1),
                '环境维度得分': round(plan['environmental_score'], 1),
                '三维综合得分': round(plan['score'], 1),
            }
            if include_rank:
                row = {'排名': rank, **row}
            rows.append(row)
        return pd.DataFrame(rows)

    budget_plans = sorted(feasible_plans, key=lambda plan: plan['score'], reverse=True)
    ranking_frame = build_ranking_rows(budget_plans, include_rank=True)

    def highlight_feasible_plan(row):
        if '当前组合' in row['标记']:
            color = 'background-color: #DCEEFF; color: #163B65;'
        elif '系统推荐' in row['标记']:
            color = 'background-color: #FCE4E4; color: #9B2525;'
        else:
            color = ''
        return [color] * len(row)

    st.dataframe(
        ranking_frame.style.apply(highlight_feasible_plan, axis=1),
        hide_index=True,
        width="stretch",
    )

    over_budget_plans = sorted(
        (plan for plan in all_plan_options if not plan['within_budget']),
        key=lambda plan: plan['score'],
        reverse=True,
    )
    if over_budget_plans:
        with st.expander(f"超预算组合（{len(over_budget_plans)} 项）", expanded=False):
            st.caption("以下组合不参与系统推荐或主表排名，仅用于预算调整时的备选核对。")
            over_budget_frame = build_ranking_rows(over_budget_plans, include_rank=False)
            st.dataframe(
                over_budget_frame.style.apply(
                    lambda row: ['background-color: #EDF0F2; color: #7A8793;'] * len(row),
                    axis=1,
                ),
                hide_index=True,
                width="stretch",
            )

    before_after_frame = pd.DataFrame({
        '指标': ['R 值', '恢复时间', '污染物排放指数', '能耗指数', '投资'],
        '改造前': [f"{qra_result['risk_value']:.1f}", f"{resilience_result['recovery_before'][scenario_index]:.1f} 天", '100', '100', '0 万元'],
        '所选组合后': [
            f"{qra_result['risk_value'] * (1 - selected_plan['risk_reduction']):.1f}",
            f"{resilience_result['recovery_after'][scenario_index]:.1f} 天",
            f"{100 * (1 - selected_plan['emission_reduction']):.1f}",
            f"{100 * (1 - selected_plan['energy_reduction']):.1f}",
            f"{selected_plan['investment']:.0f} 万元",
        ],
    })
    st.subheader("改造前后对比")
    st.dataframe(before_after_frame, hide_index=True)
    if selected_plan_rank is None:
        conclusion = (
            f"当前组合“{selected_plan['plan_name']}”超出预算；预算内系统推荐为"
            f"“{system_recommended_plan['plan_name']}”。"
        )
    elif selected_plan['measures'] == system_recommended_plan['measures']:
        conclusion = f"当前组合即系统推荐，排名第 {selected_plan_rank}。"
    else:
        conclusion = (
            f"当前组合排名第 {selected_plan_rank}；预算内系统推荐为"
            f"“{system_recommended_plan['plan_name']}”。"
        )
    st.markdown(
        f"<div class='conclusion-note'><strong>方案结论：</strong>{conclusion}</div>",
        unsafe_allow_html=True,
    )
    render_sources((
        "措施技术效果为课程研究用参数，实际取值应由工艺设计、环境影响评价和供应商技术资料复核。",
        "投资额应依据设备报价、施工组织、停产损失和生命周期成本进行详细测算。",
    ))

with st.container(key="page_strategy"):
    st.header("策略对照")
    render_module_description("在相同预算上限下，分别按风险收益优先、投资成本优先和安全经济环境三维综合优选生成措施组合。")
    render_method("预算内策略生成 + 三维评分", "风险优先与低成本策略按固定顺序依次纳入可负担措施；综合优选在全部预算可行组合中按安全 40%、经济 30%、环境 30% 进行评分。")
    st.metric("当前预算上限", f"{budget_wan:.0f} 万元")

    strategy_frame = pd.DataFrame([
        {
            '策略': name,
            '措施组合': plan['plan_name'],
            '投资（万元）': plan['investment'],
            '风险降低': f"{plan['risk_reduction']:.1%}",
            '恢复改善': f"{plan['recovery_reduction']:.1%}",
            '减排': f"{plan['emission_reduction']:.1%}",
            '节能': f"{plan['energy_reduction']:.1%}",
            '安全维度得分': round(plan['safety_score'], 1),
            '经济维度得分': round(plan['economic_score'], 1),
            '环境维度得分': round(plan['environmental_score'], 1),
            '三维综合得分': round(score, 1),
        }
        for (name, plan), score in zip(strategy_plans, strategy_scores)
    ])
    strategy_figure, strategy_axis = plt.subplots(figsize=(10.0, 4.6))
    strategy_names = [name for name, _ in strategy_plans]
    strategy_bars = strategy_axis.bar(
        strategy_names,
        strategy_scores,
        color=[BLUE, STEEL, NAVY],
        width=0.58,
    )
    label_bar_values(strategy_axis, strategy_bars, fmt='%.1f', fontsize=9)
    strategy_axis.set_ylabel('三维综合得分（0-100）')
    strategy_axis.set_ylim(0, max(100, max(strategy_scores) * 1.18))
    strategy_axis.set_title('相同预算下的策略组合三维综合得分')
    strategy_axis.grid(True, axis='y', alpha=0.28)
    strategy_figure.subplots_adjust(top=0.88, bottom=0.18, left=0.09, right=0.98)
    st.pyplot(strategy_figure, width="stretch")
    plt.close(strategy_figure)
    st.dataframe(strategy_frame, hide_index=True, width="stretch")
    st.success(
        f"综合优选在全部预算可行组合中得分最高：{comprehensive_strategy['plan_name']}，"
        f"投资 {comprehensive_strategy['investment']:.0f} 万元，"
        f"三维综合得分 {comprehensive_strategy['score']:.1f}。"
    )
    st.subheader("预算敏感性分析")
    st.caption("横轴为预算上限 50-500 万元，纵轴为三种策略在对应预算下的三维综合得分。")
    budget_sensitivity = calculate_strategy_budget_sensitivity(SCORING_SCHEMA_VERSION)
    sensitivity_figure, sensitivity_axis = plt.subplots(figsize=(10.0, 4.8))
    strategy_colors = {
        '按风险高低依次改造': BLUE,
        '优先低成本': STEEL,
        '综合优选': NAVY,
    }
    for strategy_name in ('按风险高低依次改造', '优先低成本', '综合优选'):
        strategy_series = budget_sensitivity[budget_sensitivity['策略'] == strategy_name]
        sensitivity_axis.plot(
            strategy_series['预算（万元）'],
            strategy_series['三维综合得分'],
            color=strategy_colors[strategy_name],
            linewidth=2.6 if strategy_name == '综合优选' else 1.8,
            label=strategy_name,
        )
    sensitivity_axis.set_xlabel('预算（万元）')
    sensitivity_axis.set_ylabel('三维综合得分（0-100）')
    sensitivity_axis.set_title('不同预算下的策略三维综合得分')
    sensitivity_axis.grid(True, alpha=0.28)
    sensitivity_axis.legend(frameon=False, ncol=3, loc='lower center')
    sensitivity_figure.subplots_adjust(top=0.88, bottom=0.18, left=0.09, right=0.98)
    st.pyplot(sensitivity_figure, width="stretch")
    plt.close(sensitivity_figure)
    score_matrix = budget_sensitivity.pivot(
        index='预算（万元）',
        columns='策略',
        values='三维综合得分',
    )
    comprehensive_always_best = np.allclose(
        score_matrix['综合优选'].to_numpy(),
        score_matrix.max(axis=1).to_numpy(),
    )
    if comprehensive_always_best:
        st.success("在 50-500 万元的全部预算档位中，综合优选均为得分最高或并列最高的策略。")
    else:
        st.warning("综合优选并非在所有预算档位均为最高分；请结合折线图识别需复核的预算区间。")
    render_sources((
        "策略对照为同一措施库与同一预算下的算法情景比较，不代表工程实施顺序建议。",
        "投入产出比基于脱敏年风险收益和投资额计算，实际决策需核对现场数据、报价和收益测算。",
    ))

with st.container(key="page_economy"):
    st.header("投入产出比分析")
    render_module_description("根据当前手动组合和预算，动态估算改造投入、风险降低收益、投入产出比和静态投资回收期。")
    render_method("投入—风险—收益—回收期动态传导", "当前投资取手动组合总投资；年风险收益按风险降低幅度 × 40 万元脱敏基数计算；投入产出比为年风险收益 ÷ 当前投资，回收期为当前投资 ÷ 年风险收益。")

    safety_investment = selected_economics['investment']
    risk_reduction = float(selected_plan['risk_reduction'])
    annual_risk_benefit = selected_economics['annual_risk_benefit']
    payback_years = selected_economics['payback_years']
    if risk_reduction < 0.20:
        risk_level_change = "一级（未降级）"
    elif risk_reduction <= 0.40:
        risk_level_change = "一级 → 二级"
    else:
        risk_level_change = "一级 → 三级"

    if safety_investment > budget_wan:
        st.warning(
            f"当前组合投资约 {safety_investment:.1f} 万元，超过当前预算上限 "
            f"{budget_wan:.0f} 万元；以下经济测算仍按当前勾选组合展示。"
        )
    st.caption(
        f"当前事故情景：{selected_scenario}；当前措施：{selected_plan['plan_name']}；"
        f"预算上限：{budget_wan:.0f} 万元。"
    )

    economics_columns = st.columns(4)
    economics_columns[0].metric("当前投资", f"{safety_investment:.1f} 万元")
    economics_columns[1].metric("年风险收益", f"约 {annual_risk_benefit:.1f} 万元")
    economics_columns[2].metric("投入产出比", f"{selected_economics['roi']:.1%}")
    economics_columns[3].metric(
        "安全投资回收期",
        f"约 {payback_years:.1f} 年" if payback_years is not None else "不可回收",
    )

    flow_columns = st.columns((1, 0.22, 1, 0.22, 1, 0.22, 1))
    flow_nodes = (
        (0, "安全整改投入", f"{safety_investment:.1f} 万元"),
        (2, "风险等级改善", risk_level_change),
        (4, "年风险降低收益", f"{annual_risk_benefit:.1f} 万元"),
        (6, "投资回收期", f"{payback_years:.1f} 年" if payback_years is not None else "不可回收"),
    )
    for column_index, title, value in flow_nodes:
        with flow_columns[column_index]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.subheader(value)
                if title == "年风险降低收益":
                    st.caption("保险费用降低 + 预期损失减少 + 合规成本降低")
    for arrow_index in (1, 3, 5):
        with flow_columns[arrow_index]:
            st.markdown("### →")

    st.subheader("年风险降低收益构成")
    benefit_frame = pd.DataFrame([
        {'收益路径': '保险费用降低', '占比': '75.0%', '年度收益（万元）': round(annual_risk_benefit * 0.75, 2)},
        {'收益路径': '预期损失减少', '占比': '13.5%', '年度收益（万元）': round(annual_risk_benefit * 0.135, 2)},
        {'收益路径': '合规成本降低', '占比': '11.5%', '年度收益（万元）': round(annual_risk_benefit * 0.115, 2)},
        {'收益路径': '合计', '占比': '100.0%', '年度收益（万元）': round(annual_risk_benefit, 2)},
    ])
    st.dataframe(benefit_frame, hide_index=True, width="stretch")

    st.subheader("投资回收期敏感性分析")
    if annual_risk_benefit > 0:
        benefit_range = np.linspace(annual_risk_benefit * 0.5, annual_risk_benefit * 1.5, 80)
        payback_range = safety_investment / benefit_range
    else:
        benefit_range = np.linspace(0.1, 1.0, 80)
        payback_range = np.full_like(benefit_range, np.nan)
    sensitivity_figure, sensitivity_axis = plt.subplots(figsize=(10.0, 4.8))
    sensitivity_axis.plot(benefit_range, payback_range, color=NAVY, linewidth=2.4, label='投资回收期')
    sensitivity_axis.axhline(5, color=STEEL, linestyle='--', linewidth=1.2, label='5年参考线')
    if payback_years is not None:
        sensitivity_axis.scatter([annual_risk_benefit], [payback_years], color='#B8860B', s=80, zorder=5, label='当前方案')
        sensitivity_axis.annotate(
            f"当前点：{annual_risk_benefit:.1f} 万元 / {payback_years:.1f} 年",
            xy=(annual_risk_benefit, payback_years), xytext=(8, 8),
            textcoords='offset points', color=NAVY, fontsize=9,
        )
    sensitivity_axis.set_xlabel('年风险降低收益（万元）')
    sensitivity_axis.set_ylabel('投资回收期（年）')
    sensitivity_axis.set_title('年收益变化下的投资回收期敏感性')
    sensitivity_axis.grid(True, alpha=0.28)
    sensitivity_axis.legend(frameon=False)
    sensitivity_figure.subplots_adjust(top=0.88, bottom=0.16, left=0.10, right=0.97)
    st.pyplot(sensitivity_figure)
    plt.close(sensitivity_figure)
    st.caption("收益区间为当前年收益的 50%-150%；回收期公式为当前投入 ÷ 年风险降低收益。以上为脱敏估算，不构成财务承诺。")
    render_sources((
        "当前投资由所选五项改造措施的脱敏投资额叠加形成。",
        "年风险收益、投入产出比和回收期为课程展示用参数化估算，不构成财务承诺。",
    ))

with st.container(key="page_sources"):
    st.header("数据来源说明")
    render_module_description("汇总系统中主要参数的来源类别和使用边界；展示数据已经过脱敏、取整或量级模糊处理。")
    render_method("数据分类与适用边界说明", "系统按风险、气象、措施成本和公开因子分类引用数据，并以脱敏情景参数进行研究性展示。")
    source_frame = pd.DataFrame([
        {'参数类别': 'QRA 数据', '主要内容': '储量、校正系数、临界量、事故后果半径', '来源说明': '某港口配套罐区安全验收评价报告'},
        {'参数类别': '碳排因子', '主要内容': '污染物/碳排放核算系数', '来源说明': '国家发展和改革委员会公开数据'},
        {'参数类别': '设备价格', '主要内容': '改造设备与配套设施价格区间', '来源说明': '行业公开数据'},
        {'参数类别': '气象数据', '主要内容': '风向、风速、温度、大气稳定度', '来源说明': '某港口配套罐区安全验收评价报告'},
        {'参数类别': '措施成本', '主要内容': '密封、监测、切断、回收、节能措施投资', '来源说明': '行业经验估算'},
    ])
    st.dataframe(source_frame, hide_index=True, width="stretch")
    st.info("当前工作区展示的是脱敏情景参数；具体报告版本、设备报价与适用系数应在正式项目中依据原始资料复核。")
    render_sources((
        "储量、设备与作业条件应以项目台账、设计资料和现场调查为准。",
        "公开因子、行业价格和气象统计仅用于情景化展示，正式应用需经过数据授权与专业复核。",
    ))

with st.container(key="page_weather"):
    st.header("气象条件")
    render_process_flow((
        ("输入", "风向、风速、大气稳定度"),
        ("计算", "风玫瑰统计、稳定度分类"),
        ("输出", "风玫瑰、饼图、关键指标"),
        ("结论", "主导风向、下风向暴露区"),
    ))
    render_module_description("展示某港口配套区的脱敏风向风速频率、大气稳定度和代表性气象地貌参数，为 QRA 事故后果与扩散情景提供背景条件。")
    render_method("风向风速频率统计 + 大气稳定度分类", "风玫瑰图按 16 方位和三类风速区间展示情景化频率；稳定度按 C、D、E、F 类的年内占比表达。")
    weather = calculate_weather_conditions()
    weather_metrics = st.columns(4)
    weather_metrics[0].metric("常风向", "ESE-SE")
    weather_metrics[1].metric("强风向", "NNE-NE")
    weather_metrics[2].metric("年平均风速", weather['mean_wind_speed'])
    weather_metrics[3].metric("年平均气温", weather['mean_temperature'])
    st.caption("常风向以 ESE-SE 为主，强风向集中在 NNE-NE；风玫瑰频率为脱敏情景化示意。")
    st.markdown(f"<div class='conclusion-note'><strong>周边地貌：</strong>{weather['landform']}</div>", unsafe_allow_html=True)

    wind_column, stability_column = st.columns((1.25, 0.75))
    with wind_column:
        st.subheader("风向风速频率")
        angles = np.linspace(0, 2 * np.pi, len(weather['directions']), endpoint=False)
        fig8, ax8 = plt.subplots(figsize=(7.2, 6.0), subplot_kw=dict(polar=True))
        ax8.set_theta_zero_location('N')
        ax8.set_theta_direction(-1)
        bottom = np.zeros(len(weather['directions']))
        for values, label, color in zip(weather['speed_groups'], ('低风速', '中风速', '较强风速'), (STEEL, BLUE, NAVY)):
            ax8.bar(angles, values, width=2 * np.pi / len(weather['directions']) * 0.82,
                    bottom=bottom, label=label, color=color, edgecolor=WHITE, linewidth=0.5)
            bottom += values
        ax8.set_xticks(angles)
        ax8.set_xticklabels(weather['directions'], fontsize=8)
        ax8.set_title('脱敏风玫瑰图（情景化频率）', pad=20)
        fig8.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=8)
        fig8.subplots_adjust(top=0.86, bottom=0.16, left=0.06, right=0.94)
        st.pyplot(fig8)
        plt.close(fig8)
    with stability_column:
        st.subheader("大气稳定度分布")
        stability_labels = list(weather['stability'])
        stability_values = list(weather['stability'].values())
        fig9, ax9 = plt.subplots(figsize=(5.2, 5.3))
        ax9.pie(stability_values, labels=stability_labels, autopct='%1.0f%%', startangle=90,
                colors=(NAVY, BLUE, STEEL, SLATE), wedgeprops=dict(edgecolor=WHITE, linewidth=1.2),
                textprops=dict(fontsize=10))
        ax9.set_title('大气稳定度年内占比')
        fig9.subplots_adjust(top=0.87, bottom=0.05, left=0.05, right=0.95)
        st.pyplot(fig9)
        plt.close(fig9)

    weather_table = pd.DataFrame({
        '气象要素': ['常风向', '强风向', '年平均风速', '年平均气温', '周边地貌'],
        '脱敏情景值': ['ESE-SE', 'NNE-NE', weather['mean_wind_speed'], weather['mean_temperature'], weather['landform']],
    })
    st.dataframe(weather_table, hide_index=True)
    st.markdown("<div class='conclusion-note'><strong>气象结论：</strong>常风向集中于 ESE-SE，强风向集中于 NNE-NE；大气稳定度以 D 类为主。建议在事故扩散情景中重点关注下风向暴露区和近地层稳定条件。</div>", unsafe_allow_html=True)
    render_sources((
        "风向风速与稳定度为课程展示用脱敏统计参数，采用量级模糊处理。",
        "周边地貌为泛化描述，不对应任何真实地理位置或企业信息。",
    ))

with st.container(key="page_model"):
    st.header("港口配套区完整沙盘（已脱敏）")
    render_process_flow((
        ("输入", "风向角度、风速、事故情景"),
        ("计算", "下风向、影响范围计算"),
        ("输出", "三维模型、流线、球壳、热力图"),
        ("结论", "影响范围、下风向扩散方向"),
    ))
    st.markdown("按总平面相对关系展示罐区、北侧水域及码头、道路、周边设施与事故影响范围；编号、尺寸和容量均已脱敏。")
    render_module_description("以通用编号 V-01 至 V-06 和量级模糊后的设施尺度构建港口配套区完整沙盘，周边水域、道路、既有罐区和企业均为泛化对象，不对应真实企业或地理位置。")
    render_method("交互式三维总平面示意", "使用 Plotly 图形对象构建储罐、建筑、道路、水域、码头、管道及事故影响范围；模型仅用于教学展示与方案讨论，不替代工程总平面图。")

    model_scenario = selected_scenario
    st.caption(f"当前全局事故情景：{model_scenario}（由“熵权情景韧性”模块中的事故情景选择联动）")
    wind_columns = st.columns(2)
    wind_direction_deg = wind_columns[0].slider(
        "风向角度（度）", 0, 360, 135, key="model_wind_direction"
    )
    wind_speed = wind_columns[1].number_input(
        "风速（m/s）", value=4.7, min_value=0.1, max_value=30.0, step=0.1,
        key="model_wind_speed",
    )
    model_controls = st.columns(4)
    show_dispersion_lines = model_controls[0].toggle("显示泄漏扩散线", value=True, key="show_dispersion_lines")
    show_impact_shells = model_controls[1].toggle("显示事故影响球壳", value=True, key="show_impact_shells")
    show_risk_heatmap = model_controls[2].toggle("显示风险热力图", value=False, key="show_risk_heatmap")
    enable_dispersion_animation = model_controls[3].toggle("启用扩散动画", value=False, key="enable_dispersion_animation")
    remediation_stage = st.segmented_control(
        "影响范围阶段",
        ("整改前", "整改后"),
        default="整改前",
        key="model_remediation_stage",
    ) or "整改前"
    downwind_angle, _ = calculate_downwind_vector(wind_direction_deg)
    scenario_definition = SCENARIO_LIBRARY[model_scenario]
    dispersion_length = scenario_definition['dispersion_length']
    nominal_radii = tuple(
        int(round(float(radius) * scenario_definition['impact_scale'] / 10) * 10)
        for radius in qra_result['radii'][:, 0]
    )
    remediation_factor = 1.0 if remediation_stage == "整改前" else max(0.25, 1 - selected_plan['risk_reduction'])
    model_radii = tuple(max(10, int(round(radius * remediation_factor / 10) * 10)) for radius in nominal_radii)
    model_dispersion_length = int(round(dispersion_length * remediation_factor))
    impact_columns = st.columns(4)
    impact_columns[0].metric("死亡影响范围", f"约{model_radii[0]} m")
    impact_columns[1].metric("重伤影响范围", f"约{model_radii[1]} m")
    impact_columns[2].metric("轻伤影响范围", f"约{model_radii[2]} m")
    impact_columns[3].metric("下风向扩散长度", f"约{model_dispersion_length} m")
    st.caption(
        f"当前为{remediation_stage}情景；输入风向 {wind_direction_deg}°、风速 {wind_speed:.1f} m/s；"
        f"下风向为 {downwind_angle:.0f}°。泄漏扩散主线为深红，副线为浅红；"
        "三条灰色工艺主干线依次表示码头至罐区、罐区至装车台、罐区至油污水收集池。"
    )
    if model_scenario == '随机冲击':
        st.info("随机冲击采用 V-03 作为脱敏代表受影响点，用于稳定展示不确定影响范围；它不是每次运行都随机改变事故位置的真实事件仿真。")

    if "model_selected_tank" not in st.session_state:
        st.session_state["model_selected_tank"] = scenario_definition['incident_tank']
    if "model_view_mode" not in st.session_state:
        st.session_state["model_view_mode"] = "重置视角"
    tank_options = [tank['code'] for tank in MODEL_TANKS]
    scenario_default_tank = scenario_definition['incident_tank']
    if st.session_state.get("model_incident_scenario") != model_scenario:
        st.session_state["model_incident_tank"] = scenario_default_tank
        st.session_state["model_incident_scenario"] = model_scenario
    incident_tank_code = st.selectbox(
        "事故储罐（可调整）",
        tank_options,
        key="model_incident_tank",
    )
    st.caption(f"当前情景默认事故储罐：{scenario_default_tank}；当前实际事故储罐：{incident_tank_code}。切换全局情景后将自动恢复该情景默认值。")
    if "model_detail_tank_pending" in st.session_state:
        st.session_state["model_detail_tank"] = st.session_state.pop("model_detail_tank_pending")
    if "model_detail_tank" not in st.session_state:
        st.session_state["model_detail_tank"] = st.session_state["model_selected_tank"]
    detail_selection = st.selectbox(
        "选择储罐查看详情",
        tank_options,
        key="model_detail_tank",
    )
    if detail_selection != st.session_state["model_selected_tank"]:
        st.session_state["model_selected_tank"] = detail_selection
        st.rerun()
    view_columns = st.columns((1, 1, 1, 3))
    if view_columns[0].button("重置视角", use_container_width=True, key="model_reset_view"):
        st.session_state["model_view_mode"] = "重置视角"
    if view_columns[1].button("俯视视角", use_container_width=True, key="model_top_view"):
        st.session_state["model_view_mode"] = "俯视视角"
    if view_columns[2].button("侧视视角", use_container_width=True, key="model_side_view"):
        st.session_state["model_view_mode"] = "侧视视角"
    selected_model_tank = st.session_state["model_selected_tank"]
    model_chart_placeholder = st.empty()

    def render_model_chart(animation_phase=0.0):
        model_figure = build_tank_model(
            selected_model_tank,
            model_scenario,
            incident_tank_code,
            model_radii,
            st.session_state["model_view_mode"],
            wind_direction_deg,
            wind_speed,
            show_dispersion_lines,
            show_impact_shells,
            show_risk_heatmap,
            animation_phase,
            remediation_factor,
        )
        model_chart_placeholder.plotly_chart(
            model_figure,
            width="stretch",
            key="anonymized_tank_model",
            config={"displaylogo": False, "scrollZoom": True},
        )

    if st.session_state["active_page"] == "三维罐区模型":
        if enable_dispersion_animation:
            st.caption("扩散动画已启用，流线按短周期持续更新。")

            @st.fragment(run_every="1.5s")
            def render_animated_model():
                phase = round((float(st.session_state.get("model_animation_phase", 0.0)) + 0.10) % 1.0, 2)
                st.session_state["model_animation_phase"] = phase
                render_model_chart(phase)

            render_animated_model()
        else:
            render_model_chart()
        st.caption("通过上方“选择储罐查看详情”下拉框查看单个储罐参数。")

    st.subheader("当前措施组合的空间落地与预期成效")
    effect_columns = st.columns(4)
    effect_columns[0].metric("当前投资", f"{selected_plan['investment']:.0f} 万元")
    effect_columns[1].metric("风险降低", f"{selected_plan['risk_reduction']:.1%}")
    effect_columns[2].metric("恢复改善", f"{selected_plan['recovery_reduction']:.1%}")
    effect_columns[3].metric("年风险降低收益", f"约 {selected_economics['annual_risk_benefit']:.1f} 万元")
    if selected_plan['measures']:
        spatial_measure_frame = pd.DataFrame([
            {
                '当前措施': measure_name,
                '三维实施位置': MEASURE_SPATIAL_INFO[measure_name]['location'],
                '预期作用': MEASURE_SPATIAL_INFO[measure_name]['mechanism'],
            }
            for measure_name in selected_plan['measures']
        ])
        st.dataframe(spatial_measure_frame, hide_index=True, width="stretch")
    else:
        st.info("当前未勾选改造措施；可在左侧选择措施后查看其在三维模型中的实施落点与预期作用。")

    selected_tank_detail = next(
        (tank for tank in MODEL_TANKS if tank['code'] == selected_model_tank),
        None,
    )
    if selected_tank_detail:
        with st.container(border=True):
            st.subheader(f"储罐详情 · {selected_tank_detail['code']}")
            detail_columns = st.columns(4)
            detail_columns[0].metric("介质", selected_tank_detail['medium'])
            detail_columns[1].metric("容量", selected_tank_detail['capacity'])
            detail_columns[2].metric("直径", selected_tank_detail['diameter'])
            detail_columns[3].metric("高度", selected_tank_detail['height_label'])

    st.caption(
        f"当前选中查看储罐：{selected_model_tank}；当前事故储罐：{incident_tank_code}。"
        f"{model_scenario}情景在{remediation_stage}时的气体流线延伸约 {model_dispersion_length} m；"
        "球壳和热力图均为可切换的脱敏影响范围示意，所有长度为量级模糊值；"
        "所有对象名称、容量和尺寸均为脱敏示意。"
    )
    render_sources((
        "储罐编号为通用代号；介质类别、容量、尺寸及设施尺度均经过概略化处理。",
        "西侧原有罐区、东侧企业、装车台、动力辅房、码头设施等均为泛化对象，不对应真实企业或地理位置。",
        "影响半径来自本系统 QRA 参数化结果，并以约数展示；实际项目应以专项 QRA 结论为准。",
    ))

with st.container(key="page_architecture"):
    st.header("系统架构图")
    render_module_description("展示从情景参数输入、风险与韧性计算、措施优化到综合报告输出的系统决策链。")
    render_method("参数化计算 + 方案比选", "各模块共享同一组脱敏情景、预算与措施选择，保证风险、韧性、措施和报告指标联动一致。")
    st.graphviz_chart(
        """
        digraph decision_system {
            graph [rankdir=LR, bgcolor="transparent", nodesep=0.48, ranksep=0.8];
            node [shape=box, style="rounded,filled", color="#163B65", fillcolor="#F4F7FA", fontname="Microsoft YaHei", fontcolor="#163B65", margin="0.18,0.12"];
            edge [color="#4C78A8", penwidth=1.5, arrowsize=0.8];
            input [label="脱敏情景参数\n储量、气象、预算、措施"];
            qra [label="QRA 风险识别\n风险等级与影响范围"];
            resilience [label="熵权情景韧性\n吸收、适应、恢复"];
            planning [label="措施组合优化\n风险、减排、节能、投入产出"];
            model [label="三维罐区模型\n扩散线与影响范围"];
            report [label="综合结论报告\n推荐方案与决策链"];
            input -> qra;
            input -> resilience;
            input -> planning;
            qra -> model;
            resilience -> planning;
            planning -> model;
            qra -> report;
            resilience -> report;
            planning -> report;
            model -> report;
        }
        """,
        use_container_width=True,
    )
    st.markdown(
        "<div class='conclusion-note'><strong>决策链：</strong>统一参数输入驱动风险识别、韧性评估和措施组合优化；三维模型用于空间化展示，综合报告汇总关键指标和推荐方案。</div>",
        unsafe_allow_html=True,
    )
    render_sources((
        "系统架构为本软件的功能组织说明，不对应任何生产控制系统或真实项目网络拓扑。",
        "所有输入与输出均为脱敏、情景化展示数据，不用于实时监测、联锁控制或合规报告。",
    ))

with st.container(key="page_version"):
    st.header("版本与适用范围")
    render_module_description("说明系统版本、功能边界与数据使用要求，帮助使用者正确理解当前结果。")
    render_method("课堂研究与方案比选", "系统以简化 QRA、熵权评价和措施组合评分支持研究性比较，不替代专项工程分析、设计审查或安全评价。")
    version_columns = st.columns(3)
    version_columns[0].metric("系统版本", "研究展示版 1.0")
    version_columns[1].metric("适用对象", "脱敏港口罐区情景")
    version_columns[2].metric("数据状态", "取整与量级模糊")
    scope_frame = pd.DataFrame([
        {"范围": "支持内容", "说明": "风险识别、韧性评价、措施组合比选、三维空间化展示与决策报告"},
        {"范围": "不替代内容", "说明": "正式 QRA、工艺设计、设备选型、应急预案审批、投资决策与合规评价"},
        {"范围": "使用要求", "说明": "正式项目应以现场台账、专业软件计算、设计文件和审查意见为准"},
    ])
    st.dataframe(scope_frame, hide_index=True, width="stretch")
    st.info("本系统仅用于脱敏教学展示与方案研讨，输出结果不构成工程、财务或合规承诺。")
    render_sources((
        "版本信息与功能范围依据当前软件实现维护。",
        "正式应用时应补充项目授权、数据治理、模型校准、专家复核和审查记录。",
    ))