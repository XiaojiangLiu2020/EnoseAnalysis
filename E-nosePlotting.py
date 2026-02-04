import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import base64
import io
import os
import numpy as np
import sys
import webbrowser
import threading
import copy

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# --- PyInstaller 路径处理 ---
if getattr(sys, 'frozen', False):
    assets_path = os.path.join(sys._MEIPASS, 'assets')
else:
    assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


# --- 颜色生成函数 ---
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = int(360 * i / n)
        colors.append(f'hsl({hue}, 80%, 50%)')
    return colors


# --- 初始化应用 ---
app = dash.Dash(__name__, assets_folder=assets_path, suppress_callback_exceptions=True)
server = app.server

# --- 自定义 Plotly 模板 ---
custom_template = {
    "layout": go.Layout(
        font={"family": "Segoe UI, sans-serif", "color": "#333"},
        title_font={"size": 20, "color": "#111"},
        legend_title_font_color="#444",
        xaxis={"gridcolor": "#e5e5e5", "zerolinecolor": "#ddd", "linecolor": "#ddd"},
        yaxis={"gridcolor": "#e5e5e5", "zerolinecolor": "#ddd", "linecolor": "#ddd"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=True,
    )
}

# --- 应用布局 (已按要求重构) ---
app.layout = html.Div(id="app-container", children=[
    # --- 后台数据存储 ---
    dcc.Store(id='uploaded-files-store', data={}),
    dcc.Store(id='active-file-store', data=None),
    dcc.Store(id='labeled-data-store', data=[]),
    dcc.Store(id='temp-label-info-store', data={}),
    dcc.Store(id='interaction-mode-store', data='none'),  # 'none', 'labeling', 'baseline'
    dcc.Store(id='calibration-store', data={'applied': False}),
    dcc.Store(id='baseline-points-store', data=[]),
    dcc.Download(id="download-pca-data"),

    # --- 页面结构 ---
    html.Div(id="header", children=[html.H1("电子鼻数据分析与气味识别平台")]),

    html.Div(id="main-content", children=[
        # --- 左侧控制面板 ---
        html.Div(id="control-panel", children=[
            dcc.Tabs(id="control-panel-tabs", value='tab-ops-calib', className='custom-tabs-container', children=[

                # --- 标签页 1: 操作与校准 ---
                dcc.Tab(label='操作与校准', value='tab-ops-calib', className='custom-tab',
                        selected_className='custom-tab--selected', children=[
                        html.Div(className='tab-content', children=[
                            html.Div(className="control-card", children=[
                                html.H3("1. 文件管理"),
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(id="upload-text", children=["📂 拖放或点击上传 CSV/Excel 文件"]),
                                    multiple=True,
                                    accept='.csv,.xls,.xlsx'
                                ),
                                html.Hr(),
                                html.Label("选择活动文件进行分析:"),
                                dcc.Dropdown(id='file-selector-dropdown', placeholder="请先上传文件..."),
                                html.Div(id='uploaded-files-list', className='files-list-container')
                            ]),
                            html.Div(className="control-card", children=[
                                html.H3("2. 基线校准 (针对活动文件)"),
                                # *** 修改点：调整了此处的样式以修复下拉框宽度问题 ***
                                html.Div(className="control-group", children=[
                                    html.Label("校准算法:", style={'flex-basis': 'auto', 'align-self': 'center',
                                                                   'margin-right': '10px'}),
                                    dcc.Dropdown(
                                        id='calib-method',
                                        options=[
                                            {'label': '比值法 (R / R0)', 'value': 'div'},
                                            {'label': '差值法 (R - R0)', 'value': 'sub'},
                                            {'label': '反比值法 (1 - R/R0)', 'value': 'one_minus_div'},
                                        ],
                                        value='div',
                                        clearable=False,
                                        style={'flex': '1'}  # 确保下拉框填充剩余空间
                                    ),
                                ]),
                                html.Hr(style={'margin': '15px 0', 'borderTop': '1px dashed #ccc'}),
                                html.Label("方式一：固定范围平均", style={'fontWeight': 'bold', 'color': '#555'}),
                                html.Div(className="control-group", style={'gap': '5px', 'marginBottom': '5px'},
                                         children=[
                                             dcc.Input(id="calib-start", type="number", placeholder="Start", value=0,
                                                       className="half-width"),
                                             dcc.Input(id="calib-end", type="number", placeholder="End", value=10,
                                                       className="half-width"),
                                             html.Button("应用固定", id="apply-calib-constant-button", n_clicks=0,
                                                         style={'flex': '1'}),
                                         ]),
                                html.Hr(style={'margin': '15px 0', 'borderTop': '1px dashed #ccc'}),
                                html.Label("方式二：多点线性拟合 (漂移校准)",
                                           style={'fontWeight': 'bold', 'color': '#555'}),
                                html.Div(className="control-group", style={'flexDirection': 'column', 'gap': '10px'},
                                         children=[
                                             html.Button("1. 点击选择基线点", id="btn-select-baseline-points",
                                                         n_clicks=0),
                                             html.Div(style={'display': 'flex', 'gap': '10px', 'width': '100%'},
                                                      children=[
                                                          html.Button("2. 拟合并应用", id="apply-calib-linear-button",
                                                                      n_clicks=0, style={'flex': '1'}),
                                                          html.Button("清除点", id="clear-baseline-points-button",
                                                                      n_clicks=0, className='btn-secondary',
                                                                      style={'flex': '0.5'}),
                                                      ])
                                         ]),
                                html.Hr(style={'margin': '20px 0'}),
                                html.Button("重置为原始数据", id="reset-calib-button", n_clicks=0,
                                            className='btn-danger'),
                                html.Div(id='calib-status',
                                         style={'marginTop': '15px', 'fontSize': '0.85em', 'color': '#007bff',
                                                'whiteSpace': 'pre-wrap'})
                            ]),
                        ])
                    ]),

                # --- 标签页 2: 数据标记 ---
                dcc.Tab(label='数据标记', value='tab-labeling', className='custom-tab',
                        selected_className='custom-tab--selected', children=[
                        html.Div(className='tab-content', children=[
                            html.Div(className="control-card", children=[
                                html.H3("1. 数据标记"),
                                html.Button("开始标记", id="toggle-labeling-button", n_clicks=0),
                                html.Div(id='labeling-interface', style={'display': 'none'}, children=[
                                    html.P(id='temp-selection-info', style={'fontStyle': 'italic', 'color': '#555'}),
                                    html.Button('清除当前选择', id='clear-temp-selection-button', n_clicks=0,
                                                className='btn-secondary'),
                                    html.Hr(),
                                    dcc.Input(id='label-name-input', placeholder="输入标签名称 (例如: 苹果)",
                                              type='text'),
                                    html.Button('保存标签', id='save-label-button', n_clicks=0)
                                ]),
                                html.Button("清除所有标签", id="clear-labels-button", n_clicks=0,
                                            className="btn-danger", style={'marginTop': '10px'}),
                                html.H4("已标记数据列表", style={'marginTop': '20px', 'marginBottom': '10px'}),
                                html.Div(id='labeled-data-list-container', className='labeled-list')
                            ]),
                        ])
                    ]),

                # --- 标签页 3: 降维与分类 ---
                dcc.Tab(label='降维与分类', value='tab-analysis', className='custom-tab',
                        selected_className='custom-tab--selected', children=[
                        html.Div(className='tab-content', children=[
                            html.Div(className="control-card", children=[
                                html.H3("1. PCA 降维分析"),
                                html.Label("数据预处理方法:"),
                                dcc.RadioItems(
                                    id='pca-scaling-method-radio',
                                    options=[{'label': ' 标准化 (Standardization)', 'value': 'standard'},
                                             {'label': ' 归一化 (Normalization)', 'value': 'minmax'}],
                                    value='standard', labelStyle={'display': 'block'}
                                ),
                                html.Label("降维维度:", style={'marginTop': '15px'}),
                                dcc.RadioItems(
                                    id='pca-dimension-radio',
                                    options=[{'label': ' 2D', 'value': 2},
                                             {'label': ' 3D', 'value': 3}],
                                    value=2, labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                                ),
                                html.Button("生成/更新 PCA 图", id="generate-pca-button", n_clicks=0,
                                            style={'marginTop': '10px'}),
                                html.Button("下载PCA数据", id="btn-download-pca", n_clicks=0, className="btn-secondary",
                                            style={'marginTop': '10px'}),
                            ]),
                            html.Div(className="control-card", children=[
                                html.H3("2. SVM 决策边界"),
                                html.Label("核函数 (Kernel):"),
                                dcc.Dropdown(
                                    id='svm-kernel-select',
                                    options=[{'label': '线性核 (Linear)', 'value': 'linear'},
                                             {'label': '径向基核 (RBF)', 'value': 'rbf'},
                                             {'label': '多项式核 (Poly)', 'value': 'poly'}],
                                    value='rbf', clearable=False
                                ),
                                html.Div(className="control-group", style={'marginTop': '15px'}, children=[
                                    html.Label("正则化参数 (C):", className="half-width"),
                                    dcc.Input(id="svm-c-input", type="number", value=1.0, min=0.01, step=0.1,
                                              className="half-width"),
                                ]),
                                html.Div(id='svm-gamma-container', className="control-group", children=[
                                    html.Label("Gamma:", className="half-width"),
                                    dcc.Input(id="svm-gamma-input", type="text", value='scale',
                                              placeholder="e.g., scale, auto, 0.1", className="half-width"),
                                ]),
                                html.Div(id='svm-degree-container', className="control-group",
                                         style={'display': 'none'}, children=[
                                        html.Label("Degree:", className="half-width"),
                                        dcc.Input(id="svm-degree-input", type="number", value=3, min=1, step=1,
                                                  className="half-width"),
                                    ]),
                                html.Button("生成/更新 SVM 边界", id="draw-svm-button", n_clicks=0,
                                            style={'marginTop': '10px'}),
                                html.Div(id='svm-warning-message',
                                         style={'marginTop': '15px', 'fontSize': '0.8em', 'color': '#666',
                                                'backgroundColor': '#f0f0f0', 'padding': '8px', 'borderRadius': '4px'})
                            ]),
                        ])
                    ]),
            ]),
        ]),

        # --- 右侧图表与结果区域 ---
        html.Div(id="graph-container", children=[
            html.Div(className="graph-card", children=[
                html.H3("时间序列数据"),
                dcc.Graph(id="timeseries-plot", style={'flex-grow': '1', 'min-height': '0'}),
            ]),
            html.Div(className="graph-card", children=[
                html.H3("PCA 与 SVM 决策边界"),
                dcc.Graph(id="pca-plot", style={'flex-grow': '1', 'min-height': '0'}),
            ]),
        ]),
    ]),
])


# --- 回调函数 (无需修改) ---

# 1. 文件上传与管理
@app.callback(
    [Output('uploaded-files-store', 'data'),
     Output('active-file-store', 'data'),
     Output('uploaded-files-list', 'children'),
     Output('upload-text', 'children')],
    Input('upload-data', 'contents'),
    [State('upload-data', 'filename'),
     State('uploaded-files-store', 'data')],
    prevent_initial_call=True
)
def handle_file_upload(list_of_contents, list_of_names, existing_files_data):
    if not list_of_contents: return no_update
    new_files_data = existing_files_data.copy()
    newly_uploaded_names = []
    for contents, filename in zip(list_of_contents, list_of_names):
        if filename in new_files_data: continue
        try:
            _, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            if '.csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif any(ext in filename for ext in ['.xls', '.xlsx']):
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                continue
            df = df.reset_index(drop=True)
            # 存储原始数据和处理后数据
            new_files_data[filename] = {'original': df.to_json(orient='split'), 'processed': df.to_json(orient='split')}
            newly_uploaded_names.append(filename)
        except Exception as e:
            print(f"Error parsing {filename}: {e}");
            continue
    if not newly_uploaded_names: return no_update, no_update, no_update, "❌ 文件已存在或解析失败"
    file_list_items = [html.Div(f"✔️ {name}", className='file-item') for name in new_files_data.keys()]
    return new_files_data, newly_uploaded_names[0], file_list_items, f"✅ 成功上传 {len(newly_uploaded_names)} 个新文件"


# 2. 更新文件选择下拉菜单
@app.callback(
    [Output('file-selector-dropdown', 'options'),
     Output('file-selector-dropdown', 'value'),
     Output('file-selector-dropdown', 'disabled')],
    Input('uploaded-files-store', 'data'),
    State('active-file-store', 'data')
)
def update_file_selector(files_data, active_file):
    if not files_data: return [], None, True
    filenames = list(files_data.keys())
    options = [{'label': name, 'value': name} for name in filenames]
    current_active = active_file if active_file in filenames else filenames[0]
    return options, current_active, False


# 3. 切换活动文件 (并重置校准状态)
@app.callback(
    [Output('active-file-store', 'data', allow_duplicate=True),
     Output('calibration-store', 'data', allow_duplicate=True),
     Output('baseline-points-store', 'data', allow_duplicate=True)],
    Input('file-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def switch_active_file(selected_filename):
    if not selected_filename:
        return no_update, no_update, no_update
    # 切换文件时，重置校准状态和选点
    return selected_filename, {'applied': False}, []


# 4. 管理交互模式 (标记 vs 选点)
@app.callback(
    [Output('interaction-mode-store', 'data'),
     Output('toggle-labeling-button', 'children'),
     Output('labeling-interface', 'style'),
     Output('btn-select-baseline-points', 'children'),
     Output('temp-label-info-store', 'data', allow_duplicate=True),
     Output('baseline-points-store', 'data', allow_duplicate=True)],
    [Input('toggle-labeling-button', 'n_clicks'),
     Input('btn-select-baseline-points', 'n_clicks')],
    State('interaction-mode-store', 'data'),
    prevent_initial_call=True
)
def manage_interaction_mode(label_clicks, baseline_clicks, current_mode):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_mode = current_mode

    if trigger_id == 'toggle-labeling-button':
        new_mode = 'labeling' if current_mode != 'labeling' else 'none'
    elif trigger_id == 'btn-select-baseline-points':
        new_mode = 'baseline' if current_mode != 'baseline' else 'none'

    label_btn_text = "停止标记" if new_mode == 'labeling' else "开始标记"
    label_interface_style = {'display': 'block'} if new_mode == 'labeling' else 'none'
    baseline_btn_text = "停止选点" if new_mode == 'baseline' else "1. 点击选择基线点"

    # 切换模式时清空临时选择
    return new_mode, label_btn_text, label_interface_style, baseline_btn_text, {}, []


# 5. 更新校准参数存储
@app.callback(
    [Output('calibration-store', 'data'), Output('calib-status', 'children'),
     Output('baseline-points-store', 'data', allow_duplicate=True)],
    [Input('apply-calib-constant-button', 'n_clicks'),
     Input('apply-calib-linear-button', 'n_clicks'),
     Input('reset-calib-button', 'n_clicks'),
     Input('clear-baseline-points-button', 'n_clicks')],
    [State('calib-start', 'value'), State('calib-end', 'value'),
     State('calib-method', 'value'), State('baseline-points-store', 'data')],
    prevent_initial_call=True
)
def update_calibration_store(btn_constant, btn_linear, btn_reset, btn_clear, start, end, method, baseline_points):
    ctx = callback_context
    if not ctx.triggered: return no_update, no_update, no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-calib-button':
        return {'applied': False}, "状态: 已重置 (无校准)", []

    if trigger_id == 'clear-baseline-points-button':
        return no_update, "状态: 已清除选择点", []

    method_map = {'div': '比值法 (R/R0)', 'sub': '差值法 (R-R0)', 'one_minus_div': '反比值法 (1 - R/R0)'}
    method_name = method_map.get(method, '未知算法')

    if trigger_id == 'apply-calib-constant-button':
        if start is None or end is None or start >= end:
            return no_update, "错误: 起始行必须小于结束行", no_update
        return {'applied': True, 'type': 'constant', 'range': [start, end], 'method': method}, \
            f"状态: 已应用 [固定范围] 校准\n算法: {method_name}\n范围: 行 {start} - {end}", no_update

    if trigger_id == 'apply-calib-linear-button':
        if not baseline_points or len(baseline_points) < 2:
            return no_update, "错误: 线性拟合至少需要选择 2 个点", no_update
        return {'applied': True, 'type': 'linear', 'indices': sorted(baseline_points), 'method': method}, \
            f"状态: 已应用 [线性拟合] 校准\n算法: {method_name}\n拟合点数: {len(baseline_points)}", no_update

    return no_update, no_update, no_update


# 6. 应用高级基线校准到数据
@app.callback(
    Output('uploaded-files-store', 'data', allow_duplicate=True),
    [Input('calibration-store', 'data')],
    [State('active-file-store', 'data'), State('uploaded-files-store', 'data')],
    prevent_initial_call=True
)
def apply_advanced_calibration(calib_params, active_file, files_data):
    if not active_file or active_file not in files_data: return no_update

    files_data_copy = copy.deepcopy(files_data)
    original_df = pd.read_json(files_data_copy[active_file]['original'], orient='split')

    if not calib_params or not calib_params.get('applied'):
        # 如果取消校准，则恢复原始数据
        files_data_copy[active_file]['processed'] = files_data_copy[active_file]['original']
        return files_data_copy

    temp_data = original_df.copy()
    method = calib_params.get('method', 'div')
    calib_type = calib_params.get('type')
    numeric_cols = temp_data.select_dtypes(include=np.number).columns

    try:
        if calib_type == 'constant':
            c_start, c_end = int(calib_params['range'][0]), int(calib_params['range'][1])
            if 0 <= c_start < c_end <= len(temp_data):
                baseline_vals = temp_data.iloc[c_start:c_end][numeric_cols].mean()
                if method == 'div' or method == 'one_minus_div':
                    baseline_vals = baseline_vals.replace(0, 1e-9)

                if method == 'div':
                    temp_data[numeric_cols] /= baseline_vals
                elif method == 'sub':
                    temp_data[numeric_cols] -= baseline_vals
                elif method == 'one_minus_div':
                    temp_data[numeric_cols] = 1 - (temp_data[numeric_cols] / baseline_vals)

        elif calib_type == 'linear':
            indices = calib_params.get('indices', [])
            valid_indices = [i for i in indices if 0 <= i < len(temp_data)]
            if len(valid_indices) >= 2:
                X_fit = np.array(valid_indices)
                for col in numeric_cols:
                    Y_fit = temp_data[col].iloc[valid_indices].values
                    slope, intercept = np.polyfit(X_fit, Y_fit, 1)
                    baseline_curve = slope * temp_data.index + intercept

                    if method == 'div' or method == 'one_minus_div':
                        baseline_curve = np.where(np.abs(baseline_curve) < 1e-9, 1e-9, baseline_curve)

                    if method == 'div':
                        temp_data[col] /= baseline_curve
                    elif method == 'sub':
                        temp_data[col] -= baseline_curve
                    elif method == 'one_minus_div':
                        temp_data[col] = 1 - (temp_data[col] / baseline_curve)

        files_data_copy[active_file]['processed'] = temp_data.to_json(orient='split')
    except Exception as e:
        print(f"Error during advanced calibration: {e}")
        return no_update

    return files_data_copy


# 7. 处理图表点击 (合并了标签和基线选点)
@app.callback(
    [Output('temp-label-info-store', 'data', allow_duplicate=True),
     Output('baseline-points-store', 'data', allow_duplicate=True)],
    Input('timeseries-plot', 'clickData'),
    [State('interaction-mode-store', 'data'), State('active-file-store', 'data'),
     State('uploaded-files-store', 'data'), State('temp-label-info-store', 'data'),
     State('baseline-points-store', 'data')],
    prevent_initial_call=True
)
def handle_graph_click_combined(clickData, mode, active_file, files_data, temp_info, baseline_points):
    if not clickData or mode == 'none' or not active_file:
        return no_update, no_update

    index = clickData['points'][0]['x']

    if mode == 'labeling':
        if temp_info.get('file') and temp_info.get('file') != active_file: temp_info = {}
        if not temp_info: temp_info = {'file': active_file, 'points': []}
        if index in {p['index'] for p in temp_info['points']}: return no_update, no_update

        df = pd.read_json(files_data[active_file]['processed'], orient='split')
        if index < len(df):
            numeric_cols = df.select_dtypes(include=np.number).columns
            temp_info['points'].append({'index': index, 'data': df.loc[index, numeric_cols].tolist()})
            return temp_info, no_update

    elif mode == 'baseline':
        new_baseline_points = baseline_points if baseline_points else []
        if index not in new_baseline_points:
            new_baseline_points.append(index)
            new_baseline_points.sort()
        return no_update, new_baseline_points

    return no_update, no_update


# 8. 保存标签
@app.callback(
    [Output('labeled-data-store', 'data'), Output('temp-label-info-store', 'data', allow_duplicate=True),
     Output('label-name-input', 'value')],
    Input('save-label-button', 'n_clicks'),
    [State('label-name-input', 'value'), State('temp-label-info-store', 'data'), State('labeled-data-store', 'data')],
    prevent_initial_call=True
)
def save_label(n_clicks, label_name, temp_info, existing_labels):
    if not label_name or not temp_info or not temp_info.get('points'): return no_update, no_update, ''
    new_labels = [{'label': label_name, 'data': p['data'], 'file': temp_info['file'], 'index': p['index']} for p in
                  temp_info['points']]
    return existing_labels + new_labels, {}, ''


# 9. 清除所有标签
@app.callback(
    Output('labeled-data-store', 'data', allow_duplicate=True),
    Input('clear-labels-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_all_labels(n_clicks): return []


# 10. 更新时间序列图 (增加基线点可视化)
@app.callback(
    Output('timeseries-plot', 'figure'),
    [Input('active-file-store', 'data'),
     Input('uploaded-files-store', 'data'),
     Input('labeled-data-store', 'data'),
     Input('temp-label-info-store', 'data'),
     Input('baseline-points-store', 'data')]  # 新增输入
)
def update_timeseries_plot(active_file, files_data, labeled_data, temp_info, baseline_points):
    if not active_file or not files_data or active_file not in files_data:
        fig = go.Figure(layout=custom_template)
        fig.update_layout(title="请上传并选择一个文件", annotations=[
            {"text": "无数据显示", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return fig

    # 始终从 'processed' 读取数据进行显示
    df = pd.read_json(files_data[active_file]['processed'], orient='split')
    fig = px.line(df, x=df.index, y=df.select_dtypes(include=np.number).columns, template=custom_template)

    # 绘制已保存的标签
    labels_for_this_file = [label for label in labeled_data if label['file'] == active_file]
    for label in labels_for_this_file:
        fig.add_vline(x=label['index'], line_width=2, line_dash="dash", line_color="rgba(220, 53, 69, 0.8)",
                      annotation_text=label['label'], annotation_position="top", annotation_font_size=10)

    # 绘制临时选择的标签点
    if temp_info and temp_info.get('file') == active_file:
        for point in temp_info.get('points', []):
            fig.add_vline(x=point['index'], line_width=2, line_dash="dot", line_color="rgba(0, 123, 255, 0.9)")

    # 新增: 绘制用于基线拟合的选点
    if baseline_points:
        for x_idx in baseline_points:
            fig.add_vline(x=x_idx, line_width=2, line_dash="dashdot", line_color="#6f42c1")

    fig.update_layout(
        title=f"文件: {active_file}",
        xaxis_title="数据点索引 (Index)",
        yaxis_title="传感器响应值",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# 11. 更新已标记数据列表
@app.callback(
    Output('labeled-data-list-container', 'children'),
    Input('labeled-data-store', 'data')
)
def update_labeled_data_list(labeled_data):
    if not labeled_data: return html.P("暂无已标记的数据。", style={'textAlign': 'center', 'color': '#888'})
    header = [html.Thead(html.Tr([html.Th("#"), html.Th("标签"), html.Th("文件"), html.Th("索引")]))]
    body = [html.Tbody([
        html.Tr([html.Td(i + 1), html.Td(item['label']),
                 html.Td(item['file'], style={'fontSize': '0.8em', 'color': '#666'}), html.Td(item['index'])])
        for i, item in enumerate(labeled_data)
    ])]
    return html.Table(header + body, className="styled-table")


# 12. 生成PCA图和SVM边界
@app.callback(
    Output('pca-plot', 'figure'),
    [Input('generate-pca-button', 'n_clicks'), Input('draw-svm-button', 'n_clicks')],
    [State('labeled-data-store', 'data'), State('pca-scaling-method-radio', 'value'),
     State('pca-dimension-radio', 'value'), State('svm-kernel-select', 'value'),
     State('svm-c-input', 'value'), State('svm-gamma-input', 'value'), State('svm-degree-input', 'value')]
)
def update_pca_plot(pca_clicks, svm_clicks, labeled_data, scaling_method, n_components, svm_kernel, svm_c, svm_gamma,
                    svm_degree):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial load'

    if not labeled_data:
        fig = go.Figure(layout=custom_template)
        fig.update_layout(title="PCA 与 SVM", annotations=[
            {"text": "请先标记至少一个数据点", "xref": "paper", "yref": "paper", "showarrow": False,
             "font": {"size": 16}}])
        return fig

    df_labeled = pd.DataFrame(labeled_data)
    if df_labeled['data'].apply(len).nunique() > 1:
        fig = go.Figure(layout=custom_template)
        fig.update_layout(title="PCA 错误", annotations=[
            {"text": "错误：标记的数据维度不一致！\n请清除标签后重新标记。", "xref": "paper", "yref": "paper",
             "showarrow": False, "font": {"size": 16, "color": "red"}}])
        return fig

    X = np.array(df_labeled['data'].tolist())
    labels = df_labeled['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    unique_labels = le.classes_
    num_labels = len(unique_labels)

    if num_labels <= 10:
        color_sequence = px.colors.qualitative.Plotly
    elif num_labels <= 24:
        color_sequence = px.colors.qualitative.Light24
    else:
        color_sequence = generate_distinct_colors(num_labels)
    color_map = {label: color_sequence[i % len(color_sequence)] for i, label in enumerate(unique_labels)}

    if X.shape[0] < n_components:
        fig = go.Figure(layout=custom_template)
        fig.update_layout(title="PCA 与 SVM", annotations=[
            {"text": f"请标记至少 {n_components} 个数据点以进行 {n_components}D PCA", "xref": "paper", "yref": "paper",
             "showarrow": False, "font": {"size": 16}}])
        return fig

    scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    if n_components == 2:
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['label'] = labels
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='label', color_discrete_map=color_map, title="2D PCA 降维结果",
                         labels={'PC1': f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%})',
                                 'PC2': f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%})'}, template=custom_template)
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))

        if trigger_id == 'draw-svm-button' and len(unique_labels) >= 2:
            try:
                gamma_val = float(svm_gamma)
            except (ValueError, TypeError):
                gamma_val = svm_gamma
            model = SVC(kernel=svm_kernel, C=svm_c, gamma=gamma_val, degree=svm_degree, probability=True).fit(X_pca,
                                                                                                              y_encoded)
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            unique_z = np.unique(Z)
            colors_for_z = [color_sequence[i % len(color_sequence)] for i in unique_z]
            boundaries = np.linspace(0, 1, len(unique_z) + 1)
            discrete_colorscale = [[b, color] for i, color in enumerate(colors_for_z) for b in
                                   (boundaries[i], boundaries[i + 1])] if len(unique_z) > 1 else [[0, colors_for_z[0]],
                                                                                                  [1, colors_for_z[0]]]

            contour_trace = go.Contour(x=xx[0], y=yy[:, 0], z=Z, opacity=0.3, showscale=False, hoverinfo='none',
                                       name='SVM Boundary', line_width=0, colorscale=discrete_colorscale,
                                       zmin=np.min(y_encoded), zmax=np.max(y_encoded))
            fig.add_trace(contour_trace)
            fig.data = (fig.data[-1],) + fig.data[:-1]
            fig.update_layout(title_text=f"2D PCA with {svm_kernel.upper()} SVM Boundary")
    else:
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
        pca_df['label'] = labels
        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='label', color_discrete_map=color_map,
                            title="3D PCA 降维结果",
                            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                                    'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'}, template=custom_template)
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        if trigger_id == 'draw-svm-button' and len(unique_labels) == 2 and svm_kernel == 'linear':
            try:
                model = SVC(kernel='linear', C=svm_c).fit(X_pca, y_encoded)
                w, b = model.coef_[0], model.intercept_[0]
                x_min, x_max, y_min, y_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, X_pca[:, 1].min() - 1, X_pca[
                                                                                                                  :,
                                                                                                                  1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
                if w[2] != 0:
                    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
                    fig.add_trace(go.Surface(x=xx, y=yy, z=zz,
                                             colorscale=[[0, 'rgba(0,123,255,0.5)'], [1, 'rgba(0,123,255,0.5)']],
                                             showscale=False, name='SVM Plane', hoverinfo='none'))
                    fig.update_layout(title_text="3D PCA with Linear SVM Plane")
            except Exception as e:
                print(f"Error drawing 3D SVM plane: {e}")
    return fig


# 13. 按钮禁用状态管理
@app.callback(
    [Output('toggle-labeling-button', 'disabled'),
     Output('generate-pca-button', 'disabled'), Output('clear-labels-button', 'disabled'),
     Output('draw-svm-button', 'disabled'), Output('svm-kernel-select', 'disabled'),
     Output('svm-c-input', 'disabled'), Output('svm-gamma-input', 'disabled'),
     Output('svm-degree-input', 'disabled'), Output('btn-download-pca', 'disabled'),
     Output('apply-calib-constant-button', 'disabled'), Output('apply-calib-linear-button', 'disabled'),
     Output('reset-calib-button', 'disabled'), Output('btn-select-baseline-points', 'disabled')],
    [Input('active-file-store', 'data'), Input('labeled-data-store', 'data')]
)
def set_button_disabled_state(active_file, labeled_data):
    no_active_file = active_file is None
    no_labeled_data = not labeled_data
    svm_disabled = no_labeled_data or pd.DataFrame(labeled_data)['label'].nunique() < 2
    return (
        no_active_file, no_labeled_data, no_labeled_data,
        svm_disabled, svm_disabled, svm_disabled, svm_disabled, svm_disabled,
        no_labeled_data, no_active_file, no_active_file, no_active_file, no_active_file
    )


# 14. 清除临时选择
@app.callback(
    Output('temp-label-info-store', 'data', allow_duplicate=True),
    Input('clear-temp-selection-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_temporary_selection(n_clicks): return {}


# 15. 更新选择信息文本
@app.callback(
    Output('temp-selection-info', 'children'),
    Input('temp-label-info-store', 'data')
)
def update_selection_info_text(temp_info):
    num_points = len(temp_info.get('points', []))
    return f"已选择 {num_points} 个点进行标记。" if num_points > 0 else "请在图表中点击选择数据点。"


# 16. 动态显示/隐藏SVM参数
@app.callback(
    [Output('svm-gamma-container', 'style'), Output('svm-degree-container', 'style')],
    Input('svm-kernel-select', 'value')
)
def toggle_svm_params(kernel):
    gamma_style = {'display': 'flex'} if kernel in ['rbf', 'poly', 'sigmoid'] else {'display': 'none'}
    degree_style = {'display': 'flex'} if kernel == 'poly' else {'display': 'none'}
    return gamma_style, degree_style


# 17. 下载PCA数据回调
@app.callback(
    Output("download-pca-data", "data"),
    Input("btn-download-pca", "n_clicks"),
    [State('labeled-data-store', 'data'), State('pca-scaling-method-radio', 'value'),
     State('pca-dimension-radio', 'value')],
    prevent_initial_call=True
)
def download_pca_data(n_clicks, labeled_data, scaling_method, n_components):
    if not n_clicks or not labeled_data: return no_update
    df_labeled = pd.DataFrame(labeled_data)
    if df_labeled['data'].apply(len).nunique() > 1: return no_update
    X = np.array(df_labeled['data'].tolist())
    if X.shape[0] < n_components: return no_update

    scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    download_df = pd.DataFrame(
        {'original_index': df_labeled['index'], 'label': df_labeled['label'], 'source_file': df_labeled['file']})
    for i in range(n_components):
        download_df[f'PC{i + 1}'] = X_pca[:, i]
    return dcc.send_data_frame(download_df.to_csv, "pca_results.csv", index=False)


# 18. 更新SVM警告信息
@app.callback(
    Output('svm-warning-message', 'children'),
    [Input('pca-dimension-radio', 'value'), Input('svm-kernel-select', 'value'), Input('labeled-data-store', 'data')]
)
def update_svm_warning(n_components, svm_kernel, labeled_data):
    if not labeled_data: return "请先标记数据。"
    unique_labels = pd.DataFrame(labeled_data)['label'].nunique()
    if unique_labels < 2: return "注意：SVM边界需要至少两个不同的标签才能生成。"
    if n_components == 3 and svm_kernel != 'linear':
        return f"注意：3D模式下，SVM决策边界可视化仅支持'线性核'{'和2个标签' if unique_labels > 2 else ''}。"
    if unique_labels > 2 and svm_kernel == 'linear': return "注意：'线性核'SVM通常用于二分类问题。"
    return "SVM参数已就绪。"


# --- 运行应用的主入口 ---
if __name__ == "__main__":
    HOST, PORT = '127.0.0.1', 8050
    css_string = """
    html, body { font-family: Segoe UI, sans-serif; background-color: #f8f9fa; margin: 0; padding: 0; }
    #app-container { max-width: 1800px; margin: auto; padding: 20px; box-sizing: border-box; }
    #header { text-align: center; margin-bottom: 20px; } #header h1 { color: #333; }
    #main-content { display: flex; flex-direction: row; gap: 20px; align-items: flex-start; }
    #control-panel { flex: 0 0 400px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); padding: 0; }
    #graph-container { flex: 1; display: flex; flex-direction: column; gap: 20px; }
    .control-card { padding: 20px; padding-top: 0; }
    .control-card:first-child { padding-top: 20px; }
    .graph-card { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); padding: 20px; height: 450px; display: flex; flex-direction: column; }
    .control-card h3, .graph-card h3 { margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; color: #343a40; }
    .control-group { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
    .control-group label { font-weight: bold; font-size: 0.9em; margin-bottom: 0; flex-basis: 100%; }
    .control-group .half-width, .control-group > .Select, .control-group > input { flex: 1; min-width: 100px; }
    input[type=number], input[type=text], .Select-control { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
    #upload-data { border: 2px dashed #007bff; border-radius: 5px; padding: 20px; text-align: center; cursor: pointer; transition: background-color 0.2s; }
    #upload-data:hover { background-color: #e9f5ff; }
    #upload-text { color: #007bff; font-weight: bold; }
    .files-list-container { margin-top: 15px; max-height: 150px; overflow-y: auto; border: 1px solid #eee; padding: 10px; border-radius: 4px; }
    button { color: white; background-color: #007bff; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; transition: background-color 0.2s; width: 100%; box-sizing: border-box; font-weight: bold; margin-top: 5px; }
    button:hover:not(:disabled) { background-color: #0056b3; }
    button:disabled { background-color: #ccc !important; color: #666 !important; cursor: not-allowed; }
    button.btn-secondary { background-color: #6c757d; }
    button.btn-secondary:hover:not(:disabled) { background-color: #5a6268; }
    button.btn-danger { background-color: #dc3545; }
    button.btn-danger:hover:not(:disabled) { background-color: #c82333; }
    #labeling-interface { margin-top: 15px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9; }
    .labeled-list { max-height: 200px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 4px; }
    .styled-table { width: 100%; border-collapse: collapse; }
    .styled-table th, .styled-table td { padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }
    .styled-table th { background-color: #f8f9fa; font-size: 0.9em; }
    .custom-tabs-container { border-bottom: 1px solid #dee2e6; }
    .custom-tab { padding: 12px 16px; cursor: pointer; background-color: #f8f9fa; border: 1px solid transparent; border-top-left-radius: .25rem; border-top-right-radius: .25rem; color: #007bff; font-weight: 500; }
    .custom-tab--selected { color: #495057; background-color: #fff; border-color: #dee2e6 #dee2e6 #fff; border-bottom: 1px solid #fff; position: relative; top: 1px; }
    .tab-content { padding: 0; }
    """
    if not os.path.exists(assets_path): os.makedirs(assets_path)
    with open(os.path.join(assets_path, "style.css"), "w", encoding="utf-8") as f:
        f.write(css_string)
    threading.Timer(1, lambda: webbrowser.open_new(f"http://{HOST}:{PORT}")).start()
    app.run(host=HOST, port=PORT, debug=False)