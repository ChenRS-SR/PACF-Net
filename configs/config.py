

class CONFIG:
    
    NUM_CLASSES = 3 #每个参数的类别数
    TEST_SAMPLE = (3,6) # 测试集样本序号
    PARAM_LIST = ['flow_rate','feed_rate','z_offset','hotend']
    PARAM_THRESHOLDS = {
        "flow_rate":[90,110],
        "feed_rate":[80,120],
        "z_offset":[-0.05,0.10],
        "hotend":[190,215]
    }

    #模型参数
    INIT_FILTERS = 64 #初始卷积核数量
    BATCH_SIZE = 32 
    EPOCHS = 50
    LEARNING_RATE = 1e-3

    # 早停
    PATIENCE = 5  # 用于EarlyStopping
    
    #输出路径
    MODEL_SAVE_PATH = './saved_models'
    LOG_DIR = './logs'
    TRAIN_PLOT_DIR = './training_plots'  # 训练曲线保存路径
    EVAL_OUTPUT_DIR = './evaluation'     # 评估结果保存路径
    
    
config = CONFIG()


OCTOPRINT = {
    'url':'http://127.0.0.1:5000',
    'api_key':'UGjrS2T5n_48GF0YsWADx1EoTILjwn7ZkeWUfgGvW2Q',
    'username':'chenrs',
    'password':'cjn4085322'
}

# 热像仪温度映射配置（固定范围，铁红伪彩色）
# 用于可视化打印第一层温度分布
# 参考：铁红调色板 - 低温(黑/深蓝) -> 中温(紫/红) -> 高温(黄/白)
THERMAL_VISUALIZATION = {
    'temp_low': 26.0,      # 固定温度下限 (°C) -> 映射到调色板起点(黑/深蓝)
    'temp_high': 50.0,     # 固定温度上限 (°C) -> 映射到调色板终点(黄/白)
    'colormap': 'INFERNO',    # 默认调色板: 'INFERNO', 'IRON', 'JET', 'HOT'
}