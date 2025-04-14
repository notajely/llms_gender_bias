import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None):
    """
    设置日志记录器
    
    Args:
        name (str): 日志记录器名称
        log_file (str, optional): 日志文件路径。如果为None，将自动生成
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    if log_file is None:
        # 如果没有指定日志文件，创建一个基于时间戳的日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join('logs', f'{name}_{timestamp}.log')
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建默认日志记录器
default_logger = setup_logger('bias_analysis') 