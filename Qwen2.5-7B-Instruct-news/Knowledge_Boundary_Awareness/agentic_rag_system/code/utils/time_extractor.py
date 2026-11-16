# time_extractor.py
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
    """
    从问题中提取时间信息
    
    Args:
        question: 问题文本
        
    Returns:
        时间信息字典，只有当问题中有具体的年月或年月日时才返回时间范围
        如果只有年份或者没有时间信息，返回None
    """
    # 中文日期正则表达式
    date_patterns = [
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # 2023年1月15日 - 需要过滤
        r'(\d{4})年(\d{1,2})月',             # 2023年1月 - 需要过滤
        r'(\d{4})年',                         # 2023年 - 不需要过滤
        r'(\d{4})-(\d{1,2})-(\d{1,2})',      # 2023-01-15 - 需要过滤
        r'(\d{4})/(\d{1,2})/(\d{1,2})',      # 2023/01/15 - 需要过滤
        r'(\d{4})-(\d{1,2})',                # 2023-01 - 需要过滤
        r'(\d{4})/(\d{1,2})',                # 2023/01 - 需要过滤
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, question)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 3:  # 年月日 - 需要过滤
                    year, month, day = map(int, groups)
                    need_filter = True
                elif len(groups) == 2:  # 年月 - 需要过滤
                    year, month = map(int, groups)
                    day = 1
                    need_filter = True
                else:  # 只有年 - 不需要过滤
                    year = int(groups[0])
                    month = 1
                    day = 1
                    need_filter = False
                
                # 构建日期对象
                try:
                    extracted_date = datetime(year, month, day)
                except ValueError:
                    continue
                
                if need_filter:
                    # 计算前后1个月的范围
                    start_date = extracted_date - timedelta(days=30)  # 前1个月
                    end_date = extracted_date + timedelta(days=30)    # 后1个月
                    
                    # 转换为时间戳（秒）
                    start_timestamp = int(start_date.timestamp())
                    end_timestamp = int(end_date.timestamp())
                    
                    return {
                        "extracted_date": extracted_date,
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                        "start_date_str": start_date.strftime("%Y-%m-%d"),
                        "end_date_str": end_date.strftime("%Y-%m-%d"),
                        "has_specific_date": True
                    }
                else:
                    # 只有年份，不进行时间过滤
                    return {
                        "extracted_date": extracted_date,
                        "has_specific_date": False,
                        "year_only": True
                    }
                
            except Exception as e:
                continue
    
    # 检查相对时间表达 - 不进行时间过滤
    relative_time_patterns = {
        r'去年': -365,
        r'今年': 0,
        r'明年': 365,
        r'上个月': -30,
        r'这个月': 0,
        r'下个月': 30,
        r'上周': -7,
        r'本周': 0,
        r'下周': 7,
    }
    
    current_date = datetime.now()
    for pattern, days_offset in relative_time_patterns.items():
        if re.search(pattern, question):
            extracted_date = current_date + timedelta(days=days_offset)
            return {
                "extracted_date": extracted_date,
                "has_specific_date": False,
                "is_relative": True
            }
    
    return None