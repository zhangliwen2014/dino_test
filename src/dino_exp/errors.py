class DinoError(Exception):
    """应用层统一异常；message 必须附带修复建议（NFR-4）。"""
