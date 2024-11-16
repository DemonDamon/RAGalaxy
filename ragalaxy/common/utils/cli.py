from typing import Dict, Any, Optional
import click
import os
from pathlib import Path

class CLIUtils:
    """命令行工具类"""
    
    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """确认操作"""
        return click.confirm(message, default=default)
    
    @staticmethod
    def print_error(message: str):
        """打印错误信息"""
        click.secho(f"Error: {message}", fg="red")
    
    @staticmethod
    def print_success(message: str):
        """打印成功信息"""
        click.secho(f"Success: {message}", fg="green")
        
    @staticmethod
    def print_warning(message: str):
        """打印警告信息"""
        click.secho(f"Warning: {message}", fg="yellow")
        
    @staticmethod
    def print_info(message: str):
        """打印信息"""
        click.secho(message, fg="blue")