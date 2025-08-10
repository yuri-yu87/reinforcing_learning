#!/usr/bin/env python3
"""
自动化类文档生成工具

功能:
1. 分析Python类的方法使用情况
2. 生成标准化的文档
3. 创建可视化图表
4. 生成维护报告
"""

import ast
import os
import sys
import json
import re
from typing import Dict, List, Set, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class ClassAnalyzer:
    """类分析器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.class_info = {}
        self.method_usage = {}
        self.imports = set()
        
    def analyze_file(self) -> Dict:
        """分析Python文件"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        self._extract_imports(tree)
        self._extract_classes(tree)
        self._analyze_method_usage(content)
        
        return {
            'file_path': self.file_path,
            'classes': self.class_info,
            'method_usage': self.method_usage,
            'imports': list(self.imports)
        }
    
    def _extract_imports(self, tree: ast.AST):
        """提取导入信息"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    self.imports.add(f"{module}.{alias.name}")
    
    def _extract_classes(self, tree: ast.AST):
        """提取类信息"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                methods = []
                attributes = []
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'args': [arg.arg for arg in item.args.args],
                            'docstring': ast.get_docstring(item),
                            'line_number': item.lineno
                        })
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attributes.append(target.id)
                
                self.class_info[class_name] = {
                    'methods': methods,
                    'attributes': attributes,
                    'docstring': ast.get_docstring(node),
                    'line_count': len(node.body)
                }
    
    def _analyze_method_usage(self, content: str):
        """分析方法使用情况"""
        for class_name, class_data in self.class_info.items():
            for method in class_data['methods']:
                method_name = method['name']
                # 搜索方法调用
                pattern = rf'{method_name}\s*\('
                matches = re.findall(pattern, content)
                self.method_usage[method_name] = len(matches) - 1  # 减去定义本身

class DocumentationGenerator:
    """文档生成器"""
    
    def __init__(self, analysis_result: Dict):
        self.analysis = analysis_result
        self.output_dir = Path('docs/generated')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_class_documentation(self, class_name: str) -> str:
        """生成类文档"""
        if class_name not in self.analysis['classes']:
            return ""
        
        class_data = self.analysis['classes'][class_name]
        
        # 统计信息
        total_methods = len(class_data['methods'])
        used_methods = sum(1 for method in class_data['methods'] 
                          if self.analysis['method_usage'].get(method['name'], 0) > 0)
        usage_rate = (used_methods / total_methods * 100) if total_methods > 0 else 0
        
        # 生成文档
        doc = f"""# {class_name} 类文档

## 📊 概览

- **文件位置**: {self.analysis['file_path']}
- **总方法数**: {total_methods}
- **使用率**: {usage_rate:.1f}%
- **代码行数**: {class_data['line_count']}

## 📋 方法列表

"""
        
        # 方法表格
        doc += "| 方法名 | 参数 | 使用次数 | 状态 |\n"
        doc += "|--------|------|----------|------|\n"
        
        for method in class_data['methods']:
            method_name = method['name']
            args = ', '.join(method['args'][1:])  # 跳过self
            usage_count = self.analysis['method_usage'].get(method_name, 0)
            status = "✅ 使用中" if usage_count > 0 else "❌ 未使用"
            
            doc += f"| `{method_name}` | `{args}` | {usage_count} | {status} |\n"
        
        # 属性列表
        if class_data['attributes']:
            doc += f"\n## 🔧 属性\n\n"
            for attr in class_data['attributes']:
                doc += f"- `{attr}`\n"
        
        # 文档字符串
        if class_data['docstring']:
            doc += f"\n## 📝 类描述\n\n{class_data['docstring']}\n"
        
        return doc
    
    def generate_usage_chart(self, class_name: str):
        """生成使用情况图表"""
        class_data = self.analysis['classes'][class_name]
        
        method_names = [method['name'] for method in class_data['methods']]
        usage_counts = [self.analysis['method_usage'].get(name, 0) for name in method_names]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图
        bars = ax1.bar(method_names, usage_counts)
        ax1.set_title(f'{class_name} 方法使用次数')
        ax1.set_ylabel('调用次数')
        ax1.tick_params(axis='x', rotation=45)
        
        # 为使用中的方法着色
        for bar, count in zip(bars, usage_counts):
            if count > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # 饼图
        used_count = sum(1 for count in usage_counts if count > 0)
        unused_count = len(usage_counts) - used_count
        
        if used_count + unused_count > 0:
            ax2.pie([used_count, unused_count], 
                   labels=['使用中', '未使用'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%')
            ax2.set_title('方法使用率')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{class_name}_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_mermaid_diagram(self, class_name: str) -> str:
        """生成Mermaid类图"""
        class_data = self.analysis['classes'][class_name]
        
        mermaid = f"""```mermaid
classDiagram
    class {class_name} {{
"""
        
        # 添加方法
        for method in class_data['methods']:
            method_name = method['name']
            args = ', '.join(method['args'][1:])  # 跳过self
            usage_count = self.analysis['method_usage'].get(method_name, 0)
            status = "✅" if usage_count > 0 else "❌"
            
            mermaid += f"        {status} {method_name}({args})\n"
        
        mermaid += "    }\n```"
        
        return mermaid
    
    def generate_report(self) -> str:
        """生成综合报告"""
        report = f"""# 类分析报告

## 📊 总体统计

- **分析文件**: {self.analysis['file_path']}
- **类数量**: {len(self.analysis['classes'])}
- **总方法数**: {sum(len(c['methods']) for c in self.analysis['classes'].values())}
- **总导入数**: {len(self.analysis['imports'])}

## 📋 类详情

"""
        
        for class_name in self.analysis['classes']:
            class_data = self.analysis['classes'][class_name]
            total_methods = len(class_data['methods'])
            used_methods = sum(1 for method in class_data['methods'] 
                              if self.analysis['method_usage'].get(method['name'], 0) > 0)
            
            report += f"### {class_name}\n"
            report += f"- 方法数: {total_methods}\n"
            report += f"- 使用中: {used_methods}\n"
            report += f"- 使用率: {used_methods/total_methods*100:.1f}%\n\n"
        
        return report

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python generate_class_docs.py <python_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # 分析文件
    analyzer = ClassAnalyzer(file_path)
    analysis_result = analyzer.analyze_file()
    
    # 生成文档
    generator = DocumentationGenerator(analysis_result)
    
    # 为每个类生成文档
    for class_name in analysis_result['classes']:
        print(f"生成 {class_name} 的文档...")
        
        # 生成Markdown文档
        doc = generator.generate_class_documentation(class_name)
        with open(generator.output_dir / f'{class_name}_documentation.md', 'w', encoding='utf-8') as f:
            f.write(doc)
        
        # 生成图表
        generator.generate_usage_chart(class_name)
        
        # 生成Mermaid图
        mermaid = generator.generate_mermaid_diagram(class_name)
        with open(generator.output_dir / f'{class_name}_diagram.md', 'w', encoding='utf-8') as f:
            f.write(mermaid)
    
    # 生成综合报告
    report = generator.generate_report()
    with open(generator.output_dir / 'analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"文档已生成到 {generator.output_dir}")

if __name__ == "__main__":
    main()
