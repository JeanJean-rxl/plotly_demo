用plotly创建图的方式主要有三种：
    1. Plotly Express: 上层接口，一般的散点图、柱状图、饼图等都在这里定义
    2. Figure Factories: 上层接口，提供了特定的几种复杂图（部分被Plotly Express替代）
    3. Graph Objects: 底层接口，提供更灵活的figures, traces and layout

数据导入方式主要有三种：
    1. 基于pandas,从csv读取, 例如: df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_ebola.csv')
    2. 内建数据集plotly.data, 例如: df = px.data.iris()
    3. 也可以自定义矩阵or随机生成.

其他常用功能：
    1. plotly.colors: colorscales and utility functions
        【color-scales】https://plotly.com/python/builtin-colorscales/
        【rgb颜色查询对照表】https://blog.csdn.net/u010997144/article/details/52084386

    2. Subplots: helper function for layout out multi-plot figures
    3. I/O: low-level interface for displaying, reading and writing figures
