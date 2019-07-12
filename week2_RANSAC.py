# coding: utf-8

"""
RANSAC Pseudo Code:

    iterations = 0
    best_model = null
    best_consensus_set = null
    best_error = 无穷大
    while ( iterations < k ) # k为迭代次数
    maybe_inliers = 从数据集中随机选择n个点
    maybe_model = 适合于maybe_inliers的模型参数
    consensus_set = maybe_inliers

    for ( 每个数据集中不属于maybe_inliers的点 ）
    if ( 如果点适合于maybe_model，且错误小于t ）
    将点添加到consensus_set
    if （ consensus_set中的元素数目大于d ）
    已经找到了好的模型，现在测试该模型到底有多好
    better_model = 适合于consensus_set中所有点的模型参数
    this_error = better_model究竟如何适合这些点的度量
    if ( this_error < best_error )
    我们发现了比以前好的模型，保存该模型直到更好的模型出现
    best_model =  better_model
    best_consensus_set = consensus_set
    best_error =  this_error
    增加迭代次数
    返回 best_model, best_consensus_set, best_error

"""
