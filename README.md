# NINE

Our proposed a novel algorithm of `IM `use network embedding named as `NineIM`.



You need first generate the node influence matrix using `generate_node_matrix.py`



If you want to run NINE algorithm to get network embedding. Please run `Nine.py` file，the use application program interface like this:

```python
embeddings_ = getFinalEmbedding(database_name, args)
```

# NineIM

This `NineIM `algorithm based on the final out put of NINE algorithm.



The key of `NineIM `in how to selection top k nodes, in this paper we use like this:

```python
def get_topk_by_corr2(matrix, topk = 10):
    column_lst = [str(e) for e in list(range(len(matrix)))]
    print(len(column_lst))
    # 计算列表两两间的相关系数
    data_dict = {}  # 创建数据字典，为生成Dataframe做准备
    for col, gf_lst in zip(column_lst, matrix):
        data_dict[col] = gf_lst

    unstrtf_df = pd.DataFrame(data_dict)
    cor1 = unstrtf_df.T.corr()  # 计算相关系数，得到一个矩阵 , 按列求相关系数【df格式的数据，按照节点做关键字，就是列名】

    # 将一行或者每一列的累加，求平均值，然后作为该节点的score
    values = cor1.values
    resu = list(np.sum(values, axis=0))
    # 找每一行中的前topk个元素的下标
    _re = []
    for _, i in enumerate(column_lst):
        _re.append([i, resu[_]])

    a = np.array(_re)
    b = a[:, 1]
    index = np.lexsort((b,))
    sorted_node = a[index][:, 0]
    return sorted_node[-topk:]

```



# Postscript

If you have any suggestions or comments, you can contact me from my blog. 

Blog address: https://blog.csdn.net/qq_26460841

Thank you very much for your attention to our work !!!

