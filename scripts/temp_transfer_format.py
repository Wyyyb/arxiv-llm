import pickle5 as pickle
import numpy as np
import h5py
import json


def load_and_convert_corpus():
    corpus_base_path = "../embedded_corpus/corpus.0.pkl"
    output_path_hdf5 = "../embedded_corpus/corpus.0.h5"
    output_path_npz = "../embedded_corpus/corpus.0.npz"
    output_path_json = "../embedded_corpus/corpus.0.json"

    try:
        # 使用 pickle5 加载数据
        with open(corpus_base_path, "rb") as fi:
            data = pickle.load(fi)

        encoded, lookup_indices = data

        # 保存为 HDF5 格式
        with h5py.File(output_path_hdf5, 'w') as f:
            f.create_dataset('encoded', data=encoded)
            f.create_dataset('lookup_indices', data=lookup_indices)
        print(f"Data saved to {output_path_hdf5}")

        # 保存为 numpy 的 npz 格式
        np.savez_compressed(output_path_npz, encoded=encoded, lookup_indices=lookup_indices)
        print(f"Data saved to {output_path_npz}")

        # 保存为 JSON 格式 (注意：这可能会非常慢并占用大量空间，如果数据很大的话)
        encoded_list = encoded.tolist() if isinstance(encoded, np.ndarray) else encoded
        lookup_indices_list = lookup_indices.tolist() if isinstance(lookup_indices, np.ndarray) else lookup_indices
        with open(output_path_json, 'w') as f:
            json.dump({"encoded": encoded_list, "lookup_indices": lookup_indices_list}, f)
        print(f"Data saved to {output_path_json}")

        return encoded, lookup_indices

    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None


# 运行转换
encoded, lookup_indices = load_and_convert_corpus()