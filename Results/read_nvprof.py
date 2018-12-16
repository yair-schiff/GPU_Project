import math
import pandas as pd


def get_exponent(text):
    if text[-2:] == "ms":
        return 1
    elif text[-2:] == "us":
        return 3
    elif text[-2:] == "ns":
        return 6
    else:
        return -3


def main():
    with open("/Users/yairschiff/Development/CLion/GPU/Project/nvprof_results.txt", "r") as file:
        lines = file.readlines()
        data = []
        offset = 23
        for idx in range(len(lines)//offset):
            idx *= offset
            data_entry = {}
            # Get n
            line = [entry for entry in lines[idx].strip().split(" ") if entry != ""]
            data_entry["n"] = int(line[6])

            # Get total time
            line = [entry for entry in lines[idx+5].strip().split(" ") if entry != ""]
            data_entry["total_time"] = float(line[-2]) * 1000

            # Get GPU activities
            ## Get kernel time
            line = [entry for entry in lines[idx+9].strip().split(" ") if entry != ""]
            data_entry["kernel_time"] = float(line[1][:-2]) / math.pow(10, get_exponent(line[1]))
            ## CUDA memcpy
            line = [entry for entry in lines[idx+10].strip().split(" ") if entry != ""]
            data_entry["memcpy"] = float(line[1][:-2]) / math.pow(10, get_exponent(line[1]))
            line = [entry for entry in lines[idx+11].strip().split(" ") if entry != ""]
            data_entry["memcpy"] += float(line[1][:-2]) / math.pow(10, get_exponent(line[1]))
            for second_idx in range(8):
                line = [entry for entry in lines[idx+15+second_idx].strip().split(" ") if entry != ""]
                data_entry[line[-1]] = float(line[1][:-2]) / math.pow(10, get_exponent(line[1]))
            data.append(data_entry)
        df = pd.DataFrame(data)
        df = df[['n', 'total_time', 'kernel_time', 'memcpy', 'cudaMemcpy', 'cudaMalloc', 'cudaFree', 'cudaLaunch']]
        df.to_csv("/Users/yairschiff/Development/CLion/GPU/Project/nvprof_results.csv")


if __name__ == "__main__":
    main()
