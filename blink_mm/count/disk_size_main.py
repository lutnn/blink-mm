import os
import os.path as osp
import argparse
import csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--bin-path")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    f = open(args.output_csv, 'w')
    writer = csv.writer(f)
    writer.writerow(["model", "bytes"])

    bin_path = osp.join(args.bin_path, "pixel4/1-threads")
    for file_name in os.listdir(bin_path):
        size = osp.getsize(osp.join(bin_path, file_name))
        writer.writerow([file_name, size])

    f.close()
