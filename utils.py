import os
import pandas as pd
from argparse import ArgumentParser

def get_data(root = "PrIMuS"):
    files = {"file": [], "png": [], "agnostic": [], "semantic": [], "pkg": []}
    
    for pkg in os.listdir(root):
        try:
            for file in os.listdir(f"{root}/{pkg}"):
                path = f"{root}/{pkg}/{file}"
                
                with open(f"{path}/{file}.agnostic") as f:
                    agnostic_data = f.read()
                    
                with open(f"{path}/{file}.semantic") as f:
                    semantic_data = f.read()
                
                files["file"].append(file)
                files["png"].append(f"{path}/{file}.png")
                files["agnostic"].append(agnostic_data)
                files["semantic"].append(semantic_data)
                files["pkg"].append(pkg)

        except Exception as e:
            print(e)
            break

    return files

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="utils", description="utilities")
    parser.add_argument('-f', '--func', choices=['data', 'count'])

    return parser

def count_vocab(root="PrIMuS"):
    pass

def main():
    args = get_parser().parse_args()
    match args.func:
        case "data":
            files = get_data()
            
            df = pd.DataFrame(files)
            df.to_csv("data.csv")


        case "count":
            count_vocab()
    
if __name__ == "__main__":
    main()
